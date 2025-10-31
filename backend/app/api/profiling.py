"""画像相关的 API 路由定义。

该模块向外暴露所有画像相关接口，并串联后端多个子系统：

- 通过 :class:`~app.services.data_ingestion.DataIngestionService` 读取缓存画像与推荐；
- 使用 :class:`~app.services.hybrid_profiling_service.HybridProfilingService` 管理规则与嵌入融合；
- 由 :class:`~app.services.explainer.ExplainerService` 提供 SHAP 解释；
- 调用 :class:`~app.graph_services.graph_builder.GraphBuilder` 与
    :class:`~app.services.incremental_learner.IncrementalLearner` 支持全量刷新。

接口路径统一挂载在 ``/api/v1`` 前缀下，供前端 Streamlit 页面与第三方系统调用。
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import NotFound, BadRequest
from pydantic import BaseModel, Field, ValidationError

from ..graph_services.graph_builder import GraphBuilder
from ..services.data_ingestion import DataIngestionService
from ..services.explainer import ExplanationResult, ExplainerService
from ..services.hybrid_profiling_service import HybridProfilingService, RuleEngine
from ..services.incremental_learner import IncrementalLearner
from ..services.refresh_orchestrator import GraphRefreshMode, GraphScope
from ..services.system_controller import SystemController

bp = Blueprint("profiling", __name__, url_prefix="/api/v1")


def _get_data_ingestion_service():
    service = current_app.config.get("data_ingestion_service")
    if service is None:
        raise RuntimeError("DataIngestionService 尚未初始化")
    return service


def _get_rule_engine():
    engine = current_app.config.get("rule_engine")
    if engine is None:
        raise RuntimeError("RuleEngine 尚未初始化")
    return engine


def _get_hybrid_service():
    service = current_app.config.get("hybrid_service")
    if service is None:
        raise RuntimeError("HybridProfilingService 尚未初始化")
    return service


def _get_incremental_learner():
    learner = current_app.config.get("incremental_learner")
    if learner is None:
        raise RuntimeError("IncrementalLearner 尚未初始化")
    return learner


def _get_graph_builder():
    builder = current_app.config.get("graph_builder")
    if builder is None:
        raise RuntimeError("GraphBuilder 尚未初始化")
    return builder


def _get_system_controller():
    controller = current_app.config.get("system_controller")
    if controller is None:
        raise RuntimeError("SystemController 尚未初始化")
    return controller


def _get_explainer_service():
    service = current_app.config.get("explainer_service")
    if service is None:
        raise RuntimeError("ExplainerService 尚未初始化")
    return service


def _get_feature_encoder():
    return current_app.config.get("feature_encoder")


def _runner():
    runner = current_app.config.get("async_runner")
    if runner is None:
        raise RuntimeError("AsyncLoopRunner 未初始化")
    return runner


class RuleCreatePayload(BaseModel):
    """新增规则接口的请求体模型。"""

    name: str = Field(..., description="规则名称")
    description: str = Field("", description="规则说明")
    weight: float = Field(1.0, description="规则贡献权重")
    condition: str = Field(..., description="Python 表达式形式的触发条件")


class RuleUpdatePayload(BaseModel):
    """更新规则接口的请求体模型。"""

    description: str | None = Field(None, description="新的规则说明")
    weight: float | None = Field(None, description="新的权重")
    condition: str | None = Field(None, description="新的触发条件")


class GraphRefreshPayload(BaseModel):
    """全量图刷新请求体。"""

    mode: GraphRefreshMode = Field(GraphRefreshMode.EMBEDDING_ONLY, description="刷新策略")
    graph_scope: GraphScope = Field(GraphScope.FULL, description="图构建范围")
    sample_ratio: float | None = Field(
        None,
        ge=0.1,
        le=1.0,
        description="采样比例，graph_scope 为 sampled 时生效",
    )
    retrain_epochs: int = Field(2, ge=1, le=50, description="HGT 重新训练轮次")
    fusion_epochs: int = Field(3, ge=1, le=100, description="融合核心训练轮次")


class FusionTrainPayload(BaseModel):
    sample_size: int = Field(256, ge=32, le=10000, description="用于训练的样本数量")
    epochs: int = Field(3, ge=1, le=200, description="训练轮次")
    lr: float = Field(1e-3, gt=0, description="学习率")
    batch_size: int = Field(64, ge=8, le=1024, description="批大小")


@bp.route("/operations/status", methods=["GET"])
def get_operations_status():
    controller = _get_system_controller()
    return jsonify(_runner().run(controller.get_overview()))


@bp.route("/operations/ingestion/start", methods=["POST"])
def start_ingestion():
    controller = _get_system_controller()
    status = _runner().run(controller.start_ingestion())
    return jsonify({"message": "数据摄取已启动", "status": status})


@bp.route("/operations/ingestion/stop", methods=["POST"])
def stop_ingestion():
    controller = _get_system_controller()
    status = _runner().run(controller.stop_ingestion())
    return jsonify({"message": "数据摄取已停止", "status": status})


@bp.route("/operations/incremental/start", methods=["POST"])
def start_incremental_loop():
    controller = _get_system_controller()
    status = _runner().run(controller.start_incremental_loop())
    return jsonify({"message": "增量循环已启动", "status": status})


@bp.route("/operations/incremental/stop", methods=["POST"])
def stop_incremental_loop():
    controller = _get_system_controller()
    status = _runner().run(controller.stop_incremental_loop())
    return jsonify({"message": "增量循环已停止", "status": status})


@bp.route("/operations/fusion/train", methods=["POST"])
def manual_train_fusion_core():
    controller = _get_system_controller()
    try:
        data = request.get_json()
        payload = FusionTrainPayload(**data)
    except (TypeError, ValidationError) as exc:
        raise BadRequest(str(exc))
    metrics = _runner().run(controller.train_fusion_core(
        sample_size=payload.sample_size,
        epochs=payload.epochs,
        lr=payload.lr,
        batch_size=payload.batch_size,
    ))
    return jsonify({"message": "融合核心训练完成", "metrics": metrics})


@bp.route("/operations/rules/refresh", methods=["POST"])
def refresh_rules():
    controller = _get_system_controller()
    return jsonify(_runner().run(controller.refresh_rules()))


@bp.route("/operations/explainer/clear", methods=["POST"])
def clear_explainer_cache():
    controller = _get_system_controller()
    return jsonify(_runner().run(controller.clear_explainer_cache()))


@bp.route("/operations/shutdown", methods=["POST"])
def shutdown_backend():
    """显式释放资源：停止后台任务并关闭异步事件循环。

    注意：调用后，本进程中的后端将不可再处理需要异步执行的请求，
    该端点仅用于容器/本地调试的优雅停机场景。
    """
    controller = _get_system_controller()
    runner = _runner()
    # 先优雅停止后台任务
    runner.run(controller.shutdown())
    # 再关闭异步循环
    runner.shutdown()
    return jsonify({"message": "backend shutdown complete"})


@bp.route("/user/<user_id>", methods=["GET"])
def get_user_profile(user_id):
    service = _get_data_ingestion_service()
    profile = _runner().run(service.get_user_profile(user_id))
    if profile is None:
        raise NotFound("未找到指定用户")
    return jsonify({"user_id": user_id, "profile": profile})


@bp.route("/recommendation/<user_id>", methods=["GET"])
def get_user_recommendation(user_id):
    service = _get_data_ingestion_service()
    recommendation = _runner().run(service.get_recommendations(user_id))
    if recommendation is None:
        raise NotFound("无法生成推荐，用户可能不存在")
    return jsonify(recommendation)


@bp.route("/explain/<user_id>", methods=["GET"])
def explain_user_profile(user_id):
    explainer = _get_explainer_service()
    result = _runner().run(explainer.explain(user_id))
    if result is None:
        raise NotFound("未找到可解释的用户画像")
    return jsonify(asdict(result))


@bp.route("/rules", methods=["GET"])
def list_rules():
    engine = _get_rule_engine()
    return jsonify({"rules": engine.list_rules()})


@bp.route("/rules", methods=["POST"])
def create_rule():
    engine = _get_rule_engine()
    hybrid_service = _get_hybrid_service()
    explainer = _get_explainer_service()
    try:
        data = request.get_json()
        payload = RuleCreatePayload(**data)
        engine.add_rule(
            name=payload.name,
            description=payload.description,
            weight=payload.weight,
            condition=payload.condition,
        )
    except (ValueError, ValidationError) as exc:
        raise BadRequest(str(exc))
    hybrid_service.refresh_rule_structure()
    explainer.clear_cache()
    return jsonify({"message": "规则已创建"}), 201


@bp.route("/rules/<rule_name>", methods=["PUT"])
def update_rule(rule_name):
    engine = _get_rule_engine()
    hybrid_service = _get_hybrid_service()
    explainer = _get_explainer_service()
    try:
        data = request.get_json()
        payload = RuleUpdatePayload(**data)
        engine.update_rule(
            rule_name,
            description=payload.description,
            weight=payload.weight,
            condition=payload.condition,
        )
    except (ValueError, ValidationError) as exc:
        raise NotFound(str(exc))
    hybrid_service.refresh_rule_structure()
    explainer.clear_cache()
    return jsonify({"message": "规则已更新"})


@bp.route("/rules/<rule_name>", methods=["DELETE"])
def delete_rule(rule_name):
    engine = _get_rule_engine()
    hybrid_service = _get_hybrid_service()
    explainer = _get_explainer_service()
    try:
        engine.delete_rule(rule_name)
    except ValueError as exc:
        raise NotFound(str(exc))
    hybrid_service.refresh_rule_structure()
    explainer.clear_cache()
    return '', 204


@bp.route("/graph/refresh", methods=["POST"])
def refresh_full_graph():
    controller = _get_system_controller()
    builder = _get_graph_builder()
    feature_encoder = _get_feature_encoder()
    try:
        data = request.get_json()
        payload = GraphRefreshPayload(**data)
        metrics = _runner().run(controller.trigger_refresh(
            mode=payload.mode,
            scope=payload.graph_scope,
            sample_ratio=payload.sample_ratio,
            retrain_epochs=payload.retrain_epochs,
            fusion_epochs=payload.fusion_epochs,
        ))
    except Exception as exc:
        raise BadRequest(f"刷新失败: {exc}")
    if feature_encoder is not None:
        for node_type, mapping in builder.node_mapping.items():
            for node_id in mapping.keys():
                feature_encoder.ensure_node(node_type, node_id)
    return jsonify({
        "message": "图刷新流程已完成",
        "duration_seconds": metrics.duration_seconds,
        "sample_ratio": metrics.sample_ratio,
        "node_counts": metrics.node_counts,
        "mode": payload.mode.value,
        "graph_scope": payload.graph_scope.value,
        "retrain_loss": metrics.retrain_loss,
        "fusion_training": metrics.fusion_training,
    })
