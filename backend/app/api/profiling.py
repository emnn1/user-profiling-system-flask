"""画像相关的 API 路由定义。

该模块向外暴露所有画像相关接口,并串联后端多个子系统:

- 通过 :class:`~app.services.data_ingestion.DataIngestionService` 读取缓存画像与推荐;
- 使用 :class:`~app.services.hybrid_profiling_service.HybridProfilingService` 管理规则与嵌入融合;
- 由 :class:`~app.services.explainer.ExplainerService` 提供 SHAP 解释;
- 调用 :class:`~app.graph_services.graph_builder.GraphBuilder` 与
    :class:`~app.services.incremental_learner.IncrementalLearner` 支持全量刷新。

接口路径统一挂载在 ``/api/v1`` 前缀下,供前端 Streamlit 页面与第三方系统调用。
"""
from __future__ import annotations

import asyncio
import json
import queue
from dataclasses import asdict
from typing import Any

from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
from werkzeug.exceptions import NotFound, BadRequest
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..graph_services.graph_builder import GraphBuilder
from ..services.data_ingestion import DataIngestionService
from ..services.explainer import ExplanationResult, ExplainerService
from ..services.hybrid_profiling_service import HybridProfilingService, RuleEngine
from ..services.incremental_learner import IncrementalLearner
from ..services.refresh_orchestrator import GraphRefreshMode, GraphScope
from ..services.system_controller import SystemController
from ..services.data_config import DataGenerationConfig, DataConfigStore
from ..services.training_config import TrainingConfig

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


def _get_config_store():
    store = current_app.config.get("config_store")
    if store is None:
        raise RuntimeError("DataConfigStore 尚未初始化")
    return store


def _get_realtime_api():
    api = current_app.config.get("realtime_api")
    if api is None:
        raise RuntimeError("MockRealtimeAPI 尚未初始化")
    return api


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
    random_seed: int | None = Field(
        None,
        ge=0,
        le=2**32 - 1,
        description="可选随机种子，留空时使用系统默认值",
    )
    include_edge_timestamp: bool | None = Field(
        None,
        description="覆盖图构建时是否保留边时间戳，None 表示沿用当前配置",
    )
    retrain_epochs: int = Field(2, ge=1, le=50, description="HGT 重新训练轮次")
    fusion_epochs: int = Field(3, ge=1, le=100, description="融合核心训练轮次")
    temperature: float = Field(0.2, gt=0.0, le=5.0, description="对比学习温度系数")
    learning_rate: float | None = Field(None, gt=0.0, le=1.0, description="可选的训练学习率覆盖值")


class FusionTrainPayload(BaseModel):
    sample_size: int = Field(256, ge=32, le=10000, description="用于训练的样本数量")
    epochs: int = Field(3, ge=1, le=200, description="训练轮次")
    lr: float = Field(1e-3, gt=0, description="学习率")
    batch_size: int = Field(64, ge=8, le=1024, description="批大小")


class HGTTrainingPayload(BaseModel):
    epochs: int = Field(5, ge=1, le=200, description="HGT 训练轮次")
    train_ratio: float = Field(0.8, gt=0.0, lt=1.0, description="训练集边比例")
    val_ratio: float = Field(0.1, ge=0.0, lt=1.0, description="验证集边比例")
    negative_ratio: float = Field(1.0, ge=0.0, le=5.0, description="评估阶段负样本比例")
    temperature: float = Field(0.2, gt=0.0, le=5.0, description="对比损失温度系数")
    learning_rate: float | None = Field(None, gt=0.0, le=1.0, description="可选学习率覆盖")
    seed: int | None = Field(None, ge=0, description="随机种子，未指定时使用时间种子")
    
    # 训练模式配置
    training_mode: str = Field("full_graph", description="训练模式：full_graph 或 metis_sampling")
    metis_num_parts: int = Field(4, ge=2, le=100, description="METIS 分区数（仅 metis_sampling 模式有效）")
    metis_imbalance_factor: float = Field(0.01, ge=0.0, le=1.0, description="METIS 不平衡因子")
    metis_seed: int | None = Field(None, ge=0, description="METIS 随机种子")
    metis_recursive: bool = Field(True, description="METIS 是否使用递归二分法")
    metis_partition_id: int | None = Field(None, ge=0, description="指定使用的分区 ID（None 表示随机选择）")

    @model_validator(mode='after')
    def _check_ratios(self) -> 'HGTTrainingPayload':
        """验证训练集和验证集比例之和不超过 1.0。"""
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio 与 val_ratio 之和需小于 1.0，以预留测试集比例。")
        return self
    
    @model_validator(mode='after')
    def _check_training_mode(self) -> 'HGTTrainingPayload':
        """验证训练模式及相关参数的有效性。"""
        if self.training_mode not in ("full_graph", "metis_sampling"):
            raise ValueError("training_mode 必须是 'full_graph' 或 'metis_sampling'")
        
        # 验证 METIS 参数（仅在 metis_sampling 模式下）
        if self.training_mode == "metis_sampling":
            if self.metis_partition_id is not None and self.metis_partition_id >= self.metis_num_parts:
                raise ValueError(f"metis_partition_id 必须小于 metis_num_parts ({self.metis_num_parts})")
        
        return self


@bp.route("/operations/status", methods=["GET"])
def get_operations_status():
    controller = _get_system_controller()
    return jsonify(_runner().run(controller.get_overview()))


@bp.route("/operations/metrics", methods=["GET"])
def get_operations_metrics():
    controller = _get_system_controller()
    return jsonify(controller.get_runtime_metrics())


@bp.route("/operations/stream", methods=["GET"])
def stream_operations_metrics():
    controller = _get_system_controller()
    subscriber = controller.subscribe_metrics()

    def event_stream():
        try:
            initial_snapshot = controller.get_runtime_metrics()
            yield f"data: {json.dumps(initial_snapshot, ensure_ascii=False)}\n\n"
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            controller.unsubscribe_metrics(subscriber)

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


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


@bp.route("/operations/training/hgt", methods=["POST"])
def trigger_hgt_training():
    controller = _get_system_controller()
    try:
        data = request.get_json() or {}
        payload = HGTTrainingPayload(**data)
    except (TypeError, ValidationError) as exc:
        raise BadRequest(str(exc))

    # 构建训练配置
    training_config = TrainingConfig(
        mode=payload.training_mode,  # type: ignore[arg-type]
        metis_num_parts=payload.metis_num_parts,
        metis_imbalance_factor=payload.metis_imbalance_factor,
        metis_seed=payload.metis_seed,
        metis_recursive=payload.metis_recursive,
        metis_partition_id=payload.metis_partition_id,
    )

    summary = _runner().run(controller.run_hgt_training(
        train_ratio=payload.train_ratio,
        val_ratio=payload.val_ratio,
        negative_ratio=payload.negative_ratio,
        epochs=payload.epochs,
        temperature=payload.temperature,
        learning_rate=payload.learning_rate,
        seed=payload.seed,
        training_config=training_config,
    ))
    return jsonify({
        "message": "HGT 训练流程完成",
        "summary": summary,
    })


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
    except ValidationError as exc:
        raise BadRequest(str(exc))
    except ValueError as exc:
        message = str(exc)
        if "未找到规则" in message:
            raise NotFound(message)
        raise BadRequest(message)
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


@bp.route("/graph/statistics", methods=["GET"])
def get_graph_statistics():
    """获取当前图的统计信息。"""
    builder = _get_graph_builder()
    try:
        stats = builder.get_graph_statistics()
        return jsonify(stats)
    except RuntimeError as exc:
        raise NotFound(str(exc))


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
            random_seed=payload.random_seed,
            include_edge_timestamp=payload.include_edge_timestamp,
            retrain_epochs=payload.retrain_epochs,
            fusion_epochs=payload.fusion_epochs,
            temperature=payload.temperature,
            learning_rate=payload.learning_rate,
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
        "temperature": payload.temperature,
        "learning_rate": payload.learning_rate,
        "random_seed": payload.random_seed,
        "include_edge_timestamp": metrics.include_edge_timestamp,
    })


# ───────────────────────────────────────────────────────────────────────
# 数据配置管理端点
# ───────────────────────────────────────────────────────────────────────

@bp.route("/data/config", methods=["GET"])
def get_data_config():
    """获取当前数据生成配置。

    Returns:
        JSON 响应,包含当前配置的所有参数。
    """
    try:
        store = _get_config_store()
        config = store.load_config()
        return jsonify({
            "user_count": config.user_count,
            "product_count": config.product_count,
            "app_count": config.app_count,
            "avg_events_per_user": config.avg_events_per_user,
            "history_days": config.history_days,
            # 初始随机边参数
            "min_orders_per_user": config.min_orders_per_user,
            "max_orders_per_user": config.max_orders_per_user,
            "min_app_usages_per_user": config.min_app_usages_per_user,
            "max_app_usages_per_user": config.max_app_usages_per_user,
            "min_calls_per_user": config.min_calls_per_user,
            "max_calls_per_user": config.max_calls_per_user,
            "min_click_products_per_user": config.min_click_products_per_user,
            "max_click_products_per_user": config.max_click_products_per_user,
            "min_click_apps_per_user": config.min_click_apps_per_user,
            "max_click_apps_per_user": config.max_click_apps_per_user,
        })
    except Exception as exc:
        raise BadRequest(f"获取配置失败: {exc}")


class DataConfigUpdatePayload(BaseModel):
    """更新数据配置的请求体模型。"""

    user_count: int = Field(..., ge=10, le=100000, description="用户数量")
    product_count: int = Field(..., ge=5, le=1000, description="商品数量")
    app_count: int = Field(..., ge=5, le=500, description="APP数量")
    avg_events_per_user: int = Field(..., ge=1, le=1000, description="每用户平均事件数")
    history_days: int = Field(..., ge=1, le=365, description="历史数据天数")

    # 初始随机边（每用户）的数量范围（当 events 为空时用于图构建）
    min_orders_per_user: int = Field(1, ge=0, le=100, description="每用户最少订购边数")
    max_orders_per_user: int = Field(3, ge=0, le=100, description="每用户最多订购边数")
    min_app_usages_per_user: int = Field(1, ge=0, le=100, description="每用户最少APP使用边数")
    max_app_usages_per_user: int = Field(3, ge=0, le=100, description="每用户最多APP使用边数")
    min_calls_per_user: int = Field(0, ge=0, le=100, description="每用户最少通话边数")
    max_calls_per_user: int = Field(2, ge=0, le=100, description="每用户最多通话边数")
    min_click_products_per_user: int = Field(0, ge=0, le=100, description="每用户最少点击商品边数")
    max_click_products_per_user: int = Field(5, ge=0, le=100, description="每用户最多点击商品边数")
    min_click_apps_per_user: int = Field(0, ge=0, le=100, description="每用户最少点击APP边数")
    max_click_apps_per_user: int = Field(5, ge=0, le=100, description="每用户最多点击APP边数")


@bp.route("/data/config", methods=["POST"])
def update_data_config():
    """更新数据生成配置并保存。

    Request Body:
        JSON,包含要更新的配置参数。

    Returns:
        JSON 响应,确认配置已更新。
    """
    try:
        payload = DataConfigUpdatePayload(**request.json)
    except ValidationError as e:
        raise BadRequest(f"请求参数校验失败: {e}")

    try:
        store = _get_config_store()
        new_config = DataGenerationConfig(
            user_count=payload.user_count,
            product_count=payload.product_count,
            app_count=payload.app_count,
            avg_events_per_user=payload.avg_events_per_user,
            history_days=payload.history_days,
            min_orders_per_user=payload.min_orders_per_user,
            max_orders_per_user=payload.max_orders_per_user,
            min_app_usages_per_user=payload.min_app_usages_per_user,
            max_app_usages_per_user=payload.max_app_usages_per_user,
            min_calls_per_user=payload.min_calls_per_user,
            max_calls_per_user=payload.max_calls_per_user,
            min_click_products_per_user=payload.min_click_products_per_user,
            max_click_products_per_user=payload.max_click_products_per_user,
            min_click_apps_per_user=payload.min_click_apps_per_user,
            max_click_apps_per_user=payload.max_click_apps_per_user,
        )
        store.save_config(new_config)
        return jsonify({
            "message": "配置已更新",
            "config": {
                "user_count": new_config.user_count,
                "product_count": new_config.product_count,
                "app_count": new_config.app_count,
                "avg_events_per_user": new_config.avg_events_per_user,
                "history_days": new_config.history_days,
                "min_orders_per_user": new_config.min_orders_per_user,
                "max_orders_per_user": new_config.max_orders_per_user,
                "min_app_usages_per_user": new_config.min_app_usages_per_user,
                "max_app_usages_per_user": new_config.max_app_usages_per_user,
                "min_calls_per_user": new_config.min_calls_per_user,
                "max_calls_per_user": new_config.max_calls_per_user,
                "min_click_products_per_user": new_config.min_click_products_per_user,
                "max_click_products_per_user": new_config.max_click_products_per_user,
                "min_click_apps_per_user": new_config.min_click_apps_per_user,
                "max_click_apps_per_user": new_config.max_click_apps_per_user,
            }
        })
    except Exception as exc:
        raise BadRequest(f"更新配置失败: {exc}")


@bp.route("/data/regenerate", methods=["POST"])
def regenerate_data():
    """使用当前配置重新生成数据。

    该操作会清空现有数据并重新生成,耗时较长。

    Returns:
        JSON 响应,确认数据重新生成已启动。
    """
    try:
        store = _get_config_store()
        config = store.load_config()
        realtime_api = _get_realtime_api()
        
        # 关闭数据库连接
        if hasattr(realtime_api, '_engine') and realtime_api._engine:
            realtime_api._engine.dispose()
            realtime_api._engine = None
        
        # 清理全局 engine 缓存
        from ..data_source.mock_data_provider import _ENGINE_CACHE
        db_path = realtime_api._data_dir / "mock_data.db"
        if db_path in _ENGINE_CACHE:
            _ENGINE_CACHE[db_path].dispose()
            del _ENGINE_CACHE[db_path]
        
        # 强制垃圾回收，确保连接完全释放
        import gc
        gc.collect()
        
        # 稍作延迟，等待文件句柄完全释放
        import time
        time.sleep(0.5)
        
        # 清空现有数据
        import os
        from sqlalchemy import text
        
        # 使用 DROP TABLE 完全删除表，而不是 DELETE 清空数据
        # 这样 _ensure_seed_data() 会检测到表不存在并重新生成
        try:
            # 获取 engine（如果不存在会自动创建）
            engine = realtime_api._engine
            if engine is None:
                from ..data_source.mock_data_provider import _get_engine
                engine = _get_engine(realtime_api._data_dir)
                realtime_api._engine = engine
            
            # 按依赖顺序删除表（events 依赖其他表）
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE IF EXISTS events"))
                conn.execute(text("DROP TABLE IF EXISTS users"))
                conn.execute(text("DROP TABLE IF EXISTS products"))
                conn.execute(text("DROP TABLE IF EXISTS apps"))
                
        except Exception as e:
            # 如果表不存在等错误，忽略继续
            print(f"删除表时出现错误: {e}")
        
        # 更新实例参数
        realtime_api._user_count = config.user_count
        realtime_api._product_count = config.product_count
        realtime_api._app_count = config.app_count
        realtime_api._avg_events_per_user = config.avg_events_per_user
        realtime_api._history_days = config.history_days
        
        # 重新生成数据
        realtime_api._ensure_seed_data()
        
        # 清理并重置 engine，确保后续读取到新数据
        if hasattr(realtime_api, '_engine') and realtime_api._engine:
            realtime_api._engine.dispose()
        from ..data_source.mock_data_provider import _get_engine
        realtime_api._engine = _get_engine(realtime_api._data_dir)
        
        return jsonify({
            "message": "数据重新生成已完成",
            "config": {
                "user_count": config.user_count,
                "product_count": config.product_count,
                "app_count": config.app_count,
                "avg_events_per_user": config.avg_events_per_user,
                "history_days": config.history_days,
                "min_orders_per_user": config.min_orders_per_user,
                "max_orders_per_user": config.max_orders_per_user,
                "min_app_usages_per_user": config.min_app_usages_per_user,
                "max_app_usages_per_user": config.max_app_usages_per_user,
                "min_calls_per_user": config.min_calls_per_user,
                "max_calls_per_user": config.max_calls_per_user,
                "min_click_products_per_user": config.min_click_products_per_user,
                "max_click_products_per_user": config.max_click_products_per_user,
                "min_click_apps_per_user": config.min_click_apps_per_user,
                "max_click_apps_per_user": config.max_click_apps_per_user,
            }
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise BadRequest(f"重新生成数据失败: {exc}")