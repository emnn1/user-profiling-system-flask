"""Flask 应用入口。

本模块串联整个后端的核心组件：

- :mod:`app.data_source.mock_data_provider` 提供的 :class:`~app.data_source.mock_data_provider.MockRealtimeAPI`
    负责生成与缓存模拟数据；
- :mod:`app.graph_services.graph_builder` 中的 :class:`~app.graph_services.graph_builder.GraphBuilder`
    与 :class:`~app.graph_services.graph_builder.GraphBuildConfig` 负责构建异构图，
    为增量学习器与特征编码器提供结构元数据；
- :mod:`app.ml_models.feature_store` 与 :mod:`app.ml_models.hgt_model`
    共同定义了图神经网络的输入特征与模型结构；
- :mod:`app.services.data_ingestion`, :mod:`app.services.incremental_learner`,
    :mod:`app.services.hybrid_profiling_service`, :mod:`app.services.explainer`
    分别承担数据摄取、增量嵌入刷新、规则融合与可解释性输出。

启动流程会在 GPU 上初始化所有依赖、构建初始图结构，
确保健康检查、画像 API 等路由都能通过 :mod:`app.api.profiling` 暴露出去。
"""
from __future__ import annotations

import torch
import atexit
import os
from flask import Flask, jsonify
from .api import profiling as profiling_blueprint
from .core.async_runner import AsyncLoopRunner
from .data_source.mock_data_provider import MockRealtimeAPI
from .graph_services.graph_builder import GraphBuildConfig, GraphBuilder
from .ml_models.feature_store import HeteroFeatureEncoder
from .ml_models.hgt_model import HGTModel, HGTModelConfig
from .services.data_ingestion import DataIngestionService
from .services.explainer import ExplainerService
from .services.hybrid_profiling_service import HybridProfilingService, RuleEngine, RuleStore
from .services.incremental_learner import IncrementalLearner
from .services.model_trainer import HGTTrainer
from .services.refresh_orchestrator import RefreshOrchestrator
from .services.system_controller import SystemController

# 设备选择：默认使用 GPU，可通过环境变量 UPS_DEVICE=cpu 显式选择 CPU
_device_choice = os.getenv("UPS_DEVICE", "gpu").lower().strip()
if _device_choice not in {"gpu", "cpu"}:
    raise RuntimeError("UPS_DEVICE 仅支持 'gpu' 或 'cpu'")

if _device_choice == "gpu":
    if not torch.cuda.is_available():
        raise RuntimeError("UPS_DEVICE=gpu 但未检测到可用 CUDA 设备。")
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

# 数据摄取服务依赖的模拟实时 API，用于产出离线快照与实时事件
mock_api = MockRealtimeAPI()
data_ingestion_service = DataIngestionService(api=mock_api)

# 规则存储负责持久化运营侧定义的规则，并在无数据时写入默认规则
rule_store = RuleStore(storage_path=mock_api.data_dir / "rules.json")

# 规则引擎初始状态：若未找到持久化规则，则加载默认规则并回写磁盘
stored_rules = rule_store.load()
rule_engine = RuleEngine(rules=stored_rules if stored_rules else None)
if not stored_rules:
    rule_store.save(rule_engine.list_rules())
# 图构建器读取模拟数据生成 PyG ``HeteroData``，后续服务共享 node_mapping 等元信息
graph_builder = GraphBuilder(
    config=GraphBuildConfig(
        data_dir=mock_api.data_dir,
        file_format=mock_api.file_format,
        device=str(DEVICE),
    )
)
# 初始化阶段仍然构建一次快照，用作增量学习的基准
initial_graph = graph_builder.build_graph_from_snapshot()
metadata = initial_graph.metadata()
feature_encoder = HeteroFeatureEncoder(graph_builder).to(DEVICE)
input_dims = feature_encoder.output_dims
# HGT 模型结构参数：依据初始图自动匹配元信息与输入维度
model_config = HGTModelConfig(
    metadata=metadata,
    input_dims=input_dims,
    hidden_dim=128,
    out_dim=128,
    num_layers=2,
    num_heads=4,
    dropout=0.1,
)
incremental_model = HGTModel(model_config).to(DEVICE)
trainer = HGTTrainer(
    model=incremental_model,
    feature_encoder=feature_encoder,
    device=DEVICE,
)
# 初始阶段执行若干轮全图训练，确保编码器与模型经过联合优化
trainer.train_on_graph(initial_graph, epochs=2)

incremental_learner = IncrementalLearner(
    graph_builder=graph_builder,
    model=incremental_model,
    feature_encoder=feature_encoder,
)
incremental_learner.reset_with_graph(initial_graph)
incremental_learner.refresh_all_embeddings()
hybrid_service = HybridProfilingService(
    data_ingestion=data_ingestion_service,
    incremental_learner=incremental_learner,
    rule_engine=rule_engine,
    rule_store=rule_store,
    device=DEVICE,
)
explainer_service = ExplainerService(
    profiling_service=hybrid_service,
    background_user_ids=[],
)
refresh_orchestrator = RefreshOrchestrator(
    graph_builder=graph_builder,
    incremental_learner=incremental_learner,
    trainer=trainer,
    hybrid_service=hybrid_service,
    explainer_service=explainer_service,
    data_ingestion=data_ingestion_service,
)
system_controller = SystemController(
    data_ingestion=data_ingestion_service,
    incremental_learner=incremental_learner,
    hybrid_service=hybrid_service,
    explainer_service=explainer_service,
    refresh_orchestrator=refresh_orchestrator,
    idle_sleep=1.0,
)



def create_app():
    app = Flask(__name__)
    # 注入全局依赖到app.config
    app.config["data_ingestion_service"] = data_ingestion_service
    app.config["incremental_learner"] = incremental_learner
    app.config["rule_engine"] = rule_engine
    app.config["hybrid_service"] = hybrid_service
    app.config["explainer_service"] = explainer_service
    app.config["graph_builder"] = graph_builder
    app.config["feature_encoder"] = feature_encoder
    app.config["hgt_trainer"] = trainer
    app.config["refresh_orchestrator"] = refresh_orchestrator
    app.config["system_controller"] = system_controller
    # 启动持久异步事件循环
    runner = AsyncLoopRunner()
    runner.start()
    app.config["async_runner"] = runner

    # 注册蓝图
    app.register_blueprint(profiling_blueprint.bp)

    @app.route("/health", methods=["GET"])
    def health_check():
        """返回核心后台任务的健康状态与队列积压。"""
        ingestion = runner.run(system_controller.get_ingestion_status())
        loop_status = system_controller.get_loop_status()
        return jsonify({
            "status": "ok",
            "ingestion": ingestion,
            "incremental_loop": loop_status,
            "device": str(DEVICE),
            "device_mode": _device_choice,
            "cuda_available": bool(torch.cuda.is_available()),
        })

    # 进程退出时兜底清理
    atexit.register(lambda: (runner.run(system_controller.shutdown()), runner.shutdown()))

    return app


# 导出 WSGI 变量，便于 gunicorn 等加载
app = create_app()

if __name__ == "__main__":
    # 本地调试运行
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
