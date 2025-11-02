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
from .services.data_config import DataConfigStore

# ============================================================================
# 设备选择与初始化
# ============================================================================
# 从环境变量读取设备配置，默认使用 GPU 加速计算
# 可通过设置 UPS_DEVICE=cpu 强制使用 CPU 模式
_device_choice = os.getenv("UPS_DEVICE", "gpu").lower().strip()
if _device_choice not in {"gpu", "cpu"}:
    raise RuntimeError("UPS_DEVICE 仅支持 'gpu' 或 'cpu'")

if _device_choice == "gpu":
    # GPU 模式：检查 CUDA 可用性
    if not torch.cuda.is_available():
        raise RuntimeError("UPS_DEVICE=gpu 但未检测到可用 CUDA 设备。")
    DEVICE = torch.device("cuda")
    # 启用 cuDNN benchmark 模式以优化卷积性能
    torch.backends.cudnn.benchmark = True
else:
    # CPU 模式：适用于开发调试或无 GPU 环境
    DEVICE = torch.device("cpu")

# ============================================================================
# 数据层初始化
# ============================================================================
# 创建模拟实时 API：负责生成用户、商品、APP 和事件数据
# 数据将持久化到本地目录，支持离线快照和实时流式事件
mock_api = MockRealtimeAPI()

# 数据摄取服务：封装对 mock_api 的调用，提供统一的数据访问接口
data_ingestion_service = DataIngestionService(api=mock_api)

# 配置存储：持久化数据生成参数（用户数、商品数等）
# 配置文件采用 JSON 格式，存储在数据目录下
config_store = DataConfigStore(storage_path=mock_api.data_dir / "data_config.json")

# ============================================================================
# 规则引擎初始化
# ============================================================================
# 规则存储：持久化业务规则定义（如高价值用户判定条件）
rule_store = RuleStore(storage_path=mock_api.data_dir / "rules.json")

# 加载已保存的规则，如果没有则使用默认规则
stored_rules = rule_store.load()
rule_engine = RuleEngine(rules=stored_rules if stored_rules else None)

# 首次启动时将默认规则写入磁盘，确保规则持久化
if not stored_rules:
    rule_store.save(rule_engine.list_rules())
# ============================================================================
# 图神经网络层初始化
# ============================================================================
# 图构建器：将关系型数据（用户、商品、APP、事件）转换为异构图结构
# 生成 PyTorch Geometric 的 HeteroData 对象，包含节点特征和边索引
graph_builder = GraphBuilder(
    config=GraphBuildConfig(
        data_dir=mock_api.data_dir,           # 数据源目录
        file_format=mock_api.file_format,     # 文件格式（csv/parquet）
        device=str(DEVICE),                   # 目标计算设备
    )
)

# 构建初始异构图快照：作为增量学习的基线
# 该图包含所有节点类型（user, product, app）和边类型（purchased, used等）
initial_graph = graph_builder.build_graph_from_snapshot()

# 提取图的元数据：节点类型、边类型及其关系
metadata = initial_graph.metadata()

# 特征编码器：将原始节点特征（如用户年龄、商品价格）编码为固定维度向量
# 支持类别特征嵌入和数值特征归一化
feature_encoder = HeteroFeatureEncoder(graph_builder).to(DEVICE)
input_dims = feature_encoder.output_dims

# ============================================================================
# HGT 模型配置与训练
# ============================================================================
# HGT (Heterogeneous Graph Transformer) 模型配置
# 该模型能够处理异构图，学习不同类型节点和边的表示
model_config = HGTModelConfig(
    metadata=metadata,          # 图结构元信息（节点类型、边类型）
    input_dims=input_dims,      # 各节点类型的输入特征维度
    hidden_dim=128,             # 隐藏层维度
    out_dim=128,                # 输出嵌入维度
    num_layers=2,               # Transformer 层数
    num_heads=4,                # 多头注意力头数
    dropout=0.1,                # Dropout 比例，防止过拟合
)

# 实例化 HGT 模型并迁移到目标设备（GPU/CPU）
incremental_model = HGTModel(model_config).to(DEVICE)

# 模型训练器：封装训练逻辑，支持边预测任务的监督学习
trainer = HGTTrainer(
    model=incremental_model,
    feature_encoder=feature_encoder,
    device=DEVICE,
)

# 初始训练：在完整图上训练 2 个 epoch
# 这一步确保模型和特征编码器联合优化，建立合理的初始表示
trainer.train_on_graph(initial_graph, epochs=2)

# ============================================================================
# 增量学习与画像服务初始化
# ============================================================================
# 增量学习器：支持新事件到达时的在线嵌入更新
# 避免每次都重新训练整个图，提升效率
incremental_learner = IncrementalLearner(
    graph_builder=graph_builder,
    model=incremental_model,
    feature_encoder=feature_encoder,
)

# 使用初始图重置学习器状态
incremental_learner.reset_with_graph(initial_graph)

# 预计算所有节点的初始嵌入，缓存到内存中
incremental_learner.refresh_all_embeddings()

# 混合画像服务：结合规则引擎（符号化逻辑）和嵌入模型（神经网络）
# 融合两种方法的优势，提供更全面的用户画像
hybrid_service = HybridProfilingService(
    data_ingestion=data_ingestion_service,   # 数据源
    incremental_learner=incremental_learner, # 嵌入计算
    rule_engine=rule_engine,                 # 规则判定
    rule_store=rule_store,                   # 规则持久化
    device=DEVICE,
)

# 解释器服务：使用 SHAP 提供模型预测的可解释性
# 说明为什么某个用户被判定为高价值用户
explainer_service = ExplainerService(
    profiling_service=hybrid_service,
    background_user_ids=[],  # 背景样本，用于 SHAP 基线计算
)

# ============================================================================
# 系统编排与控制层
# ============================================================================
# 刷新编排器：协调全量/增量图刷新流程
# 支持定期重建图结构，更新模型训练
refresh_orchestrator = RefreshOrchestrator(
    graph_builder=graph_builder,
    incremental_learner=incremental_learner,
    trainer=trainer,
    hybrid_service=hybrid_service,
    explainer_service=explainer_service,
    data_ingestion=data_ingestion_service,
)

# 系统控制器：统一管理后台任务（数据摄取、增量更新等）
# 提供启动、暂停、关闭等生命周期管理接口
system_controller = SystemController(
    data_ingestion=data_ingestion_service,
    incremental_learner=incremental_learner,
    hybrid_service=hybrid_service,
    explainer_service=explainer_service,
    refresh_orchestrator=refresh_orchestrator,
    idle_sleep=1.0,  # 空闲时的休眠时间（秒）
)



def create_app():
    """创建并配置 Flask 应用实例。
    
    该函数负责：
    1. 创建 Flask 应用对象
    2. 将所有核心服务注入到 app.config 中，供路由处理函数访问
    3. 启动异步事件循环，处理后台任务
    4. 注册 API 路由蓝图
    5. 定义健康检查端点
    6. 注册进程退出清理钩子
    
    Returns:
        Flask: 配置完成的 Flask 应用实例
    """
    app = Flask(__name__)
    
    # ========================================================================
    # 依赖注入：将所有服务实例注册到 Flask 配置中
    # ========================================================================
    # 这些服务将在 API 端点中通过 current_app.config 访问
    app.config["data_ingestion_service"] = data_ingestion_service   # 数据摄取
    app.config["incremental_learner"] = incremental_learner         # 增量学习
    app.config["rule_engine"] = rule_engine                         # 规则引擎
    app.config["hybrid_service"] = hybrid_service                   # 混合画像服务
    app.config["explainer_service"] = explainer_service             # 解释器服务
    app.config["graph_builder"] = graph_builder                     # 图构建器
    app.config["feature_encoder"] = feature_encoder                 # 特征编码器
    app.config["hgt_trainer"] = trainer                             # HGT 训练器
    app.config["refresh_orchestrator"] = refresh_orchestrator       # 刷新编排器
    app.config["system_controller"] = system_controller             # 系统控制器
    app.config["config_store"] = config_store                       # 配置存储
    app.config["realtime_api"] = mock_api                           # 实时 API
    
    # ========================================================================
    # 异步事件循环：在后台线程中运行协程任务
    # ========================================================================
    runner = AsyncLoopRunner()
    runner.start()  # 启动事件循环线程
    app.config["async_runner"] = runner

    # ========================================================================
    # 路由注册：挂载画像相关的 API 端点
    # ========================================================================
    app.register_blueprint(profiling_blueprint.bp)

    @app.route("/health", methods=["GET"])
    def health_check():
        """健康检查端点。
        
        返回系统运行状态，包括：
        - 数据摄取队列状态
        - 增量学习循环状态
        - 设备信息（GPU/CPU）
        - CUDA 可用性
        
        Returns:
            JSON 响应，包含系统健康状态信息
        """
        # 异步获取数据摄取状态（待处理事件数、处理速率等）
        ingestion = runner.run(system_controller.get_ingestion_status())
        
        # 获取增量学习循环状态（运行中/暂停/已停止）
        loop_status = system_controller.get_loop_status()
        
        return jsonify({
            "status": "ok",                              # 总体状态
            "ingestion": ingestion,                      # 摄取队列状态
            "incremental_loop": loop_status,             # 学习循环状态
            "device": str(DEVICE),                       # 当前使用的设备
            "device_mode": _device_choice,               # 设备模式（gpu/cpu）
            "cuda_available": bool(torch.cuda.is_available()),  # CUDA 是否可用
        })

    # ========================================================================
    # 清理钩子：进程退出时优雅关闭所有服务
    # ========================================================================
    # 注册退出处理函数，确保资源正确释放
    def cleanup():
        """进程退出时的清理函数。
        
        执行顺序：
        1. 关闭系统控制器（停止数据摄取和增量学习）
        2. 关闭异步事件循环
        """
        runner.run(system_controller.shutdown())  # 同步等待系统关闭完成
        runner.shutdown()                          # 关闭事件循环线程
    
    atexit.register(cleanup)

    return app


# 导出 WSGI 变量，便于 gunicorn 等加载
app = create_app()

if __name__ == "__main__":
    # 本地调试运行
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
