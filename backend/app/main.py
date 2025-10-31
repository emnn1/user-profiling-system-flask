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
mock_api = MockRealtimeAPI()\n