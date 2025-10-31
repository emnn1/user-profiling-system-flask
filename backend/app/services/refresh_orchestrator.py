"""集中管理全量图刷新与模型训练流程。"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .hybrid_profiling_service import HybridProfilingService
from .incremental_learner import IncrementalLearner
from .model_trainer import HGTTrainer
from ..graph_services.graph_builder import GraphBuilder
from .data_ingestion import DataIngestionService
from .explainer import ExplainerService


class GraphRefreshMode(str, Enum):
    """图刷新时的训练策略。"""

    FULL_RETRAIN = "full_retrain"
    EMBEDDING_ONLY = "embedding_only"


class GraphScope(str, Enum):
    """全量图重构范围。"""

    FULL = "full"
    SAMPLED = "sampled"


@dataclass(slots=True)
class RefreshMetrics:
    """记录刷新流程中的关键指标。"""

    duration_seconds: float
    sample_ratio: float
    retrain_loss: Optional[list[float]]
    fusion_training: Dict[str, Any]
    node_counts: Dict[str, int]


class RefreshOrchestrator:
    """封装全图刷新流程，协调图、模型与融合层更新。"""

    def __init__(
        self,
        *,
        graph_builder: GraphBuilder,
        incremental_learner: IncrementalLearner,
        trainer: HGTTrainer,
        hybrid_service: HybridProfilingService,
        explainer_service: ExplainerService,
        data_ingestion: DataIngestionService,
    ) -> None:
        self.graph_builder = graph_builder
        self.incremental_learner = incremental_learner
        self.trainer = trainer
        self.hybrid_service = hybrid_service
        self.explainer_service = explainer_service
        self.data_ingestion = data_ingestion

    async def refresh_graph(
        self,
        *,
        mode: GraphRefreshMode,
        scope: GraphScope,
        sample_ratio: float | None,
        retrain_epochs: int,
        fusion_epochs: int,
    ) -> RefreshMetrics:
        """根据指定策略执行全量图刷新，并返回指标。"""

        start = time.perf_counter()
        ratio = 1.0 if scope is GraphScope.FULL else max(0.1, min(sample_ratio or 0.5, 1.0))

        hetero_data = await asyncio.to_thread(
            self.graph_builder.build_graph_from_snapshot,
            sample_ratio=ratio,
            random_state=int(start * 1e6) % (2**32 - 1),
        )

        retrain_loss: Optional[list[float]] = None
        if mode is GraphRefreshMode.FULL_RETRAIN:
            retrain_loss = await asyncio.to_thread(
                self.trainer.train_on_graph,
                hetero_data,
                epochs=retrain_epochs,
            )

        self.incremental_learner.reset_with_graph(hetero_data)
        await asyncio.to_thread(self.incremental_learner.refresh_all_embeddings)

        if mode is GraphRefreshMode.FULL_RETRAIN:
            fusion_metrics = await self.hybrid_service.train_fusion_core(
                sample_size=512,
                epochs=fusion_epochs,
            )
        else:
            fusion_metrics = {"trained": False, "reason": "skipped by mode"}

        self.explainer_service.clear_cache()

        duration = time.perf_counter() - start
        node_counts = {
            node_type: len(mapping)
            for node_type, mapping in self.graph_builder.node_mapping.items()
        }

        return RefreshMetrics(
            duration_seconds=duration,
            sample_ratio=ratio,
            retrain_loss=retrain_loss,
            fusion_training=fusion_metrics,
            node_counts=node_counts,
        )


__all__ = [
    "RefreshOrchestrator",
    "GraphRefreshMode",
    "GraphScope",
    "RefreshMetrics",
]
