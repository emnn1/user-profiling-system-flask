"""图刷新与训练编排（简化实现）。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class GraphRefreshMode(str, Enum):
    EMBEDDING_ONLY = "embedding_only"
    RETRAIN_MODEL = "retrain_model"
    RETRAIN_AND_FUSION = "retrain_and_fusion"


class GraphScope(str, Enum):
    FULL = "full"
    SAMPLED = "sampled"


@dataclass(slots=True)
class RefreshMetrics:
    duration_seconds: float
    sample_ratio: float | None
    node_counts: dict[str, int]
    retrain_loss: list[float]
    fusion_training: dict[str, Any]


class RefreshOrchestrator:
    def __init__(self, *, graph_builder, incremental_learner, trainer, hybrid_service, explainer_service, data_ingestion) -> None:
        self.graph_builder = graph_builder
        self.incremental_learner = incremental_learner
        self.trainer = trainer
        self.hybrid_service = hybrid_service
        self.explainer_service = explainer_service
        self.data_ingestion = data_ingestion

    async def trigger_refresh(
        self,
        *,
        mode: GraphRefreshMode,
        scope: GraphScope,
        sample_ratio: float | None,
        retrain_epochs: int,
        fusion_epochs: int,
    ) -> RefreshMetrics:
        start = time.perf_counter()
        if scope == GraphScope.FULL:
            graph = self.graph_builder.build_graph_from_snapshot()
            ratio_used = None
        else:
            ratio_used = float(sample_ratio or 1.0)
            graph = self.graph_builder.build_graph_from_snapshot(sample_ratio=ratio_used)

        # 统计节点数
        counts = {
            nt: int(getattr(self.graph_builder.hetero_data[nt], "num_nodes", 0))
            for nt in ("user", "product", "app")
        }

        retrain_loss: list[float] = []
        fusion_training: dict[str, Any] = {"epochs": 0}

        if mode in (GraphRefreshMode.RETRAIN_MODEL, GraphRefreshMode.RETRAIN_AND_FUSION):
            retrain_loss = self.trainer.train_on_graph(graph, epochs=max(1, int(retrain_epochs)))
            self.incremental_learner.reset_with_graph(graph)
            self.incremental_learner.refresh_all_embeddings()

        if mode == GraphRefreshMode.RETRAIN_AND_FUSION:
            # 这里不做实际训练，仅返回配置占位
            fusion_training = {"epochs": int(fusion_epochs), "status": "ok"}

        dur = time.perf_counter() - start
        return RefreshMetrics(
            duration_seconds=dur,
            sample_ratio=ratio_used,
            node_counts=counts,
            retrain_loss=retrain_loss,
            fusion_training=fusion_training,
        )


__all__ = [
    "GraphRefreshMode",
    "GraphScope",
    "RefreshMetrics",
    "RefreshOrchestrator",
]
