"""增量学习器（简化实现）。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..graph_services.graph_builder import GraphBuilder
from ..ml_models.hgt_model import HGTModel
from ..ml_models.feature_store import HeteroFeatureEncoder


@dataclass(slots=True)
class LoopStatus:
    running: bool
    steps: int


class IncrementalLearner:
    def __init__(self, *, graph_builder: GraphBuilder, model: HGTModel, feature_encoder: HeteroFeatureEncoder) -> None:
        self.graph_builder = graph_builder
        self.model = model
        self.encoder = feature_encoder
        self._running = False
        self._steps = 0
        self._task: asyncio.Task | None = None

    def reset_with_graph(self, hetero_graph) -> None:
        # 这里不维护内部状态，兼容接口
        self._steps = 0

    def refresh_all_embeddings(self) -> None:
        # 前向一次，作为占位
        x = self.encoder.build_features_for_graph(self.graph_builder.hetero_data)
        _ = self.model(x, getattr(self.graph_builder.hetero_data, "edge_index_dict", {}))

    async def start_loop(self) -> LoopStatus:
        if self._running:
            return LoopStatus(True, self._steps)
        self._running = True
        self._task = asyncio.create_task(self._loop())
        return LoopStatus(True, self._steps)

    async def stop_loop(self) -> LoopStatus:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return LoopStatus(False, self._steps)

    def get_status(self) -> LoopStatus:
        return LoopStatus(self._running, self._steps)

    async def _loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(1.0)
                self.refresh_all_embeddings()
                self._steps += 1
        except asyncio.CancelledError:
            pass


__all__ = ["IncrementalLearner", "LoopStatus"]
