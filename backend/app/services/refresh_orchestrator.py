"""集中管理全量图刷新与模型训练流程。"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

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
    include_edge_timestamp: bool
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
        random_seed: int | None,
        include_edge_timestamp: bool | None,
        retrain_epochs: int,
        fusion_epochs: int,
        temperature: float,
        learning_rate: float | None,
        progress_cb: Callable[[Dict[str, Any]], None] | None = None,
    ) -> RefreshMetrics:
        """根据指定策略执行全量图刷新，并返回指标。"""

        start = time.perf_counter()
        ratio = 1.0 if scope is GraphScope.FULL else max(0.1, min(sample_ratio or 0.5, 1.0))
        effective_seed = random_seed if random_seed is not None else int(start * 1e6) % (2**32 - 1)
        if include_edge_timestamp is not None:
            self.graph_builder.config.include_edge_timestamp = include_edge_timestamp
        current_include_ts = self.graph_builder.config.include_edge_timestamp

        if progress_cb is not None:
            progress_cb(
                {
                    "channel": "refresh",
                    "event": "stage",
                    "stage": "graph_build",
                    "scope": scope.value,
                    "sample_ratio": ratio,
                }
            )

        hetero_data = await asyncio.to_thread(
            self.graph_builder.build_graph_from_snapshot,
            sample_ratio=ratio,
            random_state=effective_seed,
        )

        retrain_loss: Optional[list[float]] = None
        if mode is GraphRefreshMode.FULL_RETRAIN:
            if progress_cb is not None:
                progress_cb(
                    {
                        "channel": "refresh",
                        "event": "start",
                        "component": "hgt",
                        "total_epochs": retrain_epochs,
                        "parameters": {
                            "temperature": temperature,
                            "learning_rate": learning_rate,
                        },
                    }
                )

            def hgt_progress(event: Dict[str, Any]) -> None:
                if progress_cb is None:
                    return
                payload = dict(event)
                payload["channel"] = "refresh"
                payload["component"] = "hgt"
                progress_cb(payload)

            retrain_loss = await asyncio.to_thread(
                self.trainer.train_on_graph,
                hetero_data,
                epochs=retrain_epochs,
                temperature=temperature,
                learning_rate=learning_rate,
                progress_cb=hgt_progress,
            )
            if progress_cb is not None:
                final_loss = retrain_loss[-1] if retrain_loss else None
                progress_cb(
                    {
                        "channel": "refresh",
                        "event": "complete",
                        "component": "hgt",
                        "final_loss": final_loss,
                        "loss_history": retrain_loss or [],
                    }
                )

        self.incremental_learner.reset_with_graph(hetero_data)
        await asyncio.to_thread(self.incremental_learner.refresh_all_embeddings)

        if mode is GraphRefreshMode.FULL_RETRAIN:
            if progress_cb is not None:
                progress_cb(
                    {
                        "channel": "refresh",
                        "event": "start",
                        "component": "fusion",
                        "total_epochs": fusion_epochs,
                    }
                )

            def fusion_progress(event: Dict[str, Any]) -> None:
                if progress_cb is None:
                    return
                payload = dict(event)
                payload["channel"] = "refresh"
                payload["component"] = "fusion"
                progress_cb(payload)

            fusion_metrics = await self.hybrid_service.train_fusion_core(
                sample_size=512,
                epochs=fusion_epochs,
                progress_cb=fusion_progress,
            )
        else:
            fusion_metrics = {"trained": False, "reason": "skipped by mode"}
            if progress_cb is not None:
                progress_cb(
                    {
                        "channel": "refresh",
                        "event": "skip",
                        "component": "fusion",
                        "reason": "mode",
                    }
                )

        self.explainer_service.clear_cache()

        duration = time.perf_counter() - start
        node_counts = {
            node_type: len(mapping)
            for node_type, mapping in self.graph_builder.node_mapping.items()
        }

        return RefreshMetrics(
            duration_seconds=duration,
            sample_ratio=ratio,
            include_edge_timestamp=current_include_ts,
            retrain_loss=retrain_loss,
            fusion_training=fusion_metrics,
            node_counts=node_counts,
        )

    async def run_training_workflow(
        self,
        *,
        train_ratio: float,
        val_ratio: float,
        negative_ratio: float,
        epochs: int,
        temperature: float,
        learning_rate: float | None,
        seed: int | None,
        progress_cb: Callable[[Dict[str, Any]], None] | None = None,
        training_config: Any = None,
    ) -> Dict[str, Any]:
        """构建全量图并执行 HGT 自动化训练与评估。"""

        start = time.perf_counter()
        hetero_data = await asyncio.to_thread(
            self.graph_builder.build_graph_from_snapshot,
        )
        
        # 保存完整大图到磁盘
        graph_save_path = self.graph_builder.config.data_dir / "full_graph.pt"
        await asyncio.to_thread(
            self.graph_builder.save_graph,
            graph_save_path,
        )
        
        # 获取大图统计信息
        graph_stats = self.graph_builder.get_graph_statistics()

        if progress_cb is not None:
            progress_cb(
                {
                    "channel": "hgt_training",
                    "event": "stage",
                    "stage": "split_edges",
                    "graph_stats": graph_stats,
                }
            )

        training_summary = await asyncio.to_thread(
            self.trainer.run_automated_training,
            hetero_data,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            negative_ratio=negative_ratio,
            epochs=epochs,
            temperature=temperature,
            learning_rate=learning_rate,
            seed=seed,
            progress_cb=progress_cb,
            training_config=training_config,
        )

        self.incremental_learner.reset_with_graph(hetero_data)
        await asyncio.to_thread(self.incremental_learner.refresh_all_embeddings)
        self.explainer_service.clear_cache()

        node_counts = {
            node_type: len(mapping)
            for node_type, mapping in self.graph_builder.node_mapping.items()
        }

        training_summary.update(
            {
                "duration_seconds": time.perf_counter() - start,
                "node_counts": node_counts,
                "graph_save_path": str(graph_save_path),
                "graph_statistics": graph_stats,
            }
        )
        return training_summary


__all__ = [
    "RefreshOrchestrator",
    "GraphRefreshMode",
    "GraphScope",
    "RefreshMetrics",
]
