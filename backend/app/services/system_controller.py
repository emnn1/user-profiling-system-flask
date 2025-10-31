"""Centralized runtime controller for orchestrating backend workflows."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
from typing import Any, Dict, List, Optional

from .data_ingestion import DataIngestionService
from .explainer import ExplainerService
from .hybrid_profiling_service import HybridProfilingService
from .incremental_learner import IncrementalLearner
from .refresh_orchestrator import (
    GraphRefreshMode,
    GraphScope,
    RefreshMetrics,
    RefreshOrchestrator,
)


class SystemController:
    """Expose lifecycle management and manual control hooks for the system."""

    def __init__(
        self,
        *,
        data_ingestion: DataIngestionService,
        incremental_learner: IncrementalLearner,
        hybrid_service: HybridProfilingService,
        explainer_service: ExplainerService,
        refresh_orchestrator: RefreshOrchestrator,
        idle_sleep: float = 1.0,
    ) -> None:
        self.data_ingestion = data_ingestion
        self.incremental_learner = incremental_learner
        self.hybrid_service = hybrid_service
        self.explainer_service = explainer_service
        self.refresh_orchestrator = refresh_orchestrator
        self._idle_sleep = max(0.1, idle_sleep)

        self._loop_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

        self._ingestion_started_at: datetime | None = None
        self._loop_stats: Dict[str, Any] = {
            "running": False,
            "started_at": None,
            "stopped_at": None,
            "processed_batches": 0,
            "processed_events": 0,
            "last_batch_at": None,
        }
        self._task_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 数据摄取控制
    # ------------------------------------------------------------------
    async def start_ingestion(self) -> Dict[str, Any]:
        async with self._lock:
            if self.data_ingestion.is_running:
                return await self.get_ingestion_status()
            await self.data_ingestion.start()
            self._ingestion_started_at = datetime.utcnow()
        return await self.get_ingestion_status()

    async def stop_ingestion(self) -> Dict[str, Any]:
        async with self._lock:
            if not self.data_ingestion.is_running:
                return await self.get_ingestion_status()
            await self.data_ingestion.stop()
            self._ingestion_started_at = None
        return await self.get_ingestion_status()

    async def get_ingestion_status(self) -> Dict[str, Any]:
        user_count = await self.data_ingestion.cached_user_count()
        last_event = self.data_ingestion.last_event_timestamp
        return {
            "running": self.data_ingestion.is_running,
            "pending_events": self.data_ingestion.pending_events,
            "cached_users": user_count,
            "last_event_at": last_event.isoformat() if last_event else None,
            "started_at": self._ingestion_started_at.isoformat() if self._ingestion_started_at else None,
        }

    # ------------------------------------------------------------------
    # 增量学习循环
    # ------------------------------------------------------------------
    async def start_incremental_loop(self) -> Dict[str, Any]:
        async with self._lock:
            if self._loop_task is not None and not self._loop_task.done():
                return self.get_loop_status()
            self._loop_stats.update(
                {
                    "running": True,
                    "started_at": datetime.utcnow(),
                    "stopped_at": None,
                    "processed_batches": 0,
                    "processed_events": 0,
                    "last_batch_at": None,
                }
            )
            self._loop_task = asyncio.create_task(self._incremental_loop())
        return self.get_loop_status()

    async def stop_incremental_loop(self) -> Dict[str, Any]:
        async with self._lock:
            if self._loop_task is None:
                return self.get_loop_status()
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
            self._loop_stats["running"] = False
            self._loop_stats["stopped_at"] = datetime.utcnow()
        return self.get_loop_status()

    def get_loop_status(self) -> Dict[str, Any]:
        status = dict(self._loop_stats)
        for key in ("started_at", "stopped_at", "last_batch_at"):
            if isinstance(status.get(key), datetime):
                status[key] = status[key].isoformat()  # type: ignore[index]
        status["pending_events"] = self.data_ingestion.pending_events
        status["task_active"] = self._loop_task is not None and not self._loop_task.done()
        return status

    async def _incremental_loop(self) -> None:
        try:
            while True:
                events = await self.data_ingestion.drain_events(limit=256)
                if events:
                    self.incremental_learner.register_events(events)
                    self._loop_stats["processed_batches"] += 1
                    self._loop_stats["processed_events"] += len(events)
                    self._loop_stats["last_batch_at"] = datetime.utcnow()
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(self._idle_sleep)
        except asyncio.CancelledError:
            raise
        finally:
            self._loop_stats["running"] = False
            self._loop_stats["stopped_at"] = datetime.utcnow()

    # ------------------------------------------------------------------
    # 手动任务
    # ------------------------------------------------------------------
    async def trigger_refresh(
        self,
        *,
        mode: GraphRefreshMode,
        scope: GraphScope,
        sample_ratio: Optional[float],
        retrain_epochs: int,
        fusion_epochs: int,
    ) -> RefreshMetrics:
        metrics = await self.refresh_orchestrator.refresh_graph(
            mode=mode,
            scope=scope,
            sample_ratio=sample_ratio,
            retrain_epochs=retrain_epochs,
            fusion_epochs=fusion_epochs,
        )
        self._append_history(
            "refresh_graph",
            {
                "mode": mode.value,
                "scope": scope.value,
                "duration_seconds": metrics.duration_seconds,
                "sample_ratio": metrics.sample_ratio,
            },
        )
        return metrics

    async def train_fusion_core(
        self,
        *,
        sample_size: int = 256,
        epochs: int = 3,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        metrics = await self.hybrid_service.train_fusion_core(
            sample_size=sample_size,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        self._append_history("train_fusion", metrics)
        return metrics

    async def refresh_rules(self) -> Dict[str, Any]:
        self.hybrid_service.refresh_rule_structure()
        self.explainer_service.clear_cache()
        self._append_history("refresh_rules", {"rule_dim": self.hybrid_service.fusion_core.rule_dim})
        return {"message": "Rules refreshed and explainer cache cleared."}

    async def clear_explainer_cache(self) -> Dict[str, Any]:
        self.explainer_service.clear_cache()
        self._append_history("clear_explainer", {})
        return {"message": "Explainer cache cleared."}

    async def get_overview(self) -> Dict[str, Any]:
        ingestion = await self.get_ingestion_status()
        loop_status = self.get_loop_status()
        overview = {
            "ingestion": ingestion,
            "incremental_loop": loop_status,
            "history": self._task_history[-20:],
        }
        return overview

    def _append_history(self, kind: str, payload: Dict[str, Any]) -> None:
        entry = {
            "type": kind,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload,
        }
        self._task_history.append(entry)
        if len(self._task_history) > 50:
            self._task_history = self._task_history[-50:]

    async def shutdown(self) -> None:
        await self.stop_incremental_loop()
        await self.stop_ingestion()


__all__ = ["SystemController"]
