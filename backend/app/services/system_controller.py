"""Centralized runtime controller for orchestrating backend workflows."""
from __future__ import annotations

import asyncio
import queue
import threading
from contextlib import suppress
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

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
from .metrics_utils import capture_resource_snapshot


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

        self._metrics_lock = threading.Lock()
        self._metric_snapshot: Dict[str, Any] = {
            "refresh": {"status": "idle"},
            "fusion_training": {"status": "idle"},
            "hgt_training": {"status": "idle"},
            "ingestion": {},
            "incremental_loop": {},
        }
        self._metric_subscribers: List[queue.Queue] = []

    # ------------------------------------------------------------------
    # 数据摄取控制
    # ------------------------------------------------------------------
    async def start_ingestion(self) -> Dict[str, Any]:
        async with self._lock:
            if self.data_ingestion.is_running:
                status = await self.get_ingestion_status()
                self._update_simple_metric("ingestion", status, publish=True)
                return status
            await self.data_ingestion.start()
            self._ingestion_started_at = datetime.utcnow()
        status = await self.get_ingestion_status()
        self._update_simple_metric("ingestion", status, publish=True)
        return status

    async def stop_ingestion(self) -> Dict[str, Any]:
        async with self._lock:
            if not self.data_ingestion.is_running:
                status = await self.get_ingestion_status()
                self._update_simple_metric("ingestion", status, publish=True)
                return status
            await self.data_ingestion.stop()
            self._ingestion_started_at = None
        status = await self.get_ingestion_status()
        self._update_simple_metric("ingestion", status, publish=True)
        return status

    async def get_ingestion_status(self) -> Dict[str, Any]:
        user_count = await self.data_ingestion.cached_user_count()
        last_event = self.data_ingestion.last_event_timestamp
        status = {
            "running": self.data_ingestion.is_running,
            "pending_events": self.data_ingestion.pending_events,
            "cached_users": user_count,
            "last_event_at": last_event.isoformat() if last_event else None,
            "started_at": self._ingestion_started_at.isoformat() if self._ingestion_started_at else None,
        }
        self._update_simple_metric("ingestion", status)
        return status

    # ------------------------------------------------------------------
    # 增量学习循环
    # ------------------------------------------------------------------
    async def start_incremental_loop(self) -> Dict[str, Any]:
        async with self._lock:
            if self._loop_task is not None and not self._loop_task.done():
                status = self.get_loop_status()
                self._update_simple_metric("incremental_loop", status, publish=True)
                return status
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
        status = self.get_loop_status()
        self._update_simple_metric("incremental_loop", status, publish=True)
        return status

    async def stop_incremental_loop(self) -> Dict[str, Any]:
        async with self._lock:
            if self._loop_task is None:
                status = self.get_loop_status()
                self._update_simple_metric("incremental_loop", status, publish=True)
                return status
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
            self._loop_stats["running"] = False
            self._loop_stats["stopped_at"] = datetime.utcnow()
        status = self.get_loop_status()
        self._update_simple_metric("incremental_loop", status, publish=True)
        return status

    def get_loop_status(self) -> Dict[str, Any]:
        status = dict(self._loop_stats)
        for key in ("started_at", "stopped_at", "last_batch_at"):
            if isinstance(status.get(key), datetime):
                status[key] = status[key].isoformat()  # type: ignore[index]
        status["pending_events"] = self.data_ingestion.pending_events
        status["task_active"] = self._loop_task is not None and not self._loop_task.done()
        self._update_simple_metric("incremental_loop", status)
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
                    status = self.get_loop_status()
                    self._update_simple_metric("incremental_loop", status, publish=True)
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(self._idle_sleep)
        except asyncio.CancelledError:
            raise
        finally:
            self._loop_stats["running"] = False
            self._loop_stats["stopped_at"] = datetime.utcnow()
            status = self.get_loop_status()
            self._update_simple_metric("incremental_loop", status, publish=True)

    # ------------------------------------------------------------------
    # 手动任务
    # ------------------------------------------------------------------
    async def trigger_refresh(
        self,
        *,
        mode: GraphRefreshMode,
        scope: GraphScope,
        sample_ratio: Optional[float],
        random_seed: Optional[int],
        include_edge_timestamp: Optional[bool],
        retrain_epochs: int,
        fusion_epochs: int,
        temperature: float,
        learning_rate: Optional[float],
    ) -> RefreshMetrics:
        parameters = {
            "mode": mode.value,
            "scope": scope.value,
            "sample_ratio": sample_ratio,
            "random_seed": random_seed,
            "include_edge_timestamp": include_edge_timestamp,
            "retrain_epochs": retrain_epochs,
            "fusion_epochs": fusion_epochs,
            "temperature": temperature,
            "learning_rate": learning_rate,
        }
        self._handle_metric_event(
            {
                "channel": "refresh",
                "event": "start",
                "parameters": parameters,
                "mode": mode.value,
                "scope": scope.value,
            }
        )

        def progress_cb(event: Dict[str, Any]) -> None:
            if "channel" not in event:
                event["channel"] = "refresh"
            self._handle_metric_event(event)

        metrics = await self.refresh_orchestrator.refresh_graph(
            mode=mode,
            scope=scope,
            sample_ratio=sample_ratio,
            random_seed=random_seed,
            include_edge_timestamp=include_edge_timestamp,
            retrain_epochs=retrain_epochs,
            fusion_epochs=fusion_epochs,
            temperature=temperature,
            learning_rate=learning_rate,
            progress_cb=progress_cb,
        )
        self._handle_metric_event(
            {
                "channel": "refresh",
                "event": "complete",
                "summary": {
                    "duration_seconds": metrics.duration_seconds,
                    "sample_ratio": metrics.sample_ratio,
                    "node_counts": metrics.node_counts,
                },
            }
        )
        self._append_history(
            "refresh_graph",
            {
                "mode": mode.value,
                "scope": scope.value,
                "duration_seconds": metrics.duration_seconds,
                "sample_ratio": metrics.sample_ratio,
                "random_seed": random_seed,
                "include_edge_timestamp": metrics.include_edge_timestamp,
                "retrain_epochs": retrain_epochs,
                "fusion_epochs": fusion_epochs,
                "temperature": temperature,
                "learning_rate": learning_rate,
                "retrain_loss": metrics.retrain_loss,
                "fusion_training": metrics.fusion_training,
                "node_counts": metrics.node_counts,
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
        parameters = {
            "sample_size": sample_size,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
        }
        self._handle_metric_event(
            {
                "channel": "fusion_training",
                "event": "start",
                "parameters": parameters,
                "total_epochs": max(1, epochs),
            }
        )

        def progress_cb(event: Dict[str, Any]) -> None:
            event.setdefault("channel", "fusion_training")
            event.setdefault("parameters", parameters)
            self._handle_metric_event(event)

        metrics = await self.hybrid_service.train_fusion_core(
            sample_size=sample_size,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            progress_cb=progress_cb,
        )
        self._handle_metric_event(
            {
                "channel": "fusion_training",
                "event": "complete",
                "summary": metrics,
                "final_loss": metrics.get("final_loss"),
            }
        )
        self._append_history("train_fusion", metrics)
        return metrics

    async def run_hgt_training(
        self,
        *,
        train_ratio: float,
        val_ratio: float,
        negative_ratio: float,
        epochs: int,
        temperature: float,
        learning_rate: float | None,
        seed: int | None,
        training_config: Any = None,
    ) -> Dict[str, Any]:
        parameters = {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "negative_ratio": negative_ratio,
            "epochs": epochs,
            "temperature": temperature,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        
        # 添加训练配置信息到参数
        if training_config is not None:
            parameters["training_mode"] = training_config.mode
            if training_config.mode == "metis_sampling":
                parameters["metis_num_parts"] = training_config.metis_num_parts
                parameters["metis_imbalance_factor"] = training_config.metis_imbalance_factor
                parameters["metis_seed"] = training_config.metis_seed
                parameters["metis_recursive"] = training_config.metis_recursive
                parameters["metis_partition_id"] = training_config.metis_partition_id

        self._handle_metric_event(
            {
                "channel": "hgt_training",
                "event": "start",
                "parameters": parameters,
                "total_epochs": epochs,
            }
        )

        def progress_cb(event: Dict[str, Any]) -> None:
            event.setdefault("channel", "hgt_training")
            event.setdefault("parameters", parameters)
            self._handle_metric_event(event)

        summary = await self.refresh_orchestrator.run_training_workflow(
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
        loss_history = summary.get("loss_history")
        final_loss = loss_history[-1] if isinstance(loss_history, list) and loss_history else None
        self._handle_metric_event(
            {
                "channel": "hgt_training",
                "event": "complete",
                "summary": summary,
                "final_loss": final_loss,
            }
        )
        self._append_history("train_hgt", summary)
        return summary

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

    def get_runtime_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            snapshot = deepcopy(self._metric_snapshot)
        snapshot["resources"] = capture_resource_snapshot()
        snapshot["timestamp"] = datetime.utcnow().isoformat()
        return snapshot

    def subscribe_metrics(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue()
        with self._metrics_lock:
            self._metric_subscribers.append(subscriber)
        return subscriber

    def unsubscribe_metrics(self, subscriber: queue.Queue) -> None:
        with self._metrics_lock:
            if subscriber in self._metric_subscribers:
                self._metric_subscribers.remove(subscriber)

    # ------------------------------------------------------------------
    # Metrics aggregation helpers
    # ------------------------------------------------------------------
    def _broadcast_metric_event(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("timestamp", datetime.utcnow().isoformat())
        event["resources"] = capture_resource_snapshot()
        with self._metrics_lock:
            subscribers = list(self._metric_subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(deepcopy(event))
            except queue.Full:
                continue

    def _update_simple_metric(self, channel: str, payload: Dict[str, Any], *, publish: bool = False) -> None:
        with self._metrics_lock:
            state = dict(payload)
            state["updated_at"] = datetime.utcnow().isoformat()
            self._metric_snapshot[channel] = state
        if publish:
            self._broadcast_metric_event({"channel": channel, "payload": state})

    def _handle_metric_event(self, event: Dict[str, Any]) -> None:
        channel = event.get("channel")
        if not channel:
            return

        with self._metrics_lock:
            state = deepcopy(self._metric_snapshot.get(channel, {"status": "idle"}))
            if channel == "refresh":
                state = self._apply_refresh_event(state, event)
            elif channel in {"fusion_training", "hgt_training"}:
                state = self._apply_training_event(state, event)
            else:
                state.update(event.get("payload", {}))
            state["updated_at"] = event.get("timestamp", datetime.utcnow().isoformat())
            self._metric_snapshot[channel] = state
        self._broadcast_metric_event({"channel": channel, "payload": state})

    def _apply_refresh_event(self, state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        event_type = event.get("event")
        component = event.get("component")
        state.setdefault("hgt", {"loss_history": []})
        state.setdefault("fusion", {"loss_history": []})

        if event_type == "start":
            state["status"] = "running"
            state["mode"] = event.get("mode", state.get("mode"))
            state["scope"] = event.get("scope", state.get("scope"))
            state["parameters"] = event.get("parameters", state.get("parameters", {}))
            state["started_at"] = event.get("timestamp", datetime.utcnow().isoformat())
        elif event_type == "stage":
            state["stage"] = event.get("stage")
            state["status"] = state.get("status", "idle")
        elif event_type == "complete":
            state["status"] = "completed"
            state["completed_at"] = event.get("timestamp", datetime.utcnow().isoformat())
            state["summary"] = event.get("summary", state.get("summary", {}))
        elif event_type == "error":
            state["status"] = "error"
            state["error"] = event.get("error")

        if component == "hgt" and event_type in {"epoch", "complete", "start"}:
            hgt_state = state.setdefault("hgt", {"loss_history": []})
            if event_type == "start":
                hgt_state.update(
                    {
                        "loss_history": [],
                        "total_epochs": event.get("total_epochs"),
                        "samples": event.get("samples"),
                    }
                )
            if event_type == "epoch":
                history = hgt_state.setdefault("loss_history", [])
                loss = event.get("loss")
                if loss is not None:
                    history.append({
                        "epoch": event.get("epoch"),
                        "loss": loss,
                    })
                    if len(history) > 200:
                        del history[: len(history) - 200]
                hgt_state.update(
                    {
                        "current_epoch": event.get("epoch"),
                        "total_epochs": event.get("total_epochs", hgt_state.get("total_epochs")),
                        "last_loss": loss,
                        "duration_seconds": event.get("duration_seconds"),
                        "learning_rate": event.get("learning_rate"),
                    }
                )
            if event_type == "complete":
                hgt_state["final_loss"] = event.get("final_loss")

        if component == "fusion" and event_type in {"epoch", "complete", "start"}:
            fusion_state = state.setdefault("fusion", {"loss_history": []})
            if event_type == "start":
                fusion_state.update(
                    {
                        "loss_history": [],
                        "total_epochs": event.get("total_epochs"),
                        "samples": event.get("samples"),
                    }
                )
            if event_type == "epoch":
                history = fusion_state.setdefault("loss_history", [])
                loss = event.get("loss")
                if loss is not None:
                    history.append(
                        {
                            "epoch": event.get("epoch"),
                            "loss": loss,
                        }
                    )
                    if len(history) > 200:
                        del history[: len(history) - 200]
                fusion_state.update(
                    {
                        "current_epoch": event.get("epoch"),
                        "total_epochs": event.get("total_epochs", fusion_state.get("total_epochs")),
                        "last_loss": loss,
                        "duration_seconds": event.get("duration_seconds"),
                    }
                )
            if event_type == "complete":
                fusion_state["final_loss"] = event.get("final_loss")
        if component == "fusion" and event_type == "skip":
            fusion_state = state.setdefault("fusion", {})
            fusion_state.update(
                {
                    "status": "skipped",
                    "reason": event.get("reason"),
                }
            )

        return state

    def _apply_training_event(self, state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        event_type = event.get("event")

        if event_type == "start":
            state.update(
                {
                    "status": "running",
                    "loss_history": [],
                    "total_epochs": event.get("total_epochs"),
                    "started_at": event.get("timestamp", datetime.utcnow().isoformat()),
                    "parameters": event.get("parameters", {}),
                }
            )
        elif event_type == "stage":
            state["stage"] = event.get("stage")
            state.setdefault("status", "idle")
        elif event_type == "epoch":
            history = state.setdefault("loss_history", [])
            loss = event.get("loss")
            if loss is not None:
                history.append({"epoch": event.get("epoch"), "loss": loss})
                if len(history) > 200:
                    del history[: len(history) - 200]
            state.update(
                {
                    "status": "running",
                    "current_epoch": event.get("epoch"),
                    "total_epochs": event.get("total_epochs", state.get("total_epochs")),
                    "last_loss": loss,
                    "duration_seconds": event.get("duration_seconds"),
                }
            )
        elif event_type == "complete":
            state.update(
                {
                    "status": "completed",
                    "completed_at": event.get("timestamp", datetime.utcnow().isoformat()),
                    "final_loss": event.get("final_loss"),
                    "summary": event.get("summary", {}),
                }
            )
        elif event_type == "error":
            state.update(
                {
                    "status": "error",
                    "error": event.get("error"),
                }
            )
        return state

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
