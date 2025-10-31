"""系统控制器：集中管理后台任务与编排。"""
from __future__ import annotations

from typing import Any


class SystemController:
    def __init__(self, *, data_ingestion, incremental_learner, hybrid_service, explainer_service, refresh_orchestrator, idle_sleep: float = 1.0) -> None:
        self.data_ingestion = data_ingestion
        self.incremental_learner = incremental_learner
        self.hybrid_service = hybrid_service
        self.explainer_service = explainer_service
        self.refresh_orchestrator = refresh_orchestrator
        self.idle_sleep = idle_sleep

    async def get_overview(self) -> dict[str, Any]:
        ingest = await self.data_ingestion.status()
        loop = self.incremental_learner.get_status()
        return {
            "ingestion": {"running": ingest.running, "events_consumed": ingest.events_consumed},
            "incremental_loop": {"running": loop.running, "steps": loop.steps},
        }

    async def start_ingestion(self) -> dict[str, Any]:
        s = await self.data_ingestion.start()
        return {"running": s.running, "events_consumed": s.events_consumed}

    async def stop_ingestion(self) -> dict[str, Any]:
        s = await self.data_ingestion.stop()
        return {"running": s.running, "events_consumed": s.events_consumed}

    async def start_incremental_loop(self) -> dict[str, Any]:
        s = await self.incremental_learner.start_loop()
        return {"running": s.running, "steps": s.steps}

    async def stop_incremental_loop(self) -> dict[str, Any]:
        s = await self.incremental_learner.stop_loop()
        return {"running": s.running, "steps": s.steps}

    async def train_fusion_core(self, *, sample_size: int, epochs: int, lr: float, batch_size: int) -> dict[str, Any]:
        # 简化：直接回显配置
        return {"sample_size": sample_size, "epochs": epochs, "lr": lr, "batch_size": batch_size}

    async def refresh_rules(self) -> dict[str, str]:
        self.hybrid_service.refresh_rule_structure()
        return {"status": "rules refreshed"}

    async def clear_explainer_cache(self) -> dict[str, str]:
        return self.explainer_service.clear_cache()

    async def shutdown(self) -> dict[str, str]:
        await self.stop_ingestion()
        await self.stop_incremental_loop()
        return {"status": "stopped"}

    async def get_ingestion_status(self) -> dict[str, Any]:
        s = await self.data_ingestion.status()
        return {"running": s.running, "events_consumed": s.events_consumed}

    def get_loop_status(self) -> dict[str, Any]:
        s = self.incremental_learner.get_status()
        return {"running": s.running, "steps": s.steps}

    async def trigger_refresh(self, *, mode, scope, sample_ratio, retrain_epochs, fusion_epochs):
        m = await self.refresh_orchestrator.trigger_refresh(
            mode=mode,
            scope=scope,
            sample_ratio=sample_ratio,
            retrain_epochs=retrain_epochs,
            fusion_epochs=fusion_epochs,
        )
        return {
            "duration_seconds": m.duration_seconds,
            "sample_ratio": m.sample_ratio,
            "node_counts": m.node_counts,
            "retrain_loss": m.retrain_loss,
            "fusion_training": m.fusion_training,
        }


__all__ = ["SystemController"]
