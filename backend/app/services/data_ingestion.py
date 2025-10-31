"""数据摄取与缓存服务。

- 负责从 MockRealtimeAPI 读取静态画像与产品信息；
- 异步后台可选消费事件流，仅做计数；
- 对外提供画像查询与推荐查询接口（简化实现）。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..data_source.mock_data_provider import MockRealtimeAPI


@dataclass(slots=True)
class IngestionStatus:
    running: bool
    events_consumed: int


class DataIngestionService:
    def __init__(self, api: MockRealtimeAPI) -> None:
        self._api = api
        # 启动时快照进内存，便于快速查询
        self._users_df = pd.read_csv(api.data_dir / f"users.{api.file_format}") if (api.data_dir / f"users.{api.file_format}").exists() else None
        self._products_df = pd.read_csv(api.data_dir / f"products.{api.file_format}") if (api.data_dir / f"products.{api.file_format}").exists() else None
        self._running = False
        self._events_consumed = 0
        self._task: asyncio.Task | None = None
        self._since = None

    async def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        df = self._users_df
        if df is None:
            return None
        mask = df["user_id"].astype(str) == str(user_id)
        if not mask.any():
            return None
        row = df[mask].iloc[0]
        return {
            "user_id": str(row["user_id"]),
            "plan_type": row["plan_type"],
            "monthly_fee": float(row["monthly_fee"]),
            "user_level": row["user_level"],
            "tenure_months": int(row["tenure_months"]),
            "device_brand": row["device_brand"],
        }

    async def get_recommendations(self, user_id: str) -> dict[str, Any] | None:
        # 简化：随机返回 3 个产品做演示
        if self._products_df is None:
            return None
        sample = self._products_df.sample(n=min(3, len(self._products_df)))
        recs = [
            {
                "product_id": str(r["product_id"]),
                "product_name": r["product_name"],
                "score": float(i + 1) / (len(sample) + 1),
            }
            for i, (_, r) in enumerate(sample.iterrows())
        ]
        return {"user_id": user_id, "recommendations": recs}

    async def start(self) -> IngestionStatus:
        if self._running:
            return IngestionStatus(True, self._events_consumed)
        self._running = True
        self._events_consumed = 0
        self._since = asyncio.get_event_loop().time()
        self._task = asyncio.create_task(self._consume_loop())
        return IngestionStatus(True, self._events_consumed)

    async def stop(self) -> IngestionStatus:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        return IngestionStatus(False, self._events_consumed)

    async def status(self) -> IngestionStatus:
        return IngestionStatus(self._running, self._events_consumed)

    async def _consume_loop(self) -> None:
        from datetime import datetime

        since = datetime.utcnow()
        try:
            async for _ in self._api.get_new_events(since):
                if not self._running:
                    break
                self._events_consumed += 1
        except asyncio.CancelledError:
            pass


__all__ = ["DataIngestionService", "IngestionStatus"]
