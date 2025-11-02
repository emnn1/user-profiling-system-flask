"""数据摄取服务，负责连接模拟实时 API 并维护轻量级缓存。

模块协作关系：

- 依赖 :class:`~app.data_source.mock_data_provider.MockRealtimeAPI` 拉取模拟数据，
    同时维护 ``asyncio.Queue`` 供 :class:`~app.services.incremental_learner.IncrementalLearner`
    消费；
- 将用户画像基础信息提供给 :class:`~app.services.hybrid_profiling_service.HybridProfilingService`
    和 FastAPI 路由 (:mod:`app.api.profiling`)；
- 通过 ``pending_events`` 指标让前端 `frontend/app.py` 的健康检查页可视化队列积压情况。

主要职责包括：

1. 预拉取全量用户信息并构建缓存；
2. 异步监听实时事件流，更新用户画像与事件统计；
3. 向增量学习模块产出事件批次，并暴露画像/推荐接口。
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, AsyncIterator

from ..data_source.mock_data_provider import Event, EventType, MockRealtimeAPI, User


class DataIngestionService:
    """负责与 ``MockRealtimeAPI`` 交互并提供查询接口。"""

    def __init__(
        self,
        *,
        api: MockRealtimeAPI,
        preload_batch_size: int = 500,
        realtime_start_offset: timedelta = timedelta(minutes=5),
    ) -> None:
        """构造数据摄取服务实例。

        :param api: 模拟实时 API 客户端，隐藏数据生成细节；
        :param preload_batch_size: 初始化时每批拉取的用户数量；
        :param realtime_start_offset: 事件监听起始时间偏移量，避免重复消费历史事件。
        """
        self._api = api
        self._preload_batch_size = max(1, preload_batch_size)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._warm_lock = asyncio.Lock()
        self._event_task: asyncio.Task[None] | None = None
        self._user_profiles: dict[str, dict[str, Any]] = {}
        self._last_event_timestamp = datetime.utcnow() - realtime_start_offset

    async def start(self) -> None:
        """加载初始用户数据并启动事件监听任务。

        调用顺序：
        1. 通过 :meth:`_load_all_users` 预热缓存；
        2. 创建后台协程 :meth:`_stream_events` 监听实时事件。
        """

        if self._event_task is not None:
            return

        # 预先拉取所有用户画像，填充本地缓存
        await self._load_all_users()
        # 启动后台协程持续监听实时事件
        self._event_task = asyncio.create_task(self._stream_events())

    async def stop(self) -> None:
        """终止后台事件任务，确保事件循环收尾时释放资源。"""

        if self._event_task is None:
            return

        # 主动取消协程，并吞掉正常的取消异常
        self._event_task.cancel()
        try:
            await self._event_task
        except asyncio.CancelledError:  # pragma: no cover - 正常取消流程
            pass
        finally:
            self._event_task = None

    async def _load_all_users(self) -> None:
        """遍历分页接口缓存所有用户画像。"""
        page = 1
        while True:
            batch = await self._api.get_user_batch(self._preload_batch_size, page)
            if not batch:
                break
            async with self._lock:
                # 缓存批量用户的静态属性与行为统计容器
                for user in batch:
                    self._user_profiles.setdefault(
                        user.user_id,
                        {
                            "user": user,
                            "event_counts": Counter(),
                            "last_event_at": None,
                        },
                    )
            page += 1

    async def ensure_cache_warm(self) -> None:
        """确保在未启动摄取任务时也完成用户缓存预热。"""

        if self._user_profiles:
            return

        async with self._warm_lock:
            if self._user_profiles:
                return
            await self._load_all_users()

    async def _stream_events(self) -> None:
        """持续监听实时事件流，并交由处理函数更新缓存。"""
        since = self._last_event_timestamp
        event_iterator: AsyncIterator[Event] = self._api.get_new_events(since)

        try:
            # 顺序消费实时事件流，并交由处理函数更新缓存
            async for event in event_iterator:
                await self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - 背景任务异常时记录
            # 为示例简化，真实环境下应接入日志系统
            print(f"[DataIngestionService] 事件流处理异常: {exc}")
            raise

    async def _handle_event(self, event: Event) -> None:
        """处理单个事件，将其信息写入缓存与队列。"""
        async with self._lock:
            profile = self._user_profiles.get(event.user_id)
            if profile is None:
                # 如果是新用户，将其加入缓存
                new_user = User(
                    user_id=event.user_id,
                    plan_type="未知套餐",
                    monthly_fee=0.0,
                    user_level="未知",
                    tenure_months=0,
                    device_brand="未知",
                )
                profile = {
                    "user": new_user,
                    "event_counts": Counter(),
                    "last_event_at": None,
                }
                self._user_profiles[event.user_id] = profile

            # 更新基础事件统计，用于推荐与规则评估
            profile["event_counts"][event.event_type.value] += 1
            if event.event_type is EventType.CALL and event.target_user_id:
                profile["event_counts"]["通话对象"] += 1
            profile["last_event_at"] = event.timestamp

            self._last_event_timestamp = max(self._last_event_timestamp, event.timestamp)

        # 异步队列记录事件，供增量学习模块消费
        await self._queue.put(event)

    async def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """返回指定用户的画像字典，包含基础属性与行为统计。"""
        async with self._lock:
            profile = self._user_profiles.get(user_id)
            if profile is None:
                return None

            # 将 dataclass 转换为基础字典，以便 JSON 序列化
            user_dict = asdict(profile["user"])
            return {
                "user": user_dict,
                "event_counts": dict(profile["event_counts"]),
                "last_event_at": profile["last_event_at"].isoformat()
                if profile["last_event_at"]
                else None,
            }

    async def get_recommendations(self, user_id: str) -> dict[str, Any] | None:
        """基于缓存规则生成简单策略推荐。"""
        profile = await self.get_user_profile(user_id)
        if profile is None:
            return None

        user = profile["user"]
        monthly_fee = user.get("monthly_fee", 0.0) or 0.0
        event_counts = profile.get("event_counts", {})

        recommendations = []
        # 根据简单业务规则组合策略建议
        if monthly_fee < 100:
            recommendations.append("升级至 5G 畅享套餐")
        if event_counts.get(EventType.APP_USAGE.value, 0) > 5:
            recommendations.append("推荐大流量包")
        if event_counts.get(EventType.ORDER.value, 0) == 0:
            recommendations.append("推送首购优惠券")

        if not recommendations:
            recommendations.append("保持当前套餐，并关注阶段性优惠")

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat(),
        }

    @property
    def pending_events(self) -> int:
        """返回当前事件队列长度，供监控使用。"""
        return self._queue.qsize()

    async def drain_events(self, limit: int | None = None) -> list[Event]:
        """按照指定上限从队列中取出事件批次。"""
        drained: list[Event] = []
        while not self._queue.empty():
            if limit is not None and len(drained) >= limit:
                break
            # 使用非阻塞方式获取事件，避免协程切换开销
            drained.append(self._queue.get_nowait())
        return drained

    async def list_cached_user_ids(self, limit: int | None = None) -> list[str]:
        """返回内部缓存的用户 ID 列表，可选限制数量。"""
        async with self._lock:
            user_ids = list(self._user_profiles.keys())
        if limit is None or limit >= len(user_ids):
            return user_ids
        return user_ids[:limit]

    @property
    def is_running(self) -> bool:
        """指示后台事件流是否正在运行。"""
        return self._event_task is not None and not self._event_task.done()

    @property
    def last_event_timestamp(self) -> datetime | None:
        """返回最近一次事件时间戳。"""
        return self._last_event_timestamp

    async def cached_user_count(self) -> int:
        """返回当前缓存中的用户数量。"""
        async with self._lock:
            return len(self._user_profiles)


__all__ = ["DataIngestionService"]
