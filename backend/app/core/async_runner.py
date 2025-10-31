"""持久异步事件循环执行器，用于在 Flask 环境中运行原有 async 代码。

该执行器在后台线程中维护一个长期存活的 asyncio 事件循环：
- run(coro): 同步等待协程完成并返回结果；
- submit(coro): 提交协程并返回 Future（可选同步等待或轮询结果）；
- shutdown(): 优雅停止事件循环并回收线程。

这使得原先依赖 FastAPI/uvicorn 的异步后台任务（create_task 等）可以在 Flask 中继续工作。
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Coroutine


class AsyncLoopRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _target() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._started.set()
            try:
                self._loop.run_forever()
            finally:
                pending = asyncio.all_tasks(self._loop)
                for t in pending:
                    t.cancel()
                # 允许任务处理取消
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()

        self._thread = threading.Thread(target=_target, name="AsyncLoopRunner", daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("AsyncLoopRunner 尚未启动")
        return self._loop

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        """将协程提交到后台事件循环，返回线程安全 Future。"""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """同步执行协程并返回结果。"""
        return self.submit(coro).result()

    def shutdown(self) -> None:
        if not self._loop:
            return
        # 停止事件循环
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None


__all__ = ["AsyncLoopRunner"]
