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
    """持久异步事件循环执行器。
    
    该类在独立后台线程中运行 asyncio 事件循环，
    使得 Flask 同步环境能够调用异步函数。
    
    主要用途：
    - 在 Flask 路由中执行异步服务方法
    - 支持长期运行的后台异步任务（如数据摄取、增量更新）
    - 提供同步/异步混合编程的桥梁
    
    Attributes:
        _thread (Thread | None): 事件循环所在的后台线程
        _loop (AbstractEventLoop | None): asyncio 事件循环实例
        _started (Event): 线程同步事件，用于等待事件循环就绪
    """
    
    def __init__(self) -> None:
        """初始化执行器。
        
        创建内部状态变量，但不立即启动事件循环。
        需要调用 start() 方法才会实际启动后台线程。
        """
        self._thread: threading.Thread | None = None        # 后台线程对象
        self._loop: asyncio.AbstractEventLoop | None = None # 事件循环实例
        self._started = threading.Event()                   # 启动同步信号

    def start(self) -> None:
        """启动后台事件循环线程。
        
        该方法会：
        1. 检查是否已经启动，避免重复启动
        2. 创建新线程并在其中初始化 asyncio 事件循环
        3. 启动事件循环的永久运行模式（run_forever）
        4. 阻塞等待事件循环就绪（最多 5 秒）
        
        Note:
            线程设置为 daemon=True，进程退出时会自动终止
            如果需要优雅关闭，应显式调用 shutdown() 方法
        """
        # 如果线程已启动且存活，直接返回
        if self._thread and self._thread.is_alive():
            return

        def _target() -> None:
            """后台线程的主函数。
            
            该函数在新线程中执行，负责：
            1. 创建新的事件循环
            2. 设置为当前线程的事件循环
            3. 通知主线程启动完成
            4. 运行事件循环直到被停止
            5. 清理未完成的任务
            """
            # 为当前线程创建新的事件循环
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # 通知主线程事件循环已就绪
            self._started.set()
            
            try:
                # 永久运行事件循环，直到调用 stop()
                self._loop.run_forever()
            finally:
                # 事件循环停止后，清理所有未完成的任务
                pending = asyncio.all_tasks(self._loop)
                
                # 取消所有待处理的任务
                for t in pending:
                    t.cancel()
                
                # 等待所有任务处理取消信号（捕获 CancelledError）
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                
                # 关闭事件循环
                self._loop.close()

        # 创建并启动后台线程
        self._thread = threading.Thread(
            target=_target,
            name="AsyncLoopRunner",  # 线程名称，便于调试
            daemon=True,             # 守护线程，主进程退出时自动结束
        )
        self._thread.start()
        
        # 阻塞等待事件循环启动完成，最多等待 5 秒
        self._started.wait(timeout=5)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """获取事件循环实例。
        
        Returns:
            asyncio.AbstractEventLoop: 后台线程中的事件循环
            
        Raises:
            RuntimeError: 如果事件循环尚未启动
        """
        if self._loop is None:
            raise RuntimeError("AsyncLoopRunner 尚未启动")
        return self._loop

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future:
        """将协程提交到后台事件循环，返回线程安全 Future。
        
        该方法是异步执行的，立即返回 Future 对象。
        可以通过 Future.result() 阻塞等待结果，或通过 Future.done() 轮询状态。
        
        Args:
            coro (Coroutine): 要执行的协程对象
            
        Returns:
            Future: 线程安全的 Future 对象，可用于获取协程的执行结果
            
        Examples:
            >>> runner = AsyncLoopRunner()
            >>> runner.start()
            >>> future = runner.submit(async_function())
            >>> result = future.result()  # 阻塞等待结果
        """
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """同步执行协程并返回结果。
        
        该方法会阻塞当前线程，直到协程执行完成。
        适用于需要在同步代码中调用异步函数并等待结果的场景。
        
        Args:
            coro (Coroutine): 要执行的协程对象
            
        Returns:
            Any: 协程的返回值
            
        Raises:
            Exception: 协程执行过程中抛出的任何异常
            
        Examples:
            >>> runner = AsyncLoopRunner()
            >>> runner.start()
            >>> result = runner.run(async_function())  # 同步等待
        """
        return self.submit(coro).result()

    def shutdown(self) -> None:
        """优雅停止事件循环并回收线程。
        
        该方法会：
        1. 请求事件循环停止（通过 call_soon_threadsafe）
        2. 等待后台线程结束（最多 5 秒）
        3. 清理内部状态
        
        Note:
            该方法应在应用关闭时调用，确保资源正确释放
            如果在 5 秒内线程未结束，将强制继续（线程为守护线程会自动终止）
        """
        # 如果事件循环未启动，直接返回
        if not self._loop:
            return
        
        # 在事件循环线程中安全地调用 stop() 方法
        self._loop.call_soon_threadsafe(self._loop.stop)
        
        # 等待线程结束，最多等待 5 秒
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        # 清理状态
        self._thread = None
        self._loop = None


__all__ = ["AsyncLoopRunner"]
