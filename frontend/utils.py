"""Streamlit 前端与后端通信的辅助工具模块。

提供统一的 HTTP 请求封装，简化前端页面与后端 API 的交互。
包含错误处理、超时控制和 URL 管理功能。
"""
from __future__ import annotations

from contextlib import nullcontext
import os
from typing import Any, Dict, Optional

import requests
import streamlit as st  # type: ignore[import]


@st.cache_data(show_spinner=False)
def get_backend_base_url() -> str:
    """获取后端 API 的基础 URL。
    
    按优先级从以下来源读取后端地址：
    1. Streamlit secrets 配置（生产环境）
    2. 环境变量 BACKEND_BASE_URL（容器化部署）
    3. 默认本地地址 http://localhost:5000（开发环境）
    
    Returns:
        str: 去除尾部斜杠的后端基础 URL
        
    Note:
        该函数使用 @st.cache_data 装饰，结果会被缓存
        避免重复读取配置文件或环境变量
    """
    return (
        st.secrets.get("backend_base_url")  # 优先使用 secrets 配置
        or os.getenv("BACKEND_BASE_URL", "http://localhost:5000")  # 其次使用环境变量
    ).rstrip("/")  # 去除尾部斜杠，统一 URL 格式


# 导出常量，供页面模块直接使用
BACKEND_URL = get_backend_base_url()


def request_json(
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """调用后端 API 并统一处理错误。
    
    该函数封装了 HTTP 请求逻辑，提供：
    - 自动拼接完整 URL
    - JSON 请求体和响应体处理
    - 超时控制
    - 统一错误处理（网络错误、HTTP 错误、解析错误）
    - 404 特殊处理（返回 None 而非报错）
    
    Args:
        path (str): API 路径，如 '/api/v1/profile/123'
        method (str): HTTP 方法，支持 GET, POST, PUT, DELETE 等
        payload (Dict[str, Any] | None): 请求体数据（自动序列化为 JSON）
        timeout (float): 请求超时时间（秒），默认 10 秒
        
    Returns:
        Dict[str, Any] | None: 
            - 成功时返回解析后的 JSON 字典
            - 404 错误时返回 None
            - 响应体为空时返回空字典 {}
            - 其他错误时返回 None 并显示错误提示
            
    Examples:
        >>> # 查询用户画像
        >>> profile = request_json("/api/v1/profile/123", method="GET")
        >>> 
        >>> # 更新规则
        >>> result = request_json(
        ...     "/api/v1/rules/1",
        ...     method="PUT",
        ...     payload={"name": "新规则"}
        ... )
    """
    # 获取后端基础 URL
    base_url = get_backend_base_url()
    
    try:
        # 发送 HTTP 请求
        response = requests.request(
            method.upper(),           # 统一转换为大写（GET, POST 等）
            f"{base_url}{path}",      # 拼接完整 URL
            json=payload,             # 请求体（自动序列化为 JSON）
            timeout=timeout,          # 超时设置
        )
        
        # 404 错误特殊处理：返回 None 而不抛出异常
        # 适用于"资源不存在"的正常业务逻辑
        if response.status_code == 404:
            return None
        
        # 其他 HTTP 错误（4xx, 5xx）抛出异常
        response.raise_for_status()
        
        # 处理空响应（如 204 No Content）
        if not response.text:
            return {}
        
        # 解析并返回 JSON 响应
        return response.json()
        
    except Exception as exc:  # pragma: no cover - 前端容错处理
        # 捕获所有异常：网络错误、超时、JSON 解析错误等
        # 在 Streamlit 界面显示友好的错误提示
        st.error(f"调用后端接口失败: {exc}")
        return None


def call_backend(
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 10.0,
    spinner: str | None = None,
    success_message: str | None = None,
) -> Optional[Dict[str, Any]]:
    """统一包装带提示的后端调用。

    Args:
        path: API 路径。
        method: HTTP 方法。
        payload: JSON 请求体。
        timeout: 超时时间（秒）。
        spinner: 非空时在调用期间显示加载提示。
        success_message: 调用成功且有返回值时展示的成功提示。

    Returns:
        dict | None: 与 ``request_json`` 一致。
    """

    context = st.spinner(spinner) if spinner else nullcontext()
    with context:
        result = request_json(path, method=method, payload=payload, timeout=timeout)

    if result is not None and success_message:
        st.success(success_message)

    return result


def get_json(path: str, *, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """便捷的 GET 请求封装。
    
    简化只读 API 调用的代码,默认超时时间为 10 秒。
    
    Args:
        path (str): API 路径
        timeout (float): 请求超时时间（秒），默认 5 秒
        
    Returns:
        Dict[str, Any] | None: API 响应的 JSON 数据或 None
        
    Examples:
        >>> # 查询用户画像
        >>> profile = get_json("/api/v1/profile/123")
        >>> if profile:
        ...     print(profile["age_group"])
    """
    return request_json(path, method="GET", timeout=timeout)


def post_json(
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """便捷的 POST 请求封装。
    
    用于创建资源或触发操作，默认超时时间较长（30秒）。
    适用于可能耗时较长的操作（如模型训练、数据生成）。
    
    Args:
        path (str): API 路径
        payload (Dict[str, Any] | None): 请求体数据
        timeout (float): 请求超时时间（秒），默认 30 秒
        
    Returns:
        Dict[str, Any] | None: API 响应的 JSON 数据或 None
        
    Examples:
        >>> # 触发图刷新
        >>> result = post_json(
        ...     "/api/v1/graph/refresh",
        ...     payload={"mode": "incremental"}
        ... )
        >>> if result:
        ...     print(f"刷新完成，耗时 {result['duration_seconds']} 秒")
    """
    return request_json(path, method="POST", payload=payload, timeout=timeout)


__all__ = [
    "get_backend_base_url",
    "get_json",
    "post_json",
    "request_json",
    "call_backend",
    "BACKEND_URL",
]
