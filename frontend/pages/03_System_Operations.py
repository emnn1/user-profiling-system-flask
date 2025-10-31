"""系统运维面板：手动控制后端各项后台流程。"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
import requests
import streamlit as st


@st.cache_data(show_spinner=False)
def _get_backend_base_url() -> str:
    return (
        st.secrets.get("backend_base_url")
        or os.getenv("BACKEND_BASE_URL", "http://localhost:5000")
    ).rstrip("/")


def _get(path: str) -> Dict[str, Any]:
    base_url = _get_backend_base_url()
    try:
        response = requests.get(f"{base_url}{path}", timeout=8)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - 前端容错
        st.error(f"获取状态失败: {exc}")
        return {}


def _post(path: str, payload: Optional[Dict[str, Any]] = None, *, spinner: str) -> Dict[str, Any]:
    base_url = _get_backend_base_url()
    with st.spinner(spinner):
        try:
            response = requests.post(
                f"{base_url}{path}",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            if response.headers.get("Content-Type", "").startswith("application/json") and response.text:
                return response.json()
            return {"message": "操作已完成"}
        except Exception as exc:  # pragma: no cover - 前端容错
            st.error(f"操作失败: {exc}")
            return {}


def _render_status_panel(status: Dict[str, Any]) -> None:
    st.markdown("### 📈 当前运行态")
    ingestion = status.get("ingestion", {})
    loop_status = status.get("incremental_loop", {})

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            "数据摄取",
            "运行中" if ingestion.get("running") else "已停止",
            help="SystemController 汇报的数据摄取状态",
        )
    with col_b:
        st.metric(
            "待处理事件",
            ingestion.get("pending_events", "N/A"),
            help="当前排队等待增量学习的事件数量",
        )
    with col_c:
        st.metric(
            "增量循环",
            "运行中" if loop_status.get("running") else "已停止",
            help="后台增量训练循环状态",
        )

    extra_cols = st.columns(3)
    with extra_cols[0]:
        st.metric(
            "摄取开始时间",
            ingestion.get("started_at", "-"),
            help="最近一次启动摄取的时间戳",
        )
    with extra_cols[1]:
        st.metric(
            "最后事件时间",
            ingestion.get("last_event_at", "-"),
            help="事件队列最近一次入队时间",
        )
    with extra_cols[2]:
        st.metric(
            "增量最后批",
            loop_status.get("last_batch_at", "-"),
            help="增量学习最近一次消费事件的时间戳",
        )

    history = status.get("history", [])
    if history:
        st.markdown("#### 📝 最近操作记录")
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("暂无历史操作记录。")


st.title("系统运维面板")
st.caption("手动控制数据摄取、增量学习与模型训练流程")

health_info = _get("/health")
if health_info:
    device_mode = health_info.get("device_mode", "-")
    device_str = health_info.get("device", "-")
    mode_label = "GPU" if device_mode.lower() == "gpu" else ("CPU" if device_mode.lower() == "cpu" else device_mode)
    st.markdown(f"**设备模式**：{mode_label} · 设备：`{device_str}`")

status_placeholder = st.empty()
status_data = _get("/api/v1/operations/status")
if status_data:
    with status_placeholder.container():
        _render_status_panel(status_data)
else:
    st.warning("未能加载系统状态，请稍后重试。")

if st.button("刷新状态概览", type="secondary"):
    status_data = _get("/api/v1/operations/status")
    if status_data:
        with status_placeholder.container():
            _render_status_panel(status_data)

st.markdown("### ⚙️ 任务控制")
col1, col2 = st.columns(2)
with col1:
    if st.button("启动数据摄取", type="primary"):
        result = _post("/api/v1/operations/ingestion/start", spinner="正在启动数据摄取...")
        if result:
            st.success(result.get("message", "数据摄取已启动"))
            if result.get("status"):
                st.json(result["status"])
with col2:
    if st.button("停止数据摄取", type="secondary"):
        result = _post("/api/v1/operations/ingestion/stop", spinner="正在停止数据摄取...")
        if result:
            st.success(result.get("message", "数据摄取已停止"))
            if result.get("status"):
                st.json(result["status"])

col3, col4 = st.columns(2)
with col3:
    if st.button("启动增量循环", type="primary"):
        result = _post("/api/v1/operations/incremental/start", spinner="正在启动增量循环...")
        if result:
            st.success(result.get("message", "增量循环已启动"))
            if result.get("status"):
                st.json(result["status"])
with col4:
    if st.button("停止增量循环", type="secondary"):
        result = _post("/api/v1/operations/incremental/stop", spinner="正在停止增量循环...")
        if result:
            st.success(result.get("message", "增量循环已停止"))
            if result.get("status"):
                st.json(result["status"])

st.markdown("### 🧠 融合模型训练")
with st.form("fusion_train_form"):
    sample_size = st.number_input("采样用户数", min_value=32, max_value=10000, value=256, step=32)
    epochs = st.number_input("训练轮次", min_value=1, max_value=200, value=3, step=1)
    lr = st.number_input("学习率", min_value=1e-5, max_value=1.0, value=1e-3, step=1e-4, format="%.5f")
    batch_size = st.number_input("批大小", min_value=8, max_value=1024, value=64, step=8)
    submitted = st.form_submit_button("开始训练", type="primary")
    if submitted:
        result = _post(
            "/api/v1/operations/fusion/train",
            payload={
                "sample_size": int(sample_size),
                "epochs": int(epochs),
                "lr": float(lr),
                "batch_size": int(batch_size),
            },
            spinner="融合核心训练进行中...",
        )
        if result:
            st.success(result.get("message", "训练已完成"))
            st.json(result.get("metrics", result))

st.markdown("### 🧹 缓存与规则维护")
col5, col6 = st.columns(2)
with col5:
    if st.button("刷新规则结构", type="primary"):
        result = _post("/api/v1/operations/rules/refresh", spinner="正在刷新规则结构...")
        if result:
            st.success(result.get("message", "规则刷新完成"))
with col6:
    if st.button("清空解释器缓存", type="secondary"):
        result = _post("/api/v1/operations/explainer/clear", spinner="正在清理解释器缓存...")
        if result:
            st.success(result.get("message", "解释器缓存已清空"))

st.markdown("### 🛑 停机（仅本地/容器调试）")
with st.form("shutdown_form"):
    st.warning("该操作将显式停止后台任务并关闭后端的异步事件循环，停机后需重启后端进程以恢复服务。")
    confirm_shutdown = st.checkbox("我已了解停机影响并确认继续")
    submitted_shutdown = st.form_submit_button("执行停机", type="primary")
    if submitted_shutdown:
        if not confirm_shutdown:
            st.warning("请勾选确认后再执行停机。")
        else:
            result = _post("/api/v1/operations/shutdown", spinner="正在停机...")
            if result is not None:
                st.success(result.get("message", "后端停机完成"))
                st.info("如需继续使用，请重新启动后端服务。")

st.divider()
st.markdown("如需最新状态，请使用侧边栏的重新运行按钮或刷新页面。")
