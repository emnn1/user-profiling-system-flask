"""Streamlit 前端入口与全局概览。

该模块作为 Streamlit 多页面应用的首页，承担以下职责：

- 统一配置应用标题、布局与缓存策略；
- 提供健康检查与全量刷新等全局控制操作；
- 为侧边栏子页面（画像查询、规则管理）提供上下文信息和使用指引。

模块内部通过 :func:`utils.get_backend_base_url` 集中解析后端地址，
以确保页面之间的调用保持一致。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from utils import get_backend_base_url, get_json, post_json


st.set_page_config(page_title="用户画像分析控制台", layout="wide")

_STATUS_BADGE = {
    "running": "🟡 进行中",
    "completed": "🟢 已完成",
    "idle": "⚪ 待命",
    "error": "🔴 异常",
    "skipped": "⚫ 已跳过",
}


@st.cache_data(show_spinner=False)
def fetch_health() -> Dict[str, Any]:
    """调用后端健康检查端点并返回 JSON 结果。
    
    健康检查接口返回系统运行状态，包括：
    - 设备信息（GPU/CPU）
    - 数据摄取队列状态
    - 增量学习循环状态
    
    Returns:
        Dict[str, Any]: 健康检查结果，失败时返回空字典
        
    Note:
        使用 @st.cache_data 缓存结果，避免频繁请求
        超时设置为 10 秒
    """
    data = get_json("/health", timeout=10)
    if not data:
        st.warning("无法连接后端健康检查接口")
        return {}
    return data


@st.cache_data(ttl=5, show_spinner=False)
def fetch_metrics() -> Dict[str, Any]:
    """轮询后端运行指标数据。
    
    获取系统运行指标，包括：
    - 资源使用情况（CPU、内存、GPU）
    - 训练状态（损失、Epoch）
    - 刷新流程状态
    
    Returns:
        Dict[str, Any]: 指标数据，失败时返回空字典
        
    Note:
        TTL=5 秒，每 5 秒自动更新一次缓存
        适用于实时监控面板
    """
    return get_json("/api/v1/operations/metrics", timeout=4) or {}


def trigger_full_refresh(options: Dict[str, Any]) -> None:
    """向后端提交全量图刷新请求，并在页面上展示结果。
    
    全量刷新会重新构建异构图并重新训练模型，耗时较长。
    该操作会阻塞页面直到完成（最长 5 分钟）。
    
    Args:
        options (Dict[str, Any]): 刷新配置，包含：
            - mode: 刷新模式（incremental/full）
            - retrain_epochs: 重训练轮数
            - 其他参数参见 GraphRefreshMode
            
    Side Effects:
        - 显示加载动画（spinner）
        - 完成后显示成功消息和详细结果
        - 失败时静默返回（不显示错误）
    """
    with st.spinner("正在执行全量刷新，请稍候..."):
        # 调用后端刷新接口，超时 5 分钟
        payload = post_json(
            "/api/v1/graph/refresh",
            payload=options,
            timeout=300,
        )
        
        # 请求失败时静默返回
        if payload is None:
            return
        
        # 显示成功消息
        st.success(payload.get("message", "全量刷新已完成"))
        
        # 展示详细结果（耗时、节点数等）
        st.json(payload)


def _status_label(status: str | None) -> str:
    """将状态枚举值转换为带图标的友好标签。
    
    Args:
        status (str | None): 状态枚举（running, completed, idle等）
        
    Returns:
        str: 带颜色图标的标签文本，如 "🟢 已完成"
        
    Examples:
        >>> _status_label("running")
        '🟡 进行中'
        >>> _status_label("completed")
        '🟢 已完成'
        >>> _status_label(None)
        ''
    """
    if not status:
        return ""
    return _STATUS_BADGE.get(status, status)


def _loss_dataframe(entries: List[Dict[str, Any]]) -> pd.DataFrame | None:
    """将损失历史记录转换为 Pandas DataFrame，便于图表展示。
    
    Args:
        entries (List[Dict[str, Any]]): 损失记录列表，每项包含：
            - epoch: 训练轮次
            - loss: 损失值
            
    Returns:
        pd.DataFrame | None: 以 epoch 为索引的 DataFrame，或 None（无数据时）
        
    Note:
        自动按 epoch 排序，便于时序图表绘制
    """
    if not entries:
        return None
    
    # 转换为 DataFrame
    df = pd.DataFrame(entries)
    
    # 如果有 epoch 列，设置为索引并排序
    if "epoch" in df.columns:
        df = df.sort_values("epoch")
        df = df.set_index("epoch")
    
    return df


def _update_metrics_history(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """更新指标历史记录，用于绘制时序图表。
    
    该函数维护一个固定大小（200 条）的时序队列，
    存储 CPU、内存、GPU 等资源使用情况。
    
    Args:
        metrics (Dict[str, Any]): 当前指标快照，包含：
            - timestamp: 时间戳
            - resources: 资源使用情况
            
    Returns:
        List[Dict[str, Any]]: 完整的历史记录列表
        
    Side Effects:
        - 将历史数据存储在 st.session_state 中
        - 自动去重（基于 timestamp）
        - 超过 200 条时删除最旧记录
        
    Note:
        Session State 在页面刷新后会丢失，
        需要在每次访问时重新积累
    """
    # 从 Session State 获取或初始化历史记录
    history: List[Dict[str, Any]] = st.session_state.setdefault("metrics_history", [])
    
    # 提取关键信息
    timestamp = metrics.get("timestamp")
    resources = metrics.get("resources", {}) if metrics else {}
    
    # 无时间戳则跳过
    if not timestamp:
        return history
    
    # 去重：避免同一时间戳重复记录
    if history and history[-1].get("timestamp") == timestamp:
        return history
    
    # 添加新记录
    history.append(
        {
            "timestamp": timestamp,
            "cpu": resources.get("cpu_percent"),               # CPU 使用率（%）
            "rss": resources.get("memory_rss_mb"),             # 内存占用（MB）
            "gpu_alloc": (resources.get("gpu") or {}).get("memory_allocated_mb"),  # GPU 显存（MB）
        }
    )
    
    # 限制队列大小为 200 条
    if len(history) > 200:
        del history[: len(history) - 200]
    
    return history


def _render_training_block(title: str, state: Dict[str, Any]) -> None:
    """渲染训练状态展示块。
    
    展示模型训练的实时进度，包括：
    - 当前 Epoch 和总 Epoch
    - 最新损失值
    - 学习率
    - 损失曲线图
    
    Args:
        title (str): 展示块标题（如"模型训练"、"融合训练"）
        state (Dict[str, Any]): 训练状态数据，包含：
            - status: 训练状态（running/completed等）
            - current_epoch: 当前轮次
            - total_epochs: 总轮次
            - last_loss: 最新损失
            - learning_rate: 学习率
            - loss_history: 损失历史记录
            
    Side Effects:
        在 Streamlit 页面渲染指标卡片和折线图
    """
    # 获取状态标签（带图标）
    status = state.get("status")
    badge = _status_label(status)
    caption = f"{title} {badge}" if badge else title
    st.markdown(f"**{caption}**")

    # 渲染关键指标
    cols = st.columns(3)
    cols[0].metric("当前 Epoch", state.get("current_epoch", "-"))
    cols[1].metric("总 Epoch", state.get("total_epochs", "-"))
    
    # 损失值：优先显示 last_loss，其次 final_loss
    last_loss = state.get("last_loss") or state.get("final_loss")
    cols[2].metric(
        "最新损失",
        f"{last_loss:.4f}" if isinstance(last_loss, (int, float)) else "-"
    )

    # 显示学习率（如果有）
    if state.get("learning_rate"):
        st.caption(f"学习率：{state['learning_rate']}")

    # 绘制损失曲线
    loss_df = _loss_dataframe(state.get("loss_history") or [])
    if loss_df is not None and "loss" in loss_df.columns:
        st.line_chart(loss_df["loss"], height=160)
    else:
        st.write("暂无损失曲线数据。")


def _render_refresh_block(state: Dict[str, Any]) -> None:
    """渲染图刷新状态展示块。
    
    展示全量刷新流程的进度和结果。
    
    Args:
        state (Dict[str, Any]): 刷新状态数据，包含：
            - status: 刷新状态（running/completed/idle等）
            
    Side Effects:
        在 Streamlit 页面渲染状态指标
    """
    # 获取状态标签
    status = state.get("status", "idle")
    badge = _status_label(status)
    
    # 显示刷新状态
    cols = st.columns(3)
    cols[0].metric("刷新状态", badge or "-", help="全量刷新主流程状态")
    cols[1].metric("刷新模式", state.get("mode", "-"))
    cols[2].metric("图构建范围", state.get("scope", "-"))

    # 显示刷新参数（如采样比例、训练轮次等）
    parameters = state.get("parameters") or {}
    if parameters:
        st.caption(
            "参数：" + ", ".join(
                f"{key}={value}" for key, value in parameters.items() if value is not None
            )
        )

    # 显示当前阶段（如"构建图"、"训练模型"等）
    stage = state.get("stage")
    if stage:
        st.info(f"当前阶段：{stage}")

    # 如果包含 HGT 训练状态，单独渲染训练块
    hgt_state = state.get("hgt", {})
    if hgt_state:
        _render_training_block("HGT 训练", hgt_state)

    # 如果包含融合训练状态，单独渲染训练块
    fusion_state = state.get("fusion", {})
    if fusion_state:
        _render_training_block("融合核心训练", fusion_state)

    # 显示刷新摘要（完成后的统计信息）
    summary = state.get("summary")
    if summary:
        with st.expander("刷新摘要", expanded=False):
            st.json(summary)


def render_metrics_panel(metrics: Dict[str, Any]) -> None:
    """渲染系统运行指标监控面板。
    
    该函数负责展示完整的系统监控信息，包括：
    - 资源使用情况（CPU、内存、GPU）
    - 时序图表（资源使用趋势）
    - 训练进度（HGT 和融合训练）
    - 后台任务状态（数据摄取、增量循环）
    
    Args:
        metrics (Dict[str, Any]): 从后端获取的指标数据
        
    Side Effects:
        在 Streamlit 页面渲染多个指标卡片、图表和状态块
    """
    st.markdown("### 📡 实时运行指标")
    
    # 无数据时显示提示
    if not metrics:
        st.info("暂未获取到实时指标数据。")
        return

    # 提取资源使用情况
    resources = metrics.get("resources", {})
    gpu_info = resources.get("gpu") or {}

    # 第一行：CPU、内存、GPU 指标卡片
    col1, col2, col3 = st.columns(3)
    
    # CPU 使用率
    cpu_percent = resources.get("cpu_percent")
    col1.metric(
        "CPU 使用率",
        f"{cpu_percent:.1f}%" if isinstance(cpu_percent, (int, float)) else "N/A",
    )
    
    # 内存占用（RSS）
    col2.metric("RSS 内存 (MB)", resources.get("memory_rss_mb", "N/A"))
    
    # GPU 显存占用（如果有 GPU）
    if gpu_info:
        col3.metric(
            "GPU 显存占用 (MB)",
            gpu_info.get("memory_allocated_mb", "N/A"),
            help=gpu_info.get("device", "GPU"),
        )
    else:
        col3.metric("GPU 显存占用 (MB)", "-", help="当前运行在 CPU 上")

    # 时序图表：CPU 和内存使用趋势
    history = _update_metrics_history(metrics)
    history_df = pd.DataFrame(history)
    if not history_df.empty and {"timestamp", "cpu"}.issubset(history_df.columns):
        history_df = history_df.set_index("timestamp")
        # 绘制面积图，展示资源使用趋势
        st.area_chart(history_df[["cpu", "rss"]].dropna(how="all"), height=180)

    # 图刷新状态（如果正在刷新）
    refresh_state = metrics.get("refresh") or {}
    if refresh_state:
        _render_refresh_block(refresh_state)

    # 独立 HGT 训练状态
    hgt_state = metrics.get("hgt_training") or {}
    if hgt_state:
        st.markdown("### 🧠 独立 HGT 训练")
        _render_training_block("HGT 训练", hgt_state)

    # 融合核心训练状态
    fusion_state = metrics.get("fusion_training") or {}
    if fusion_state:
        st.markdown("### 🔗 融合核心训练")
        _render_training_block("融合核心训练", fusion_state)

    # 后台任务状态快照
    ingestion_state = metrics.get("ingestion") or {}
    loop_state = metrics.get("incremental_loop") or {}
    if ingestion_state or loop_state:
        st.markdown("### 📥 后台任务快照")
        cols = st.columns(2)
        
        # 数据摄取状态
        if ingestion_state:
            cols[0].metric(
                "数据摄取",
                _status_label("running" if ingestion_state.get("running") else "idle"),
                help="摄取任务状态",
            )
        
        # 增量学习循环状态
        if loop_state:
            cols[1].metric(
                "增量循环",
                _status_label("running" if loop_state.get("running") else "idle"),
                help="增量学习循环状态",
            )


# ============================================================================
# 主页面渲染
# ============================================================================

st.title("📊 用户画像与策略推荐系统")
st.caption("集成规则引擎 + GNN 嵌入的混合画像原型")
st.caption(f"后端服务：{get_backend_base_url()}")

# ----------------------------------------------------------------------------
# 系统健康状态展示
# ----------------------------------------------------------------------------
health = fetch_health()

ingestion = health.get("ingestion", {}) if health else {}
loop_status = health.get("incremental_loop", {}) if health else {}

# 第一行：系统运行状态指标卡片
col1, col2, col3 = st.columns(3)

with col1:
    # 数据摄取任务状态
    st.metric(
        "数据摄取状态",
        "运行中" if ingestion.get("running") else "已停止",
        help="SystemController 汇报的数据摄取任务状态",
    )

with col2:
    # 事件队列积压情况
    st.metric(
        "事件队列积压",
        ingestion.get("pending_events", "N/A"),
        help="当前待消费的事件数量",
    )

with col3:
    # 增量学习循环状态
    st.metric(
        "增量循环",
        "运行中" if loop_status.get("running") else "已停止",
        help="后台增量学习循环状态，请在运维面板中控制",
    )

st.info("提示：系统运维能力已迁移至 03_System_Operations 页面，可手动控制各项任务。")

# ----------------------------------------------------------------------------
# 全量刷新控制表单
# ----------------------------------------------------------------------------
st.markdown("### 🔧 全量刷新控制")

with st.form("global_refresh_form"):
    # 刷新策略选项（仅刷新嵌入 vs 重新训练模型）
    mode_display = {
        "embedding_only": "仅刷新嵌入 (Embedding Only)",
        "full_retrain": "重新训练模型 (Full Retrain)",
    }
    selected_mode = st.selectbox(
        "刷新策略",
        options=list(mode_display.keys()),
        format_func=lambda key: mode_display[key],
    )
    
    # 图构建范围选项（全量图 vs 采样图）
    scope_display = {
        "full": "全量图",
        "sampled": "采样图",
    }
    selected_scope = st.selectbox(
        "图构建范围",
        options=list(scope_display.keys()),
        format_func=lambda key: scope_display[key],
    )

    # 如果选择采样图，显示采样比例滑块
    sample_ratio: float | None = None
    if selected_scope == "sampled":
        sample_ratio = st.slider(
            "采样比例",
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            value=0.5,
            help="采样图的节点/边比例，用于快速测试",
        )

    # 训练参数配置
    retrain_epochs = st.slider(
        "HGT 训练轮次",
        min_value=1,
        max_value=50,
        value=3,
        help="HGT 模型的训练 Epoch 数",
    )
    
    fusion_epochs = st.slider(
        "融合核心训练轮次",
        min_value=1,
        max_value=60,
        value=5,
        help="规则与嵌入融合层的训练 Epoch 数",
    )
    
    temperature = st.slider(
        "HGT 对比学习温度",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="对比学习损失的温度参数，越小越强调困难样本",
    )
    
    learning_rate = st.number_input(
        "自定义学习率 (可选)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=1e-4,
        format="%.5f",
        help="设置为 0 表示沿用后端优化器当前学习率",
    )

    # 提交按钮
    submitted = st.form_submit_button("执行刷新", type="primary")
    
    if submitted:
        # 构建刷新配置
        refresh_options: Dict[str, Any] = {
            "mode": selected_mode,
            "graph_scope": selected_scope,
            "retrain_epochs": retrain_epochs,
            "fusion_epochs": fusion_epochs,
            "temperature": temperature,
        }
        
        # 添加采样比例（如果适用）
        if sample_ratio is not None:
            refresh_options["sample_ratio"] = sample_ratio
        
        # 添加自定义学习率（如果设置）
        if learning_rate > 0:
            refresh_options["learning_rate"] = learning_rate
        
        # 触发刷新操作
        trigger_full_refresh(refresh_options)

# ----------------------------------------------------------------------------
# 实时指标监控面板（自动刷新）
# ----------------------------------------------------------------------------
# 每 5 秒自动刷新一次页面，获取最新指标数据
_ = st_autorefresh(interval=5000, key="dashboard_metrics_refresh")

# 获取并渲染实时指标
metrics_data = fetch_metrics()
render_metrics_panel(metrics_data)

# ----------------------------------------------------------------------------
# 使用指南
# ----------------------------------------------------------------------------
st.markdown(
    """
    ### 使用指南

    - **用户画像查询** 页面：输入用户 ID，联通后端画像与解释接口，查看最终画像分、权重与 SHAP 解释。
    - **规则调优** 页面：方便运营同学在线查看、增删改业务规则，并同步后端。
    - **系统运维** 页面：手动控制数据摄取、增量学习等后台任务的启动和停止。
    - **数据配置** 页面：动态调整模拟数据生成参数，无需修改代码。

    确保后端 Flask 服务已启动，并通过 `BACKEND_BASE_URL` 环境变量或 `secrets.toml` 提供接口地址。
    """
)
