"""系统运维面板：手动控制后端各项后台流程。"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st  # type: ignore[import]

from utils import call_backend, get_json


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


@st.cache_data(ttl=5, show_spinner=False)
def _fetch_health_cached() -> Dict[str, Any]:
    data = get_json("/health", timeout=8)
    return data or {}


@st.cache_data(ttl=5, show_spinner=False)
def _fetch_status_cached() -> Dict[str, Any] | None:
    return get_json("/api/v1/operations/status", timeout=8)


def _refresh_status_panel(placeholder) -> Dict[str, Any] | None:
    _fetch_status_cached.clear()
    latest = _fetch_status_cached()
    if latest:
        with placeholder.container():
            _render_status_panel(latest)
    else:
        placeholder.empty()
    return latest


st.title("系统运维面板")
st.caption("手动控制数据摄取、增量学习与模型训练流程")

health_info = _fetch_health_cached()
if health_info:
    device_mode = health_info.get("device_mode", "-")
    device_str = health_info.get("device", "-")
    mode_label = "GPU" if device_mode.lower() == "gpu" else ("CPU" if device_mode.lower() == "cpu" else device_mode)
    st.markdown(f"**设备模式**：{mode_label} · 设备：`{device_str}`")

status_placeholder = st.empty()
status_data = _fetch_status_cached()
if status_data:
    with status_placeholder.container():
        _render_status_panel(status_data)
else:
    st.warning("未能加载系统状态，请稍后重试。")

if st.button("刷新状态概览", type="secondary"):
    status_data = _refresh_status_panel(status_placeholder)
    if not status_data:
        st.warning("未能加载系统状态，请稍后重试。")

st.markdown("### ⚙️ 任务控制")
col1, col2 = st.columns(2)
with col1:
    if st.button("启动数据摄取", type="primary"):
        result = call_backend(
            "/api/v1/operations/ingestion/start",
            method="POST",
            spinner="正在启动数据摄取...",
            timeout=60,
        )
        if result is not None:
            st.success(result.get("message", "数据摄取已启动"))
            status_snapshot = result.get("status")
            if status_snapshot:
                st.json(status_snapshot)
            _refresh_status_panel(status_placeholder)
with col2:
    if st.button("停止数据摄取", type="secondary"):
        result = call_backend(
            "/api/v1/operations/ingestion/stop",
            method="POST",
            spinner="正在停止数据摄取...",
            timeout=60,
        )
        if result is not None:
            st.success(result.get("message", "数据摄取已停止"))
            status_snapshot = result.get("status")
            if status_snapshot:
                st.json(status_snapshot)
            _refresh_status_panel(status_placeholder)

col3, col4 = st.columns(2)
with col3:
    if st.button("启动增量循环", type="primary"):
        result = call_backend(
            "/api/v1/operations/incremental/start",
            method="POST",
            spinner="正在启动增量循环...",
            timeout=60,
        )
        if result is not None:
            st.success(result.get("message", "增量循环已启动"))
            loop_snapshot = result.get("status")
            if loop_snapshot:
                st.json(loop_snapshot)
            _refresh_status_panel(status_placeholder)
with col4:
    if st.button("停止增量循环", type="secondary"):
        result = call_backend(
            "/api/v1/operations/incremental/stop",
            method="POST",
            spinner="正在停止增量循环...",
            timeout=60,
        )
        if result is not None:
            st.success(result.get("message", "增量循环已停止"))
            loop_snapshot = result.get("status")
            if loop_snapshot:
                st.json(loop_snapshot)
            _refresh_status_panel(status_placeholder)

st.markdown("### 🧠 融合模型训练")
with st.form("fusion_train_form"):
    sample_size = st.number_input("采样用户数", min_value=32, max_value=10000, value=256, step=32)
    epochs = st.number_input("训练轮次", min_value=1, max_value=200, value=3, step=1)
    lr = st.number_input("学习率", min_value=1e-5, max_value=1.0, value=1e-3, step=1e-4, format="%.5f")
    batch_size = st.number_input("批大小", min_value=8, max_value=1024, value=64, step=8)
    submitted = st.form_submit_button("开始训练", type="primary")
    if submitted:
        result = call_backend(
            "/api/v1/operations/fusion/train",
            method="POST",
            payload={
                "sample_size": int(sample_size),
                "epochs": int(epochs),
                "lr": float(lr),
                "batch_size": int(batch_size),
            },
            spinner="融合核心训练进行中...",
            timeout=600,
        )
        if result is not None:
            st.success(result.get("message", "训练已完成"))
            st.json(result.get("metrics", result))
            _refresh_status_panel(status_placeholder)

st.markdown("### 🕸️ HGT 图表征训练")
with st.form("hgt_training_form"):
    st.caption("配置遮蔽比例与训练参数，前端一键触发后端 HGT 训练与评估流程。")
    hgt_epochs = st.number_input("HGT 训练轮次", min_value=1, max_value=200, value=5, step=1)
    train_ratio = st.slider("训练集比例", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
    max_val_ratio = max(0.0, min(0.4, 0.99 - float(train_ratio)))
    default_val_ratio = 0.1 if 0.1 <= max_val_ratio else round(max_val_ratio, 2)
    val_ratio = st.slider(
        "验证集比例",
        min_value=0.0,
        max_value=max_val_ratio,
        value=default_val_ratio,
        step=0.01,
    )
    if train_ratio + val_ratio >= 1.0:
        st.warning("训练集与验证集比例之和需小于 1.0，当前设置将不会留下测试集。")
    negative_ratio = st.slider("负样本倍数", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    temperature = st.slider("对比损失温度", min_value=0.05, max_value=1.0, value=0.2, step=0.05)
    learning_rate = st.number_input(
        "自定义学习率 (可选)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=1e-4,
        format="%.5f",
        help="设置为 0 表示沿用当前优化器学习率",
    )
    seed = st.text_input("随机种子 (可选)", value="")
    
    # 训练模式配置
    st.markdown("#### 🎯 训练模式选择")
    training_mode = st.radio(
        "选择训练数据源",
        options=["完整图训练", "METIS 采样子图训练"],
        index=0,
        help="完整图训练使用所有数据，METIS 采样训练使用图分割后的子图",
    )
    
    # METIS 采样配置（仅在选择 METIS 模式时显示）
    if training_mode == "METIS 采样子图训练":
        st.markdown("##### METIS 采样参数配置")
        metis_num_parts = st.slider(
            "分区数量",
            min_value=2,
            max_value=20,
            value=4,
            step=1,
            help="将图分割为多少个分区，分区数越多，每个子图越小",
        )
        metis_imbalance_factor = st.slider(
            "不平衡因子",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            help="允许分区大小不均衡的程度，0 表示严格均衡",
        )
        metis_recursive = st.checkbox(
            "使用递归二分法",
            value=True,
            help="递归二分法通常能获得更好的分区质量",
        )
        metis_seed_input = st.text_input(
            "METIS 随机种子 (可选)",
            value="",
            help="用于可重复的分区结果",
        )
        metis_partition_id_input = st.text_input(
            "指定分区 ID (可选)",
            value="",
            help=f"指定使用哪个分区（0-{metis_num_parts-1}），留空则随机选择",
        )
    
    submitted_hgt = st.form_submit_button("运行 HGT 训练", type="primary")
    if submitted_hgt:
        payload: Dict[str, Any] = {
            "epochs": int(hgt_epochs),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "negative_ratio": float(negative_ratio),
            "temperature": float(temperature),
        }
        if learning_rate > 0:
            payload["learning_rate"] = float(learning_rate)
        seed = seed.strip()
        if seed:
            try:
                payload["seed"] = int(seed)
            except ValueError:
                st.warning("随机种子需为整数，将忽略该输入。")
        
        # 添加训练模式配置
        if training_mode == "完整图训练":
            payload["training_mode"] = "full_graph"
        else:
            payload["training_mode"] = "metis_sampling"
            payload["metis_num_parts"] = int(metis_num_parts)
            payload["metis_imbalance_factor"] = float(metis_imbalance_factor)
            payload["metis_recursive"] = bool(metis_recursive)
            
            metis_seed = metis_seed_input.strip()
            if metis_seed:
                try:
                    payload["metis_seed"] = int(metis_seed)
                except ValueError:
                    st.warning("METIS 随机种子需为整数，将忽略该输入。")
            
            metis_partition_id = metis_partition_id_input.strip()
            if metis_partition_id:
                try:
                    pid = int(metis_partition_id)
                    if 0 <= pid < metis_num_parts:
                        payload["metis_partition_id"] = pid
                    else:
                        st.warning(f"分区 ID 必须在 0-{metis_num_parts-1} 之间，将使用随机选择。")
                except ValueError:
                    st.warning("分区 ID 需为整数，将使用随机选择。")
        
        result = call_backend(
            "/api/v1/operations/training/hgt",
            method="POST",
            payload=payload,
            spinner="HGT 训练与评估进行中...",
            timeout=600,
        )
        if result is not None:
            st.success(result.get("message", "HGT 训练完成"))
            summary = result.get("summary", result)
            _refresh_status_panel(status_placeholder)
            
            # 显示大图统计信息
            if "graph_statistics" in summary:
                st.markdown("#### 📊 完整大图统计")
                graph_stats = summary["graph_statistics"]
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.metric("总节点数", graph_stats.get("total_nodes", "-"))
                with col_g2:
                    st.metric("总边数", graph_stats.get("total_edges", "-"))
                
                st.markdown("##### 节点统计")
                node_counts = graph_stats.get("node_counts", {})
                if node_counts:
                    node_df = pd.DataFrame({
                        "节点类型": list(node_counts.keys()),
                        "节点数": list(node_counts.values()),
                    })
                    st.dataframe(node_df, use_container_width=True)
                
                st.markdown("##### 边统计")
                edge_counts = graph_stats.get("edge_counts", {})
                if edge_counts:
                    edge_df = pd.DataFrame({
                        "边类型": list(edge_counts.keys()),
                        "边数": list(edge_counts.values()),
                    })
                    st.dataframe(edge_df, use_container_width=True)
                
                # 显示保存路径
                if "graph_save_path" in summary:
                    st.info(f"完整大图已保存至: {summary['graph_save_path']}")
            
            # 显示采样统计信息（如果有）
            if "sampling_stats" in summary:
                st.markdown("#### 📊 METIS 采样统计")
                stats = summary["sampling_stats"]
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("选中分区", stats.get("selected_partition", "-"))
                with col_s2:
                    st.metric("边切割数", stats.get("edge_cut", "-"))
                with col_s3:
                    original_nodes_total = sum(stats.get("original_nodes", {}).values())
                    sampled_nodes_total = sum(stats.get("sampled_nodes", {}).values())
                    sampling_ratio = sampled_nodes_total / original_nodes_total if original_nodes_total > 0 else 0
                    st.metric("采样比例", f"{sampling_ratio:.2%}")
                
                st.markdown("##### 节点统计")
                node_stats_df = pd.DataFrame({
                    "节点类型": list(stats.get("original_nodes", {}).keys()),
                    "原始节点数": list(stats.get("original_nodes", {}).values()),
                    "采样节点数": list(stats.get("sampled_nodes", {}).values()),
                })
                st.dataframe(node_stats_df, use_container_width=True)
                
                st.markdown("##### 边统计")
                edge_types = list(stats.get("original_edges", {}).keys())
                edge_stats_df = pd.DataFrame({
                    "边类型": edge_types,
                    "原始边数": [stats.get("original_edges", {}).get(et, 0) for et in edge_types],
                    "采样边数": [stats.get("sampled_edges", {}).get(et, 0) for et in edge_types],
                })
                st.dataframe(edge_stats_df, use_container_width=True)
                
                st.markdown("##### 分区大小分布")
                partition_sizes = stats.get("partition_sizes", [])
                if partition_sizes:
                    partition_df = pd.DataFrame({
                        "分区 ID": list(range(len(partition_sizes))),
                        "节点数": partition_sizes,
                    })
                    st.dataframe(partition_df, use_container_width=True)
            
            st.json(summary)

st.markdown("### 🧹 缓存与规则维护")
col5, col6 = st.columns(2)
with col5:
    if st.button("刷新规则结构", type="primary"):
        result = call_backend(
            "/api/v1/operations/rules/refresh",
            method="POST",
            spinner="正在刷新规则结构...",
            timeout=60,
        )
        if result is not None:
            st.success(result.get("message", "规则刷新完成"))
            _refresh_status_panel(status_placeholder)
with col6:
    if st.button("清空解释器缓存", type="secondary"):
        result = call_backend(
            "/api/v1/operations/explainer/clear",
            method="POST",
            spinner="正在清理解释器缓存...",
            timeout=60,
        )
        if result is not None:
            st.success(result.get("message", "解释器缓存已清空"))
            _refresh_status_panel(status_placeholder)

st.markdown("### 🛑 停机（仅本地/容器调试）")
with st.form("shutdown_form"):
    st.warning("该操作将显式停止后台任务并关闭后端的异步事件循环，停机后需重启后端进程以恢复服务。")
    confirm_shutdown = st.checkbox("我已了解停机影响并确认继续")
    submitted_shutdown = st.form_submit_button("执行停机", type="primary")
    if submitted_shutdown:
        if not confirm_shutdown:
            st.warning("请勾选确认后再执行停机。")
        else:
            result = call_backend(
                "/api/v1/operations/shutdown",
                method="POST",
                spinner="正在停机...",
                timeout=60,
            )
            if result is not None:
                st.success(result.get("message", "后端停机完成"))
                st.info("如需继续使用，请重新启动后端服务。")
                _refresh_status_panel(status_placeholder)

st.divider()
st.markdown("如需最新状态，请使用侧边栏的重新运行按钮或刷新页面。")
