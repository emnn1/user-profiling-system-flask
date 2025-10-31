"""Streamlit 前端入口与全局概览。

该模块作为 Streamlit 多页面应用的首页，承担以下职责：

- 统一配置应用标题、布局与缓存策略；
- 提供健康检查与全量刷新等全局控制操作；
- 为侧边栏子页面（画像查询、规则管理）提供上下文信息和使用指引。

模块内部通过 :func:`_get_backend_base_url` 集中解析后端地址，
以确保页面之间的调用保持一致。"""
from __future__ import annotations

import os
from typing import Any, Dict

import requests
import streamlit as st


st.set_page_config(page_title="用户画像分析控制台", layout="wide")


@st.cache_data(show_spinner=False)
def _get_backend_base_url() -> str:
	"""解析后端服务基地址。

	优先读取 ``.streamlit/secrets.toml`` 中的 ``backend_base_url``，
	当未配置时回退到环境变量 ``BACKEND_BASE_URL``，再落到本地默认值。
	最后统一去除尾部斜杠，避免重复拼接 ``//``。"""

	return (
		st.secrets.get("backend_base_url")
		or os.getenv("BACKEND_BASE_URL", "http://localhost:5000")
	).rstrip("/")


def fetch_health() -> Dict[str, Any]:
	"""调用后端健康检查端点并返回 JSON 结果。"""

	base_url = _get_backend_base_url()
	try:
		response = requests.get(f"{base_url}/health", timeout=3)
		response.raise_for_status()
		return response.json()
	except Exception as exc:  # pragma: no cover - 前端容错
		st.warning(f"无法连接后端健康检查接口: {exc}")
		return {}


def trigger_full_refresh(options: Dict[str, Any]) -> None:
	"""向后端提交全量图刷新请求，并在页面上展示结果。"""

	base_url = _get_backend_base_url()
	with st.spinner("正在执行全量刷新，请稍候..."):
		try:
			response = requests.post(
				f"{base_url}/api/v1/graph/refresh",
				json=options,
				timeout=300,
			)
			response.raise_for_status()
			payload = response.json()
			st.success(payload.get("message", "全量刷新已完成"))
			st.json(payload)
		except Exception as exc:  # pragma: no cover - 前端容错
			st.error(f"触发全图刷新失败: {exc}")


st.title("📊 用户画像与策略推荐系统")
st.caption("集成规则引擎 + GNN 嵌入的混合画像原型")

health = fetch_health()

ingestion = health.get("ingestion", {}) if health else {}
loop_status = health.get("incremental_loop", {}) if health else {}

col1, col2, col3 = st.columns(3)
with col1:
	st.metric(
		"数据摄取状态",
		"运行中" if ingestion.get("running") else "已停止",
		help="SystemController 汇报的数据摄取任务状态",
	)
with col2:
	st.metric(
		"事件队列积压",
		ingestion.get("pending_events", "N/A"),
		help="当前待消费的事件数量",
	)
with col3:
	st.metric(
		"增量循环",
		"运行中" if loop_status.get("running") else "已停止",
		help="后台增量学习循环状态，请在运维面板中控制",
	)

st.info("提示：系统运维能力已迁移至 ‘03_System_Operations’ 页面，可手动控制各项任务。")

st.markdown("### 🔧 全量刷新控制")
with st.form("global_refresh_form"):
	mode_display = {
		"embedding_only": "仅刷新嵌入 (Embedding Only)",
		"full_retrain": "重新训练模型 (Full Retrain)",
	}
	scope_display = {
		"full": "全量图",
		"sampled": "采样图",
	}

	selected_mode = st.selectbox(
		"刷新策略",
		options=["embedding_only", "full_retrain"],
		format_func=lambda key: mode_display[key],
	)
	selected_scope = st.selectbox(
		"图构建范围",
		options=["full", "sampled"],
		format_func=lambda key: scope_display[key],
	)

	sample_ratio: float | None = None
	if selected_scope == "sampled":
		sample_ratio = st.slider("采样比例", min_value=0.1, max_value=1.0, step=0.1, value=0.5)

	retrain_epochs = st.slider("HGT 训练轮次", min_value=1, max_value=20, value=2)
	fusion_epochs = st.slider("融合核心训练轮次", min_value=1, max_value=30, value=3)

	submitted = st.form_submit_button("执行刷新", type="primary")
	if submitted:
		refresh_options: Dict[str, Any] = {
			"mode": selected_mode,
			"graph_scope": selected_scope,
			"retrain_epochs": retrain_epochs,
			"fusion_epochs": fusion_epochs,
		}
		if sample_ratio is not None:
			refresh_options["sample_ratio"] = sample_ratio
		trigger_full_refresh(refresh_options)

st.markdown(
	"""
	### 使用指南

	- **用户画像查询** 页面：输入用户 ID，联通后端画像与解释接口，查看最终画像分、权重与 SHAP 解释。
	- **规则调优** 页面：方便运营同学在线查看、增删改业务规则，并同步后端。

	确保后端 Flask 服务已启动，并通过 `BACKEND_BASE_URL` 环境变量或 `secrets.toml` 提供接口地址。
	"""
)
