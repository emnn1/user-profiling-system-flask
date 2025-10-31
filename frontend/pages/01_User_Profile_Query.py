"""用户画像查询页面。

该页面提供针对单个用户的完整洞察视图，包括：

- 画像基础信息与行为统计；
- 规则命中明细与 SHAP 可解释性分析；
- 面向运营的策略推荐列表；
- 一键触发全量图刷新能力。

页面中的所有后端请求都以 :func:`_get_backend_base_url` 解析出的基地址为准，
从而保证多环境部署时的配置一致性。"""
from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st


@st.cache_data(show_spinner=False)
def _get_backend_base_url() -> str:
    """获取后端基地址，支持 secrets 与环境变量兜底。"""

    return (
        st.secrets.get("backend_base_url")
        or os.getenv("BACKEND_BASE_URL", "http://localhost:5000")
    ).rstrip("/")


def _fetch_json(path: str) -> Dict[str, Any] | None:
    """以 GET 请求方式访问后端接口，并处理常见错误。"""

    base_url = _get_backend_base_url()
    try:
        # 以 GET 请求的方式访问指定后端 API
        response = requests.get(f"{base_url}{path}", timeout=5)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - 前端容错
        st.error(f"调用后端接口失败: {exc}")
        return None


def _trigger_full_refresh(options: Dict[str, Any]) -> None:
    """触发后端全量图刷新流程。"""

    base_url = _get_backend_base_url()
    with st.spinner("正在刷新全量图..."):
        try:
            response = requests.post(
                f"{base_url}/api/v1/graph/refresh",
                json=options,
                timeout=300,
            )
            response.raise_for_status()
            payload = response.json()
            st.success(payload.get("message", "全量图刷新完成"))
            st.json(payload)
        except Exception as exc:  # pragma: no cover - 前端容错
            st.error(f"刷新失败: {exc}")


def _render_basic_info(profile: Dict[str, Any]) -> None:
    """渲染用户基础信息与行为统计模块。"""

    st.subheader("👤 用户基础信息")
    user_info = profile.get("user", {})
    if not user_info:
        st.info("未查询到用户基础信息")
        return
    # 使用表格展示用户基础属性
    df = pd.DataFrame([user_info])
    st.table(df.set_index("user_id"))

    st.subheader("📈 行为统计")
    event_counts = profile.get("event_counts", {})
    if event_counts:
        # 将行为次数绘制为柱状图，帮助理解活跃度
        df_counts = pd.DataFrame(
            {"事件类型": list(event_counts.keys()), "次数": list(event_counts.values())}
        ).set_index("事件类型")
        st.bar_chart(df_counts)
    else:
        st.info("暂无行为事件记录")


def _render_explanation(explanation: Dict[str, Any] | None) -> None:
    """展示 SHAP 解释与规则贡献细节。"""

    st.subheader("🧠 画像决策依据")
    if not explanation:
        st.warning("后端暂未提供可解释性结果，已回退至规则得分视图。")
        return

    cols = st.columns(3)
    cols[0].metric("最终画像得分", f"{explanation['final_score']:.3f}")
    cols[1].metric("规则权重 g", f"{explanation['gate'] * 100:.1f}%")
    cols[2].metric("规则分 / 模型分", f"{explanation['f_rule']:.3f} / {explanation['f_nn']:.3f}")

    st.markdown("**规则命中详情**")
    # 将规则命中情况转为表格方便业务人员核对
    details_df = pd.DataFrame(
        [
            {"规则": name, "贡献": value}
            for name, value in explanation.get("rule_details", {}).items()
        ]
    )
    if not details_df.empty:
        st.table(details_df.sort_values("贡献", ascending=False))

    st.markdown("**特征贡献 Top-K (SHAP)**")
    shap_values = explanation.get("shap_values", {})
    if shap_values:
        # 展示 SHAP 前 K 大正负贡献，辅助解释模型预测
        shap_df = (
            pd.DataFrame(
                {"特征": list(shap_values.keys()), "贡献": list(shap_values.values())}
            )
            .sort_values("贡献", ascending=False)
            .set_index("特征")
        )
        st.bar_chart(shap_df)
    else:
        st.info("未返回 SHAP 贡献向量，可能后端未启用解释模块。")


def _render_recommendations(recommendation: Dict[str, Any] | None) -> None:
    """渲染策略推荐列表。"""

    st.subheader("🎯 策略推荐")
    if not recommendation:
        st.info("暂无推荐，可能用户未命中业务策略。")
        return

    st.write("推荐更新时间:", recommendation.get("generated_at", "N/A"))
    # 顺序列出推荐策略，供业务人员参考
    for idx, item in enumerate(recommendation.get("recommendations", []), start=1):
        st.markdown(f"- {idx}. {item}")


st.title("用户画像查询")
st.caption("结合规则与向量信息的实时画像查询工具")

with st.expander("🔄 手动刷新全量图"):
    with st.form("profile_page_refresh_form"):
        mode_display = {
            "embedding_only": "仅刷新嵌入",
            "full_retrain": "重新训练模型",
        }
        scope_display = {
            "full": "全量图",
            "sampled": "采样图",
        }

        selected_mode = st.selectbox(
            "刷新策略",
            options=["embedding_only", "full_retrain"],
            format_func=lambda key: mode_display[key],
            key="profile_refresh_mode",
        )
        selected_scope = st.selectbox(
            "图构建范围",
            options=["full", "sampled"],
            format_func=lambda key: scope_display[key],
            key="profile_refresh_scope",
        )

        sample_ratio = None
        if selected_scope == "sampled":
            sample_ratio = st.slider(
                "采样比例",
                min_value=0.1,
                max_value=1.0,
                step=0.1,
                value=0.5,
                key="profile_refresh_ratio",
            )

        retrain_epochs = st.slider(
            "HGT 训练轮次",
            min_value=1,
            max_value=20,
            value=2,
            key="profile_refresh_retrain",
        )
        fusion_epochs = st.slider(
            "融合核心训练轮次",
            min_value=1,
            max_value=30,
            value=3,
            key="profile_refresh_fusion",
        )

        submitted = st.form_submit_button("执行刷新", type="primary")
        if submitted:
            options = {
                "mode": selected_mode,
                "graph_scope": selected_scope,
                "retrain_epochs": retrain_epochs,
                "fusion_epochs": fusion_epochs,
            }
            if sample_ratio is not None:
                options["sample_ratio"] = sample_ratio
            _trigger_full_refresh(options)

user_id = st.text_input("输入用户 ID", value="")
trigger = st.button("查询画像", type="primary")

if trigger and user_id:
    profile_response = _fetch_json(f"/api/v1/user/{user_id}")
    if profile_response is None:
        st.error("未找到该用户的画像信息。")
    else:
        profile = profile_response.get("profile", {})
        _render_basic_info(profile)

        # 可解释性结果
        explanation = _fetch_json(f"/api/v1/explain/{user_id}")
        _render_explanation(explanation)

        recommendation = _fetch_json(f"/api/v1/recommendation/{user_id}")
        _render_recommendations(recommendation)
elif trigger and not user_id:
    st.warning("请先输入用户 ID 再执行查询。")
else:
    st.info("输入用户 ID 并点击查询，以获取画像、解释与推荐信息。")
