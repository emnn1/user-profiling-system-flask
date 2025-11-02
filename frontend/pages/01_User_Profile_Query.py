"""用户画像查询页面。

该页面提供针对单个用户的完整洞察视图，包括：

- 画像基础信息与行为统计；
- 规则命中明细与 SHAP 可解释性分析；
- 面向运营的策略推荐列表。

页面中的所有后端请求都通过 ``utils.get_json`` 统一访问后端，
以保证多环境部署时的配置一致性。"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st  # type: ignore[import]

from utils import get_json


st.title("用户画像查询")


def _render_basic_info(profile: Dict[str, Any]) -> None:
    """渲染用户基础信息与行为统计模块。"""

    st.subheader("👤 用户基础信息")
    user_info = profile.get("user", {})
    if not user_info:
        st.info("未查询到用户基础信息")
        return
    df = pd.DataFrame([user_info])
    st.table(df.set_index("user_id"))

    st.subheader("📈 行为统计")
    event_counts = profile.get("event_counts", {})
    if event_counts:
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
    for idx, item in enumerate(recommendation.get("recommendations", []), start=1):
        st.markdown(f"- {idx}. {item}")



st.caption("结合规则与向量信息的实时画像查询工具")
