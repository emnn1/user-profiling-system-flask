"""规则管理页面。

该页面为运营同学提供可视化的规则维护工具，涵盖：

- 查看当前规则引擎中的规则集合；
- 通过表单新增规则；
- 对已存在规则执行编辑与删除操作；
- 统一展示调用结果与错误提示。

所有对后端的请求均通过 :func:`_request` 进行封装，以确保错误处理一致性。"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st  # type: ignore[import]

from utils import call_backend, get_json


def _load_rules() -> pd.DataFrame:
    """加载规则列表并转换为 DataFrame。"""

    data = get_json("/api/v1/rules", timeout=5)
    if not data:
        return pd.DataFrame(columns=["name", "description", "weight", "condition"])
    rules = data.get("rules", [])
    return pd.DataFrame(rules)


def _render_rule_table(df: pd.DataFrame) -> None:
    """以表格形式渲染规则列表。"""

    st.subheader("📜 当前规则列表")
    if df.empty:
        st.info("尚未加载到任何规则，您可以通过下方表单新增规则。")
    else:
        # 直接以 DataFrame 展示规则清单，支持排序与筛选
        display_df = df.copy()
        st.dataframe(display_df, use_container_width=True)


st.title("画像规则管理")
st.caption("在线维护规则引擎，支持新增/编辑/删除")

rules_df = _load_rules()
_render_rule_table(rules_df)

with st.expander("➕ 新增规则", expanded=False):
    with st.form("add_rule_form"):
        name = st.text_input("规则名称")
        description = st.text_area("规则说明")
        weight = st.number_input("规则权重", value=1.0, step=0.1)
        condition = st.text_input("触发条件表达式", placeholder="monthly_fee > 300 and tenure_months > 24")
        submitted = st.form_submit_button("新增规则")
        if submitted:
            if not name:
                st.warning("规则名称不能为空")
            else:
                payload = {
                    "name": name,
                    "description": description,
                    "weight": weight,
                    "condition": condition,
                }
                call_backend(
                    "/api/v1/rules",
                    method="POST",
                    payload=payload,
                    timeout=5,
                    spinner="正在新增规则...",
                    success_message="规则创建请求已发送，刷新页面以查看最新列表。",
                )

with st.expander("✏️ 编辑规则", expanded=False):
    with st.form("edit_rule_form"):
        existing_names = rules_df["name"].tolist() if not rules_df.empty else []
        target = st.selectbox("选择要编辑的规则", options=existing_names)
        new_description = st.text_area("新的规则说明")
        new_weight = st.number_input("新的权重", value=1.0, step=0.1)
        new_condition = st.text_input("新的条件表达式")
        submitted = st.form_submit_button("提交修改")
        if submitted:
            if not target:
                st.warning("请选择需要编辑的规则")
            else:
                payload = {
                    "description": new_description,
                    "weight": new_weight,
                    "condition": new_condition,
                }
                call_backend(
                    f"/api/v1/rules/{target}",
                    method="PUT",
                    payload=payload,
                    timeout=5,
                    spinner="正在更新规则...",
                    success_message="规则更新请求已发送，刷新页面以查看最新列表。",
                )

with st.expander("🗑️ 删除规则", expanded=False):
    with st.form("delete_rule_form"):
        existing_names = rules_df["name"].tolist() if not rules_df.empty else []
        target = st.selectbox("选择要删除的规则", options=existing_names)
        confirm = st.checkbox("确认删除该规则")
        submitted = st.form_submit_button("删除")
        if submitted:
            if not target:
                st.warning("请选择需要删除的规则")
            elif not confirm:
                st.warning("请勾选确认删除复选框")
            else:
                call_backend(
                    f"/api/v1/rules/{target}",
                    method="DELETE",
                    timeout=5,
                    spinner="正在删除规则...",
                    success_message="规则删除请求已发送，刷新页面以查看最新列表。",
                )

st.info(
    "提示：若规则列表无法加载，请确认后端已提供 `/api/v1/rules` 系列接口，或在开发阶段手动更新规则配置。"
)
