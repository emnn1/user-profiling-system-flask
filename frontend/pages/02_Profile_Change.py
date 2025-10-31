import streamlit as st
import requests

BASE = st.secrets.get("API_BASE", "http://localhost:5000/api/v1")

st.header("规则管理")

if st.button("刷新规则列表"):
    r = requests.get(f"{BASE}/rules")
    if r.ok:
        st.json(r.json())
    else:
        st.error(r.text)

st.subheader("新增规则")
with st.form("create_rule"):
    name = st.text_input("名称")
    description = st.text_area("描述", "")
    weight = st.number_input("权重", min_value=0.0, value=1.0)
    condition = st.text_input("条件表达式", "monthly_fee >= 129")
    sub = st.form_submit_button("创建")
    if sub and name and condition:
        r = requests.post(f"{BASE}/rules", json={
            "name": name,
            "description": description,
            "weight": weight,
            "condition": condition,
        })
        st.write(r.status_code)
        st.write(r.text)

st.subheader("更新/删除规则")
rule_name = st.text_input("要更新/删除的规则名")
col1, col2 = st.columns(2)
with col1.form("update_rule"):
    description = st.text_input("新描述", "")
    weight = st.number_input("新权重(可留空)", min_value=0.0, value=1.0)
    condition = st.text_input("新条件(可留空)", "")
    sub = st.form_submit_button("更新")
    if sub and rule_name:
        payload = {}
        if description:
            payload["description"] = description
        if condition:
            payload["condition"] = condition
        payload["weight"] = weight
        r = requests.put(f"{BASE}/rules/{rule_name}", json=payload)
        st.write(r.status_code)
        st.write(r.text)
with col2:
    if st.button("删除规则") and rule_name:
        r = requests.delete(f"{BASE}/rules/{rule_name}")
        st.write(r.status_code)
        st.write(r.text)
