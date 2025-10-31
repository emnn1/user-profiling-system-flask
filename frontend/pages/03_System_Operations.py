import streamlit as st
import requests

API = st.secrets.get("API_BASE", "http://localhost:5000/api/v1")
HEALTH = "http://localhost:5000/health"

st.header("系统运维与健康")

col1, col2 = st.columns(2)
with col1:
    if st.button("健康检查"):
        r = requests.get(HEALTH)
        if r.ok:
            data = r.json()
            mode = data.get("device_mode", "unknown").upper()
            st.markdown(f"**设备模式**: :green[{mode}]  |  CUDA: {data.get('cuda_available')}")
            st.json(data)
        else:
            st.error(r.text)
with col2:
    if st.button("总体状态"):
        r = requests.get(f"{API}/operations/status")
        st.json(r.json() if r.ok else r.text)

st.subheader("摄取与增量循环")
col3, col4, col5, col6 = st.columns(4)
if col3.button("启动摄取"):
    st.json(requests.post(f"{API}/operations/ingestion/start").json())
if col4.button("停止摄取"):
    st.json(requests.post(f"{API}/operations/ingestion/stop").json())
if col5.button("启动增量"):
    st.json(requests.post(f"{API}/operations/incremental/start").json())
if col6.button("停止增量"):
    st.json(requests.post(f"{API}/operations/incremental/stop").json())

st.subheader("规则/解释/关机")
col7, col8, col9 = st.columns(3)
if col7.button("刷新规则结构"):
    st.json(requests.post(f"{API}/operations/rules/refresh").json())
if col8.button("清空解释缓存"):
    st.json(requests.post(f"{API}/operations/explainer/clear").json())
if col9.button("停止后端(优雅)"):
    st.json(requests.post(f"{API}/operations/shutdown").json())

st.subheader("融合训练与全量刷新")
with st.form("fusion_train"):
    st.write("融合核心训练(占位)")
    sample_size = st.number_input("sample_size", value=256, min_value=32, max_value=10000)
    epochs = st.number_input("epochs", value=3, min_value=1, max_value=200)
    lr = st.number_input("lr", value=1e-3, min_value=1e-5, max_value=1e-1, format="%.5f")
    batch_size = st.number_input("batch_size", value=64, min_value=8, max_value=1024)
    if st.form_submit_button("开始训练"):
        r = requests.post(f"{API}/operations/fusion/train", json={
            "sample_size": int(sample_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "batch_size": int(batch_size),
        })
        st.json(r.json() if r.ok else r.text)

st.divider()

with st.form("graph_refresh"):
    st.write("全量图刷新")
    mode = st.selectbox("模式", ["embedding_only", "retrain_model", "retrain_and_fusion"])
    scope = st.selectbox("范围", ["full", "sampled"])
    sample_ratio = st.slider("sample_ratio (sampled 生效)", 0.1, 1.0, 0.5, step=0.1)
    retrain_epochs = st.number_input("retrain_epochs", 1, 50, 2)
    fusion_epochs = st.number_input("fusion_epochs", 1, 100, 3)
    if st.form_submit_button("刷新"):
        payload = {
            "mode": mode,
            "graph_scope": scope,
            "sample_ratio": float(sample_ratio) if scope == "sampled" else None,
            "retrain_epochs": int(retrain_epochs),
            "fusion_epochs": int(fusion_epochs),
        }
        r = requests.post(f"{API}/graph/refresh", json=payload)
        st.json(r.json() if r.ok else r.text)
