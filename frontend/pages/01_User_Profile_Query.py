import streamlit as st
import requests

BASE = st.secrets.get("API_BASE", "http://localhost:5000/api/v1")

st.header("用户画像查询与解释")
user_id = st.text_input("用户ID", value="")

col1, col2, col3 = st.columns(3)

if col1.button("获取画像", use_container_width=True) and user_id:
    r = requests.get(f"{BASE}/user/{user_id}")
    if r.ok:
        st.json(r.json())
    else:
        st.error(r.text)

if col2.button("获取推荐", use_container_width=True) and user_id:
    r = requests.get(f"{BASE}/recommendation/{user_id}")
    if r.ok:
        st.json(r.json())
    else:
        st.error(r.text)

if col3.button("解释画像", use_container_width=True) and user_id:
    r = requests.get(f"{BASE}/explain/{user_id}")
    if r.ok:
        st.json(r.json())
    else:
        st.error(r.text)
