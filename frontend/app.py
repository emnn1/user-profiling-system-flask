import streamlit as st

st.set_page_config(page_title="User Profiling System", page_icon="🧭", layout="wide")

st.title("User Profiling System (Flask)")

st.markdown(
    """
- 左侧 Pages 可进入功能页面：
  - 01_User_Profile_Query: 查询画像/推荐/解释
  - 02_Profile_Change: 规则列表与增删改
  - 03_System_Operations: 运维操作与健康检查

后端默认监听 http://localhost:5000
    """
)
