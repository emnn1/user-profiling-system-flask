"""数据配置页面。

该页面允许管理员调整数据生成参数并重新生成模拟数据。
"""
import streamlit as st  # type: ignore[import]
import sys
from pathlib import Path

# 添加项目根目录到路径以便导入 utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import call_backend, get_json

st.set_page_config(
    page_title="数据配置 - 用户画像系统",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ 数据配置管理")
st.markdown("---")

# 获取当前配置
def fetch_current_config():
    """从后端获取当前数据生成配置。"""
    return get_json("/api/v1/data/config", timeout=5)


# 更新配置
def update_config(config_data):
    """更新数据生成配置。"""
    return call_backend(
        "/api/v1/data/config",
        method="POST",
        payload=config_data,
        timeout=10,
        spinner="正在更新配置...",
    )


# 重新生成数据
def regenerate_data():
    """触发数据重新生成。"""
    return call_backend(
        "/api/v1/data/regenerate",
        method="POST",
        timeout=300,
        spinner="正在重新生成数据,请稍候...",
    )


# 显示当前配置
st.subheader("📊 当前配置")
current_config = fetch_current_config()

if current_config:
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("用户数量", f"{current_config['user_count']:,}")
        st.metric("商品数量", f"{current_config['product_count']:,}")
        st.metric("APP数量", f"{current_config['app_count']:,}")
    
    with col2:
        st.metric("每用户平均事件数", current_config["avg_events_per_user"])
        st.metric("历史数据天数", current_config["history_days"])

    st.markdown("---")
    
    # 配置更新表单
    st.subheader("🔧 更新配置")
    
    with st.form("config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            user_count = st.number_input(
                "用户数量",
                min_value=10,
                max_value=100000,
                value=current_config.get("user_count", 1000),
                step=100,
                help="范围: 10 - 100,000"
            )
            
            product_count = st.number_input(
                "商品数量",
                min_value=5,
                max_value=1000,
                value=current_config.get("product_count", 25),
                step=10,
                help="范围: 5 - 1,000"
            )
            
            app_count = st.number_input(
                "APP数量",
                min_value=5,
                max_value=500,
                value=current_config.get("app_count", 30),
                step=5,
                help="范围: 5 - 500"
            )
        
        with col2:
            avg_events_per_user = st.number_input(
                "每用户平均事件数",
                min_value=1,
                max_value=1000,
                value=current_config.get("avg_events_per_user", 20),
                step=10,
                help="范围: 1 - 1,000"
            )
            
            history_days = st.number_input(
                "历史数据天数",
                min_value=1,
                max_value=365,
                value=current_config.get("history_days", 30),
                step=7,
                help="范围: 1 - 365"
            )
        
        st.markdown("---")
        st.subheader("🕸️ 初始随机边（每用户）")

        # 读取当前（或默认）边参数
        def _cfg(key: str, default: int) -> int:
            try:
                return int(current_config.get(key, default))
            except Exception:
                return default

        # 订购边
        c1, c2 = st.columns(2)
        with c1:
            min_orders_per_user = st.number_input(
                "最少订购边数",
                min_value=0,
                max_value=100,
                value=_cfg("min_orders_per_user", 1),
                step=1,
            )
        with c2:
            max_orders_per_user = st.number_input(
                "最多订购边数",
                min_value=0,
                max_value=100,
                value=_cfg("max_orders_per_user", 3),
                step=1,
            )

        # APP 使用边
        c3, c4 = st.columns(2)
        with c3:
            min_app_usages_per_user = st.number_input(
                "最少 APP 使用边数",
                min_value=0,
                max_value=100,
                value=_cfg("min_app_usages_per_user", 1),
                step=1,
            )
        with c4:
            max_app_usages_per_user = st.number_input(
                "最多 APP 使用边数",
                min_value=0,
                max_value=100,
                value=_cfg("max_app_usages_per_user", 3),
                step=1,
            )

        # 通话边
        c5, c6 = st.columns(2)
        with c5:
            min_calls_per_user = st.number_input(
                "最少通话边数",
                min_value=0,
                max_value=100,
                value=_cfg("min_calls_per_user", 0),
                step=1,
            )
        with c6:
            max_calls_per_user = st.number_input(
                "最多通话边数",
                min_value=0,
                max_value=100,
                value=_cfg("max_calls_per_user", 2),
                step=1,
            )

        # 点击边（商品）
        c7, c8 = st.columns(2)
        with c7:
            min_click_products_per_user = st.number_input(
                "最少点击商品边数",
                min_value=0,
                max_value=100,
                value=_cfg("min_click_products_per_user", 0),
                step=1,
            )
        with c8:
            max_click_products_per_user = st.number_input(
                "最多点击商品边数",
                min_value=0,
                max_value=100,
                value=_cfg("max_click_products_per_user", 5),
                step=1,
            )

        # 点击边（APP）
        c9, c10 = st.columns(2)
        with c9:
            min_click_apps_per_user = st.number_input(
                "最少点击 APP 边数",
                min_value=0,
                max_value=100,
                value=_cfg("min_click_apps_per_user", 0),
                step=1,
            )
        with c10:
            max_click_apps_per_user = st.number_input(
                "最多点击 APP 边数",
                min_value=0,
                max_value=100,
                value=_cfg("max_click_apps_per_user", 5),
                step=1,
            )

        submitted = st.form_submit_button("💾 保存配置")
        
        if submitted:
            new_config = {
                "user_count": int(user_count),
                "product_count": int(product_count),
                "app_count": int(app_count),
                "avg_events_per_user": int(avg_events_per_user),
                "history_days": int(history_days),
                "min_orders_per_user": int(min_orders_per_user),
                "max_orders_per_user": int(max_orders_per_user),
                "min_app_usages_per_user": int(min_app_usages_per_user),
                "max_app_usages_per_user": int(max_app_usages_per_user),
                "min_calls_per_user": int(min_calls_per_user),
                "max_calls_per_user": int(max_calls_per_user),
                "min_click_products_per_user": int(min_click_products_per_user),
                "max_click_products_per_user": int(max_click_products_per_user),
                "min_click_apps_per_user": int(min_click_apps_per_user),
                "max_click_apps_per_user": int(max_click_apps_per_user),
            }
            
            result = update_config(new_config)
            if result is not None:
                st.success("✅ 配置已成功更新!")
                if isinstance(result, dict) and "config" in result:
                    st.json(result["config"])
                st.rerun()
    
    st.markdown("---")
    
    # 数据重新生成
    st.subheader("🔄 重新生成数据")
    st.warning(
        "⚠️ **警告**: 此操作将清空所有现有数据并使用当前配置重新生成。"
        "该过程可能需要较长时间,请确保系统处于空闲状态。"
    )
    
    if st.button("🚀 开始重新生成", type="primary"):
        result = regenerate_data()
        if result is not None:
            st.success("✅ 数据重新生成完成!")
            st.json(result)
            st.balloons()
else:
    st.error("无法获取当前配置,请检查后端服务是否正常运行。")

# 添加说明
st.markdown("---")
st.subheader("📖 参数说明")

with st.expander("查看参数详细说明"):
    st.markdown("""
    ### 用户数量
    - **范围**: 10 - 100,000
    - **说明**: 系统中模拟用户的总数量
    - **建议**: 开发环境建议 1,000 - 5,000,生产环境可增至 10,000+
    
    ### 商品数量
    - **范围**: 5 - 1,000
    - **说明**: 系统中商品的总数量
    - **建议**: 根据业务场景调整,通常 100 - 500 即可
    
    ### APP数量
    - **范围**: 5 - 500
    - **说明**: 用户可使用的APP总数
    - **建议**: 移动生态一般 50 - 200 个APP
    
    ### 每用户平均事件数
    - **范围**: 1 - 1,000
    - **说明**: 每个用户在历史期间内产生的平均事件数量
    - **建议**: 活跃用户建议 50 - 200,可根据业务调整
    
    ### 历史数据天数
    - **范围**: 1 - 365
    - **说明**: 生成历史数据的时间跨度
    - **建议**: 一般 30 - 90 天,长期分析可用 180 - 365 天
    
    ### 注意事项
    - 参数越大,生成数据耗时越长
    - 大量数据会占用更多存储和内存
    - 修改配置后需手动触发"重新生成"
    - 重新生成会清空现有数据和模型状态
    """)

st.markdown("---")
st.caption("💡 提示: 修改配置后记得点击'保存配置'按钮,然后根据需要触发'重新生成数据'操作。")
