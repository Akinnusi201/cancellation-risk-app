import streamlit as st

from src.auth import login_screen, logout_button
from src.bootstrap import initialize_runtime

st.set_page_config(page_title="Cancellation Risk Operations", page_icon="🛒", layout="wide")
initialize_runtime()

if not st.session_state.get("authenticated"):
    login_screen()
    st.stop()

logout_button()
role = st.session_state.get("role")

if role == "manager":
    pages = {
        "Operations": [
            st.Page("pages/1_Operations_Dashboard.py", title="Operations Dashboard", icon="📊", default=True),
            st.Page("pages/2_Score_Order.py", title="Score Order", icon="🛒"),
            st.Page("pages/3_Decision_History.py", title="Decision History", icon="🧾"),
        ]
    }
else:
    pages = {
        "Developer": [
            st.Page("pages/4_Developer_Dashboard.py", title="Developer Dashboard", icon="🧭", default=True),
            st.Page("pages/5_DataOps.py", title="DataOps", icon="📦"),
            st.Page("pages/6_ModelOps.py", title="ModelOps", icon="🤖"),
            st.Page("pages/7_Model_Monitoring.py", title="Model Monitoring", icon="📈"),
            st.Page("pages/8_MLflow_Experiments.py", title="MLflow Experiments", icon="🧪"),
            st.Page("pages/9_System_Status.py", title="System Status", icon="⚙️"),
        ]
    }

pg = st.navigation(pages)
pg.run()
