import streamlit as st

from src.auth import login_screen, logout_button
from src.bootstrap import initialize_runtime

try:
    from src.ui.theme import apply_app_theme, render_sidebar_brand
except Exception:
    apply_app_theme = lambda: None
    render_sidebar_brand = lambda workspace=None: None

st.set_page_config(
    page_title="Cancellation Risk ML",
    page_icon="↗️",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": "Cancellation Risk ML · Profit-aware e-commerce cancellation risk · Group 10",
    },
)

apply_app_theme()


def login_page():
    login_screen()


if not st.session_state.get("authenticated"):
    # Always initialize Streamlit navigation, even before sign-in. This prevents the
    # special /pages auto-discovery behavior from exposing application pages.
    pg = st.navigation(
        [st.Page(login_page, title="Sign in", icon="🔐", default=True)],
        position="hidden",
    )
    pg.run()
else:
    # Initialize the runtime only after a workspace is selected so the landing page
    # never depends on DuckDB, MLflow, or model state. This still never retrains.
    initialize_runtime()
    role = st.session_state.get("role")

    if role == "manager":
        pages = {
            "Operations": [
                st.Page("views/1_Operations_Dashboard.py", title="Operations Dashboard", icon="📊", default=True),
                st.Page("views/2_Score_Order.py", title="Score Order", icon="🛒"),
                st.Page("views/3_Decision_History.py", title="Decision History", icon="🧾"),
            ]
        }
    elif role == "developer":
        pages = {
            "Developer": [
                st.Page("views/4_Developer_Dashboard.py", title="Developer Dashboard", icon="🧭", default=True),
                st.Page("views/5_DataOps.py", title="DataOps", icon="📦"),
                st.Page("views/6_ModelOps.py", title="Model Registry", icon="🤖"),
                st.Page("views/7_Model_Monitoring.py", title="Model Monitoring", icon="📈"),
                st.Page("views/8_MLflow_Experiments.py", title="Experiments", icon="🧪"),
                st.Page("views/9_System_Status.py", title="System Status", icon="⚙️"),
            ]
        }
    else:
        # Invalid/stale session state: return to the role-selection landing page.
        for key in ["authenticated", "role", "username"]:
            st.session_state.pop(key, None)
        st.rerun()

    render_sidebar_brand("Operations workspace" if role == "manager" else "Developer workspace")
    logout_button()
    pg = st.navigation(pages, position="sidebar")
    pg.run()
