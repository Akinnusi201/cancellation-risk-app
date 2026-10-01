import streamlit as st


ROLES = {"manager": "Operations Manager", "developer": "Developer"}


def _sign_in_as(role: str) -> None:
    if role not in ROLES:
        return
    st.session_state.authenticated = True
    st.session_state.role = role
    st.session_state.username = ROLES[role]
    st.rerun()


def login_screen() -> None:
    """Passwordless role-selection landing page for this deployment."""
    st.title("Profit-Aware Order Cancellation Risk")
    st.caption("Group 10 • Select the workspace you want to enter")

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("📦 Operations Manager")
        st.write(
            "Score incoming orders, review profit-aware intervention recommendations, "
            "and record operational decisions."
        )
        if st.button(
            "Enter Operations Workspace",
            type="primary",
            use_container_width=True,
            key="login_manager",
        ):
            _sign_in_as("manager")

    with right:
        st.subheader("🛠️ Developer")
        st.write(
            "Manage DataOps, review the model registry and MLflow experiments, "
            "monitor performance, and promote approved models."
        )
        if st.button(
            "Enter Developer Workspace",
            use_container_width=True,
            key="login_developer",
        ):
            _sign_in_as("developer")

    st.info(
        "This deployment uses passwordless role selection for the course environment. "
        "Operations and Developer pages are still registered separately so each workspace only exposes the tools intended for that role."
    )


def logout_button() -> None:
    label = ROLES.get(st.session_state.get("role"), "User")
    st.sidebar.caption(f"Workspace: **{label}**")
    if st.sidebar.button("Sign out", use_container_width=True):
        for key in [
            "authenticated",
            "role",
            "username",
            "queue_index",
            "live_queue_index",
            "historical_queue_index",
            "current_score",
            "decision_made",
            "sim_score_key",
            "manual_order_row",
            "manual_order_score",
            "manual_decision_made",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


def require_role(role: str) -> None:
    """Defense-in-depth guard for every role-specific page."""
    if not st.session_state.get("authenticated"):
        st.session_state.pop("role", None)
        st.switch_page("app.py")

    if st.session_state.get("role") != role:
        # A page file may still be invoked during development or from stale browser
        # history. Never render protected content for the wrong role.
        st.warning("That page is not available in your current workspace.")
        st.stop()
