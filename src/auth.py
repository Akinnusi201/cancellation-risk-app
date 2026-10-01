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
    st.markdown(
        """
        <div class="cr-login-wrap">
          <div class="cr-login-hero">
            <div class="cr-login-mark">↗</div>
            <h1>Profit-Aware Cancellation Risk</h1>
            <p>Profit-aware order cancellation risk for e-commerce operations. Choose a workspace to enter the system.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            """
            <div class="cr-role-card">
              <div class="cr-role-icon">📦</div>
              <h3>Operations Manager</h3>
              <p>Review incoming orders, see cancellation risk and expected value, then release orders or keep them for verification.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "📦  Enter Operations Workspace",
            type="primary",
            use_container_width=True,
            key="login_manager",
        ):
            _sign_in_as("manager")

    with right:
        st.markdown(
            """
            <div class="cr-role-card">
              <div class="cr-role-icon">🛠️</div>
              <h3>Developer</h3>
              <p>Manage DataOps, run MLflow experiments, compare models, monitor production health, retrain, and control deployment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "🛠️  Enter Developer Workspace",
            use_container_width=True,
            key="login_developer",
        ):
            _sign_in_as("developer")

    st.markdown(
        '<div class="cr-login-note">Course deployment · Passwordless role selection · Workspaces are still isolated by role-specific navigation and page guards.</div>',
        unsafe_allow_html=True,
    )


def logout_button() -> None:
    label = ROLES.get(st.session_state.get("role"), "User")
    st.sidebar.caption(f"Signed in as **{label}**")
    if st.sidebar.button("↩️  Sign out", use_container_width=True):
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
