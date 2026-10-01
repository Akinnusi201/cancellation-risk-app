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
    """Passwordless role-selection landing page for this course deployment."""
    st.markdown(
        """
        <div class="cr-login-wrap">
          <div class="cr-login-hero">
            <div class="cr-login-mark">↗</div>
            <h1>Profit-Aware Cancellation Risk</h1>
            <p>Turn cancellation risk into an operational decision: release the order, or verify it before fulfillment.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Keep role cards intentionally compact on wide screens. Streamlit's main
    # container can be very wide, so a centered inner column prevents the cards
    # and descriptions from looking stretched.
    _, center, _ = st.columns([0.65, 10, 0.65])
    with center:
        left, right = st.columns(2, gap="medium")

        with left:
            st.markdown(
                """
                <div class="cr-role-card">
                  <div class="cr-role-top">
                    <div class="cr-role-icon">📦</div>
                    <h3>Operations Manager</h3>
                  </div>
                  <p>Review incoming orders, see cancellation risk and expected dollar impact, then release the order or keep it for verification.</p>
                  <div class="cr-role-meta">
                    <span class="cr-mini-pill">Order review</span>
                    <span class="cr-mini-pill">Business impact</span>
                    <span class="cr-mini-pill">Decision history</span>
                  </div>
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
                  <div class="cr-role-top">
                    <div class="cr-role-icon">🛠️</div>
                    <h3>Developer</h3>
                  </div>
                  <p>Manage DataOps, run tracked MLflow experiments, compare models, monitor production health, retrain, and control deployment.</p>
                  <div class="cr-role-meta">
                    <span class="cr-mini-pill">DataOps</span>
                    <span class="cr-mini-pill">MLflow + ModelOps</span>
                    <span class="cr-mini-pill">Monitoring</span>
                  </div>
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
        """
        <div class="cr-about">
          <div class="cr-about-head">
            <div class="cr-about-icon">ℹ️</div>
            <div>
              <h3>About this tool</h3>
              <p class="cr-about-intro">Developed by <strong>Group 10</strong> for <strong>BANA 7075 · Machine Learning Design for Business</strong>. The project demonstrates a complete ML system for e-commerce cancellation risk, with separate operational and developer workflows.</p>
            </div>
          </div>
          <div class="cr-about-grid">
            <div class="cr-about-card">
              <strong>📊 Business purpose</strong>
              <span>Estimate cancellation risk and use expected economic value to decide whether an order is worth verifying before fulfillment.</span>
            </div>
            <div class="cr-about-card">
              <strong>🔁 End-to-end ML lifecycle</strong>
              <span>Data ingestion, validation, versioning, five-model experimentation, MLflow tracking, governed deployment, monitoring, and retraining.</span>
            </div>
            <div class="cr-about-card">
              <strong>🧪 Course deployment</strong>
              <span>Passwordless role selection keeps the demo easy to access. Role-specific navigation and page guards still isolate Operations from Developer tools.</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="cr-login-note">Group 10 · Profit-Aware E-Commerce Order Cancellation Risk · Educational deployment</div>',
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
        st.warning("That page is not available in your current workspace.")
        st.stop()
