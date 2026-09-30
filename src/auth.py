import hmac
import os
from typing import Optional

import streamlit as st


ROLES = {"manager": "Operations Manager", "developer": "Developer"}


def _secret_value(role: str, key: str) -> Optional[str]:
    try:
        auth = st.secrets.get("auth", {})
        role_cfg = auth.get(role, {}) if hasattr(auth, "get") else {}
        value = role_cfg.get(key) if hasattr(role_cfg, "get") else None
        if value is not None:
            return str(value)
    except Exception:
        pass
    env_name = f"AUTH_{role.upper()}_{key.upper()}"
    value = os.getenv(env_name)
    return value if value else None


def auth_is_configured() -> bool:
    return all(
        _secret_value(role, key)
        for role in ROLES
        for key in ("username", "password")
    )


def authenticate(username: str, password: str) -> Optional[str]:
    for role in ROLES:
        expected_user = _secret_value(role, "username")
        expected_password = _secret_value(role, "password")
        if not expected_user or not expected_password:
            continue
        if hmac.compare_digest(username.strip(), expected_user) and hmac.compare_digest(password, expected_password):
            return role
    return None


def login_screen() -> None:
    st.title("Profit-Aware Order Cancellation Risk")
    st.caption("Group 10 • Production inference separated from DataOps and ModelOps")

    if not auth_is_configured():
        st.error("Authentication is not configured for this deployment.")
        st.code(
            '[auth.manager]\nusername = "operations"\npassword = "<manager-password>"\n\n'
            '[auth.developer]\nusername = "developer"\npassword = "<developer-password>"',
            language="toml",
        )
        st.info("Add these values in Streamlit Cloud → App settings → Secrets, or use the matching AUTH_* environment variables locally.")
        return

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        role = authenticate(username, password)
        if role:
            st.session_state.authenticated = True
            st.session_state.role = role
            st.session_state.username = username.strip()
            st.rerun()
        st.error("Invalid username or password.")


def logout_button() -> None:
    label = ROLES.get(st.session_state.get("role"), "User")
    st.sidebar.caption(f"Signed in as **{label}**")
    if st.sidebar.button("Sign out", use_container_width=True):
        for key in ["authenticated", "role", "username", "queue_index", "current_score", "decision_made"]:
            st.session_state.pop(key, None)
        st.rerun()


def require_role(role: str) -> None:
    if not st.session_state.get("authenticated"):
        st.error("Sign in to access this page.")
        st.stop()
    if st.session_state.get("role") != role:
        st.error("This page is not available for your account role.")
        st.stop()
