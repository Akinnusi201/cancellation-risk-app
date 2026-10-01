"""Shared visual system for the Streamlit application.

The theme intentionally stays CSS-only so it does not change application state,
model logic, routing, or deployment behavior.  It is inspired by the clarity of
modern SaaS dashboards: strong hierarchy, restrained color, obvious actions,
and status-first information density.
"""
from __future__ import annotations

import streamlit as st


APP_CSS = r"""
<style>
:root {
  --cr-bg: #f6f7fb;
  --cr-surface: #ffffff;
  --cr-surface-soft: #f9fafc;
  --cr-ink: #171a2b;
  --cr-muted: #687086;
  --cr-border: #e6e8f0;
  --cr-accent: #5b5bd6;
  --cr-accent-2: #756cf3;
  --cr-good: #16835b;
  --cr-warn: #a66300;
  --cr-bad: #c23b46;
  --cr-radius: 16px;
  --cr-shadow: 0 1px 2px rgba(20, 24, 40, .04), 0 10px 28px rgba(20, 24, 40, .045);
}

/* App shell */
[data-testid="stAppViewContainer"] {
  background: var(--cr-bg);
}
[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1320px;
  padding-top: 2.0rem;
  padding-bottom: 4rem;
}
[data-testid="stHeader"] {
  background: rgba(246,247,251,.88);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(230,232,240,.72);
}

/* Sidebar: deliberately quieter than the workspace */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #151827 0%, #1b2032 100%);
  border-right: 1px solid rgba(255,255,255,.07);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding-top: .7rem;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
  color: #d7dbea;
}
[data-testid="stSidebarNav"] a {
  border-radius: 10px;
  margin: 3px 8px;
  min-height: 42px;
  transition: background .16s ease, transform .16s ease;
}
[data-testid="stSidebarNav"] a:hover {
  background: rgba(255,255,255,.075);
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: rgba(117,108,243,.22);
  box-shadow: inset 3px 0 0 #9d96ff;
}
[data-testid="stSidebarNav"] a[aria-current="page"] span {
  color: #ffffff !important;
  font-weight: 650;
}

/* Typography */
h1, h2, h3, h4, h5, h6 { color: var(--cr-ink); letter-spacing: -.018em; }
h1 { font-weight: 730 !important; }
h2, h3 { font-weight: 680 !important; }
p, li { color: #34394d; }
[data-testid="stCaptionContainer"] { color: var(--cr-muted); }

/* Native titles get a little more breathing room */
[data-testid="stMarkdownContainer"] h1 {
  margin-bottom: .2rem;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: var(--cr-surface);
  border: 1px solid var(--cr-border);
  border-radius: var(--cr-radius);
  padding: 1rem 1.05rem;
  box-shadow: var(--cr-shadow);
  min-height: 108px;
}
[data-testid="stMetricLabel"] {
  color: var(--cr-muted);
  font-size: .79rem;
  font-weight: 650;
  letter-spacing: .012em;
}
[data-testid="stMetricValue"] {
  color: var(--cr-ink);
  font-weight: 720;
  letter-spacing: -.025em;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
  border-radius: 10px !important;
  min-height: 42px;
  font-weight: 650;
  transition: transform .14s ease, box-shadow .14s ease, border-color .14s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stLinkButton > a:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(29,34,58,.08);
}
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-2));
  border: 0;
  color: #fff;
}

/* Inputs */
[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div {
  border-radius: 10px !important;
  border-color: #dfe2eb !important;
  background: #fff !important;
}
[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="textarea"] > div:focus-within {
  border-color: #7771e8 !important;
  box-shadow: 0 0 0 3px rgba(91,91,214,.10) !important;
}

/* Tabs: compact segmented-control feel */
[data-baseweb="tab-list"] {
  gap: .25rem;
  background: #eceef5;
  padding: .3rem;
  border-radius: 12px;
  width: fit-content;
  max-width: 100%;
  margin-bottom: 1rem;
}
[data-baseweb="tab"] {
  border-radius: 9px;
  padding-left: 1rem;
  padding-right: 1rem;
  min-height: 38px;
}
[data-baseweb="tab"][aria-selected="true"] {
  background: white;
  box-shadow: 0 1px 3px rgba(20,24,40,.10);
}
[data-baseweb="tab-highlight"] { display: none; }

/* Expanders and data containers */
[data-testid="stExpander"] {
  background: var(--cr-surface);
  border: 1px solid var(--cr-border) !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 2px rgba(20,24,40,.025);
  overflow: hidden;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--cr-border);
  border-radius: 14px;
  overflow: hidden;
  background: var(--cr-surface);
}

/* Alerts */
[data-testid="stAlert"] {
  border-radius: 12px;
  border-width: 1px;
}

/* Dividers */
hr { border-color: var(--cr-border) !important; margin: 1.75rem 0 !important; }

/* Custom brand/header pieces */
.cr-brand {
  display:flex; align-items:center; gap:.68rem; padding:.7rem .55rem 1rem .55rem;
  border-bottom:1px solid rgba(255,255,255,.09); margin-bottom:.65rem;
}
.cr-brand-mark {
  width:36px; height:36px; border-radius:11px; display:grid; place-items:center;
  background:linear-gradient(135deg,#7771e8,#a27bf3); color:#fff; font-size:19px;
  box-shadow:0 8px 20px rgba(93,84,214,.28);
}
.cr-brand-name {color:#fff; font-weight:720; font-size:.92rem; line-height:1.15;}
.cr-brand-sub {color:#9da5bd; font-size:.72rem; margin-top:.12rem;}
.cr-workspace-pill {
  display:inline-flex; align-items:center; gap:.35rem; margin:.15rem .5rem .65rem;
  padding:.34rem .58rem; border-radius:999px; background:rgba(255,255,255,.07);
  color:#cbd1e3; font-size:.72rem; border:1px solid rgba(255,255,255,.08);
}

.cr-hero {
  position:relative; overflow:hidden; background:linear-gradient(135deg,#ffffff 0%,#fafaff 58%,#f1efff 100%);
  border:1px solid var(--cr-border); border-radius:22px; padding:1.55rem 1.65rem;
  margin:.15rem 0 1.45rem; box-shadow:var(--cr-shadow);
}
.cr-hero:after {
  content:""; position:absolute; width:210px; height:210px; border-radius:50%;
  right:-80px; top:-120px; background:radial-gradient(circle,rgba(117,108,243,.20),rgba(117,108,243,0) 68%);
}
.cr-eyebrow {
  display:inline-flex; align-items:center; gap:.35rem; font-size:.72rem; font-weight:750;
  letter-spacing:.06em; text-transform:uppercase; color:#6561c8; margin-bottom:.42rem;
}
.cr-hero h1 {font-size:2rem; margin:0 0 .35rem; line-height:1.13;}
.cr-hero p {margin:0; color:var(--cr-muted); max-width:850px; font-size:.96rem;}

.cr-section-label {
  display:flex; align-items:center; gap:.48rem; font-weight:720; color:var(--cr-ink);
  margin:1.55rem 0 .55rem; font-size:1.02rem;
}
.cr-section-label .cr-icon {
  width:29px; height:29px; border-radius:9px; display:grid; place-items:center;
  background:#eeedff; font-size:15px;
}

.cr-login-wrap {max-width:1040px; margin:2.3rem auto 0;}
.cr-login-hero {text-align:center; margin-bottom:1.65rem;}
.cr-login-mark {
  width:58px; height:58px; margin:0 auto .8rem; border-radius:18px; display:grid; place-items:center;
  color:#fff; font-size:28px; background:linear-gradient(135deg,#5b5bd6,#8a70ef);
  box-shadow:0 14px 34px rgba(91,91,214,.25);
}
.cr-login-hero h1 {font-size:2.35rem; margin:0;}
.cr-login-hero p {color:var(--cr-muted); margin:.45rem auto 0; max-width:700px;}
.cr-role-card {
  min-height:172px; background:#fff; border:1px solid var(--cr-border); border-radius:18px;
  padding:1.1rem 1.2rem .7rem; box-shadow:var(--cr-shadow); margin-bottom:.65rem;
}
.cr-role-icon {
  width:42px; height:42px; display:grid; place-items:center; border-radius:12px;
  background:#f0efff; font-size:21px; margin-bottom:.65rem;
}
.cr-role-card h3 {margin:.05rem 0 .32rem; font-size:1.1rem;}
.cr-role-card p {color:var(--cr-muted); font-size:.9rem; margin:0;}
.cr-login-note {
  margin:1.1rem auto 0; text-align:center; color:var(--cr-muted); font-size:.78rem; max-width:760px;
}

.cr-flow {
  display:flex; flex-wrap:wrap; gap:.45rem; align-items:center; margin:.55rem 0 1rem;
}
.cr-flow-step {
  background:#fff; border:1px solid var(--cr-border); border-radius:10px;
  padding:.47rem .65rem; font-size:.78rem; font-weight:620; color:#3d4257;
}
.cr-flow-arrow {color:#969cb0; font-size:.86rem;}

/* Slightly cleaner mobile behavior */
@media (max-width: 760px) {
  [data-testid="stAppViewContainer"] .main .block-container {padding-top:1.2rem; padding-left:1rem; padding-right:1rem;}
  .cr-hero {padding:1.25rem 1.15rem; border-radius:16px;}
  .cr-hero h1 {font-size:1.65rem;}
  .cr-login-wrap {margin-top:1rem;}
  .cr-login-hero h1 {font-size:1.85rem;}
}
</style>
"""


def apply_app_theme() -> None:
    """Inject the shared design system once per Streamlit rerun."""
    st.markdown(APP_CSS, unsafe_allow_html=True)


def page_hero(icon: str, title: str, description: str, eyebrow: str | None = None) -> None:
    eyebrow_html = f'<div class="cr-eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="cr-hero">
          {eyebrow_html}
          <h1>{icon}&nbsp; {title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="cr-section-label"><span class="cr-icon">{icon}</span><span>{title}</span></div>',
        unsafe_allow_html=True,
    )


def render_sidebar_brand(workspace: str | None = None) -> None:
    st.sidebar.markdown(
        """
        <div class="cr-brand">
          <div class="cr-brand-mark">↗</div>
          <div>
            <div class="cr-brand-name">Cancellation Risk ML</div>
            <div class="cr-brand-sub">Profit-aware cancellation risk</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if workspace:
        st.sidebar.markdown(
            f'<div class="cr-workspace-pill">● &nbsp; {workspace}</div>',
            unsafe_allow_html=True,
        )


def workflow_strip(steps: list[tuple[str, str]]) -> None:
    parts = []
    for idx, (icon, label) in enumerate(steps):
        if idx:
            parts.append('<span class="cr-flow-arrow">→</span>')
        parts.append(f'<span class="cr-flow-step">{icon}&nbsp; {label}</span>')
    st.markdown('<div class="cr-flow">' + ''.join(parts) + '</div>', unsafe_allow_html=True)
