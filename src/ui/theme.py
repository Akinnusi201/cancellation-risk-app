"""Shared visual system for the Streamlit application.

The theme is intentionally CSS-only so visual polish never changes model state,
routing, or deployment behavior. The design prioritizes strong contrast, compact
information density, clear actions, and plain-language status cues.
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
  --cr-border: #e3e6ef;
  --cr-accent: #5b5bd6;
  --cr-accent-2: #756cf3;
  --cr-accent-soft: #efefff;
  --cr-good: #16835b;
  --cr-warn: #a66300;
  --cr-bad: #c23b46;
  --cr-radius: 16px;
  --cr-shadow: 0 1px 2px rgba(20, 24, 40, .04), 0 10px 28px rgba(20, 24, 40, .045);
  --cr-shadow-hover: 0 8px 24px rgba(20, 24, 40, .08);
}

/* App shell */
[data-testid="stAppViewContainer"] { background: var(--cr-bg); }
[data-testid="stAppViewContainer"] .main .block-container {
  max-width: 1320px;
  padding-top: 1.65rem;
  padding-bottom: 4rem;
}
[data-testid="stHeader"] {
  background: rgba(246,247,251,.88);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(230,232,240,.72);
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #151827 0%, #1b2032 100%);
  border-right: 1px solid rgba(255,255,255,.07);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .7rem; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] { color: #d7dbea; }
[data-testid="stSidebarNav"] a {
  border-radius: 10px;
  margin: 3px 8px;
  min-height: 42px;
  transition: background .16s ease, transform .16s ease;
}
[data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,.075); }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: rgba(117,108,243,.22);
  box-shadow: inset 3px 0 0 #9d96ff;
}
[data-testid="stSidebarNav"] a[aria-current="page"] span {
  color: #ffffff !important;
  font-weight: 700;
}

/* Typography */
h1, h2, h3, h4, h5, h6 { color: var(--cr-ink); letter-spacing: -.018em; }
h1 { font-weight: 740 !important; }
h2, h3 { font-weight: 690 !important; }
p, li { color: #34394d; }
[data-testid="stCaptionContainer"] { color: var(--cr-muted); }
[data-testid="stMarkdownContainer"] h1 { margin-bottom: .2rem; }

/* Metric cards */
[data-testid="stMetric"] {
  background: var(--cr-surface);
  border: 1px solid var(--cr-border);
  border-radius: var(--cr-radius);
  padding: .9rem 1rem;
  box-shadow: var(--cr-shadow);
  min-height: 102px;
}
[data-testid="stMetricLabel"] {
  color: var(--cr-muted);
  font-size: .78rem;
  font-weight: 680;
  letter-spacing: .01em;
}
[data-testid="stMetricValue"] {
  color: var(--cr-ink);
  font-weight: 730;
  letter-spacing: -.025em;
}

/* Buttons and page links */
.stButton > button,
.stDownloadButton > button,
.stLinkButton > a,
[data-testid="stPageLink"] a {
  border-radius: 10px !important;
  min-height: 42px;
  font-weight: 680 !important;
  transition: transform .14s ease, box-shadow .14s ease, border-color .14s ease, background .14s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stLinkButton > a:hover,
[data-testid="stPageLink"] a:hover {
  transform: translateY(-1px);
  box-shadow: var(--cr-shadow-hover);
}

/* Streamlit nests button labels inside p/span elements. Force accessible contrast. */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--cr-accent), var(--cr-accent-2)) !important;
  border: 0 !important;
  color: #ffffff !important;
  box-shadow: 0 8px 20px rgba(91,91,214,.18);
}
.stButton > button[kind="primary"] *,
button[data-testid="stBaseButton-primary"] * {
  color: #ffffff !important;
  font-weight: 760 !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(135deg, #5151cc, #6d63ec) !important;
  box-shadow: 0 10px 24px rgba(91,91,214,.28) !important;
}
.stButton > button[kind="primary"]:disabled *,
button[data-testid="stBaseButton-primary"]:disabled * { color: rgba(255,255,255,.78) !important; }

/* Make ordinary navigation links feel actionable without looking like CTAs. */
[data-testid="stPageLink"] a {
  border: 1px solid var(--cr-border);
  background: var(--cr-surface);
  color: var(--cr-ink) !important;
}
[data-testid="stPageLink"] a * { font-weight: 680 !important; }

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
button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible {
  outline: 3px solid rgba(91,91,214,.28) !important;
  outline-offset: 2px !important;
}

/* Tabs */
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
  padding-left: .9rem;
  padding-right: .9rem;
  min-height: 38px;
}
[data-baseweb="tab"][aria-selected="true"] {
  background: white;
  box-shadow: 0 1px 3px rgba(20,24,40,.10);
  font-weight: 680;
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
[data-testid="stExpander"] summary { font-weight: 650; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--cr-border);
  border-radius: 14px;
  overflow: hidden;
  background: var(--cr-surface);
}

/* Alerts */
[data-testid="stAlert"] { border-radius: 12px; border-width: 1px; }
hr { border-color: var(--cr-border) !important; margin: 1.6rem 0 !important; }

/* Sidebar brand */
.cr-brand {
  display:flex; align-items:center; gap:.68rem; padding:.7rem .55rem 1rem .55rem;
  border-bottom:1px solid rgba(255,255,255,.09); margin-bottom:.65rem;
}
.cr-brand-mark {
  width:36px; height:36px; border-radius:11px; display:grid; place-items:center;
  background:linear-gradient(135deg,#7771e8,#a27bf3); color:#fff; font-size:19px;
  box-shadow:0 8px 20px rgba(93,84,214,.28);
}
.cr-brand-name {color:#fff; font-weight:730; font-size:.92rem; line-height:1.15;}
.cr-brand-sub {color:#9da5bd; font-size:.72rem; margin-top:.12rem;}
.cr-workspace-pill {
  display:inline-flex; align-items:center; gap:.35rem; margin:.15rem .5rem .65rem;
  padding:.34rem .58rem; border-radius:999px; background:rgba(255,255,255,.07);
  color:#cbd1e3; font-size:.72rem; border:1px solid rgba(255,255,255,.08);
}

/* Page hero */
.cr-hero {
  position:relative; overflow:hidden; background:linear-gradient(135deg,#ffffff 0%,#fafaff 58%,#f1efff 100%);
  border:1px solid var(--cr-border); border-radius:20px; padding:1.4rem 1.55rem;
  margin:.15rem 0 1.3rem; box-shadow:var(--cr-shadow);
}
.cr-hero:after {
  content:""; position:absolute; width:210px; height:210px; border-radius:50%;
  right:-80px; top:-120px; background:radial-gradient(circle,rgba(117,108,243,.20),rgba(117,108,243,0) 68%);
}
.cr-eyebrow {
  display:inline-flex; align-items:center; gap:.35rem; font-size:.71rem; font-weight:760;
  letter-spacing:.06em; text-transform:uppercase; color:#6561c8; margin-bottom:.4rem;
}
.cr-hero h1 {font-size:1.95rem; margin:0 0 .3rem; line-height:1.13;}
.cr-hero p {margin:0; color:var(--cr-muted); max-width:850px; font-size:.94rem;}

.cr-section-label {
  display:flex; align-items:center; gap:.48rem; font-weight:720; color:var(--cr-ink);
  margin:1.45rem 0 .55rem; font-size:1rem;
}
.cr-section-label .cr-icon {
  width:29px; height:29px; border-radius:9px; display:grid; place-items:center;
  background:#eeedff; font-size:15px;
}

/* Landing page */
.cr-login-wrap {max-width:980px; margin:1.9rem auto 0;}
.cr-login-hero {text-align:center; margin-bottom:1.35rem;}
.cr-login-mark {
  width:54px; height:54px; margin:0 auto .72rem; border-radius:17px; display:grid; place-items:center;
  color:#fff; font-size:25px; background:linear-gradient(135deg,#5b5bd6,#8a70ef);
  box-shadow:0 12px 30px rgba(91,91,214,.22);
}
.cr-login-hero h1 {font-size:2.18rem; margin:0;}
.cr-login-hero p {color:var(--cr-muted); margin:.42rem auto 0; max-width:680px; font-size:.94rem;}
.cr-role-card {
  height:100%; min-height:132px; background:#fff; border:1px solid var(--cr-border); border-radius:16px;
  padding:1rem 1.05rem; box-shadow:var(--cr-shadow); margin-bottom:.55rem;
  transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}
.cr-role-card:hover {transform:translateY(-1px); box-shadow:var(--cr-shadow-hover); border-color:#d8d9f6;}
.cr-role-top {display:flex; align-items:center; gap:.75rem; margin-bottom:.55rem;}
.cr-role-icon {
  flex:0 0 auto; width:38px; height:38px; display:grid; place-items:center; border-radius:11px;
  background:var(--cr-accent-soft); font-size:19px;
}
.cr-role-card h3 {margin:0; font-size:1.02rem;}
.cr-role-card p {color:var(--cr-muted); font-size:.86rem; line-height:1.45; margin:0; max-width:48rem;}
.cr-role-meta {margin-top:.62rem; display:flex; flex-wrap:wrap; gap:.36rem;}
.cr-mini-pill {
  display:inline-flex; align-items:center; padding:.25rem .48rem; border-radius:999px;
  background:#f4f5f9; border:1px solid #e8eaf1; color:#62697e; font-size:.69rem; font-weight:650;
}
.cr-login-note {
  margin:.9rem auto 0; text-align:center; color:var(--cr-muted); font-size:.75rem; max-width:760px;
}

.cr-about {
  margin:1.2rem auto 0; max-width:980px; background:#fff; border:1px solid var(--cr-border);
  border-radius:18px; padding:1.15rem 1.2rem; box-shadow:var(--cr-shadow);
}
.cr-about-head {display:flex; align-items:flex-start; gap:.75rem; margin-bottom:.9rem;}
.cr-about-icon {
  width:38px; height:38px; border-radius:11px; display:grid; place-items:center;
  background:var(--cr-accent-soft); font-size:18px; flex:0 0 auto;
}
.cr-about h3 {margin:0 0 .22rem; font-size:1.05rem;}
.cr-about-intro {margin:0; color:var(--cr-muted); font-size:.86rem; line-height:1.5;}
.cr-about-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.7rem;}
.cr-about-card {background:var(--cr-surface-soft); border:1px solid #eceef4; border-radius:13px; padding:.78rem .82rem;}
.cr-about-card strong {display:block; color:var(--cr-ink); font-size:.82rem; margin-bottom:.2rem;}
.cr-about-card span {display:block; color:var(--cr-muted); font-size:.76rem; line-height:1.4;}

/* Workflow strip */
.cr-flow {display:flex; flex-wrap:wrap; gap:.45rem; align-items:center; margin:.55rem 0 1rem;}
.cr-flow-step {
  background:#fff; border:1px solid var(--cr-border); border-radius:10px;
  padding:.47rem .65rem; font-size:.78rem; font-weight:640; color:#3d4257;
}
.cr-flow-arrow {color:#969cb0; font-size:.86rem;}

/* Small status utility chips used by custom HTML where needed. */
.cr-chip {display:inline-flex; align-items:center; gap:.28rem; border-radius:999px; padding:.27rem .5rem; font-size:.7rem; font-weight:700; border:1px solid transparent;}
.cr-chip-good {background:#eaf7f1; color:#126844; border-color:#d2eee1;}
.cr-chip-warn {background:#fff5e6; color:#8c5600; border-color:#f7e2be;}
.cr-chip-neutral {background:#f2f3f7; color:#596176; border-color:#e5e7ee;}
.cr-chip-accent {background:#efefff; color:#4f4db9; border-color:#deddfb;}

/* Mobile */
@media (max-width: 760px) {
  [data-testid="stAppViewContainer"] .main .block-container {padding-top:1rem; padding-left:1rem; padding-right:1rem;}
  .cr-hero {padding:1.18rem 1.08rem; border-radius:16px;}
  .cr-hero h1 {font-size:1.62rem;}
  .cr-login-wrap {margin-top:.7rem;}
  .cr-login-hero h1 {font-size:1.78rem;}
  .cr-about-grid {grid-template-columns:1fr;}
  .cr-role-card {min-height:0;}
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
