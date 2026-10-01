from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_primary_buttons_force_white_bold_text():
    theme = (ROOT / "src" / "ui" / "theme.py").read_text()
    assert 'button[data-testid="stBaseButton-primary"] *' in theme
    assert 'color: #ffffff !important' in theme
    assert 'font-weight: 760 !important' in theme


def test_landing_page_has_course_about_section_and_compact_role_layout():
    auth = (ROOT / "src" / "auth.py").read_text()
    theme = (ROOT / "src" / "ui" / "theme.py").read_text()
    assert "About this tool" in auth
    assert "BANA 7075" in auth
    assert "Group 10" in auth
    assert "cr-about-grid" in auth
    assert "cr-role-top" in auth
    assert "max-width:980px" in theme


def test_landing_page_explains_both_workspaces():
    auth = (ROOT / "src" / "auth.py").read_text()
    assert "Operations Manager" in auth
    assert "Developer" in auth
    assert "DataOps" in auth
    assert "MLflow" in auth
    assert "Business impact" in auth
