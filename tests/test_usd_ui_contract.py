from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_operations_and_business_ui_use_usd_labels():
    paths = [
        ROOT / "views/1_Operations_Dashboard.py",
        ROOT / "views/2_Score_Order.py",
        ROOT / "views/3_Decision_History.py",
        ROOT / "views/7_Model_Monitoring.py",
        ROOT / "src/ui/common.py",
    ]
    combined = "\n".join(path.read_text() for path in paths)
    assert "Rs." not in combined
    assert "currency_caption" in combined
    assert "($)" in combined or "format_usd" in combined


def test_manual_scoring_converts_usd_back_to_pkr_for_model():
    score_text = (ROOT / "views/2_Score_Order.py").read_text()
    assert "usd_to_pkr(price_usd" in score_text
    assert "usd_to_pkr(grand_total_usd" in score_text
    assert "batch_currency == \"USD\"" in score_text


def test_usd_pages_tolerate_older_common_module():
    dashboard = (ROOT / "views/1_Operations_Dashboard.py").read_text()
    score = (ROOT / "views/2_Score_Order.py").read_text()
    history = (ROOT / "views/3_Decision_History.py").read_text()
    monitoring = (ROOT / "views/7_Model_Monitoring.py").read_text()

    assert "from src.ui.common import currency_caption" not in dashboard
    assert "from src.ui.common import currency_caption" not in history
    assert "from src.ui.common import FEATURE_LABELS" not in monitoring
    assert 'getattr(ui_common, "currency_caption", None)' in dashboard
    assert 'getattr(ui_common, "currency_caption", None)' in history
    assert 'getattr(ui_common, "currency_caption", None)' in score
    assert 'getattr(ui_common, "currency_caption", None)' in monitoring
    assert "ui_common.get_currency_context" not in score
    assert "ui_common.format_usd" not in score
