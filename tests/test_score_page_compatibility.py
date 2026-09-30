from pathlib import Path


def test_score_page_does_not_hard_import_optional_historical_helper():
    text = Path("views/2_Score_Order.py").read_text(encoding="utf-8")
    assert "from src.ui.common import" not in text
    assert "getattr(ui_common, \"load_historical_demo_orders\"" in text
    assert "historical_demo_orders.csv.gz" in text
