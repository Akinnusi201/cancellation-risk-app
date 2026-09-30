from pathlib import Path


def test_monitoring_page_does_not_hard_import_new_business_helper():
    root = Path(__file__).resolve().parents[1]
    text = (root / "views/7_Model_Monitoring.py").read_text()

    assert "from src.monitoring.metrics import" not in text
    assert "from src.monitoring import metrics as monitoring_metrics" in text
    assert "getattr(\n    monitoring_metrics, \"operations_business_summary\"" in text
    assert "_fallback_operations_business_summary" in text
