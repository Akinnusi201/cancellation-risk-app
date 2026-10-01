from pathlib import Path


def test_interactive_order_lifecycle_is_present_in_schema_and_scoring_flow():
    db = Path("src/database/duckdb_manager.py").read_text(encoding="utf-8")
    predict = Path("src/models/predict.py").read_text(encoding="utf-8")
    page = Path("views/2_Score_Order.py").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS prototype_orders" in db
    assert "AWAITING_OPERATIONS_REVIEW" in predict
    assert "RELEASED_TO_FULFILLMENT" in predict
    assert "VERIFICATION_REQUIRED" in predict
    assert '"simulation_live", "manual"' in predict
    assert "Customer order status" in page
    assert "Operations review" in page
    assert "Place Next Simulated Customer Order" in page


def test_live_simulation_counts_for_business_but_not_drift_monitoring():
    metrics = Path("src/monitoring/metrics.py").read_text(encoding="utf-8")
    assert '"simulation_live",' in metrics.split("OPERATIONS_BUSINESS_MODES", 1)[1]
    production_block = metrics.split("PRODUCTION_SCORING_MODES", 1)[1].split("}", 1)[0]
    assert "simulation_live" not in production_block
