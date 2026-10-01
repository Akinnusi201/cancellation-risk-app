import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QUARANTINE_DIR = DATA_DIR / "quarantine"
REPORTS_DIR = DATA_DIR / "reports"
HOLDOUT_DIR = DATA_DIR / "holdout"
ARTIFACT_DIR = ROOT / "artifacts"
DB_PATH = Path(os.getenv("DUCKDB_PATH", str(ROOT / "cancellation.duckdb")))
MLFLOW_DB = Path(os.getenv("MLFLOW_DB_PATH", str(ROOT / "mlflow.db")))
# Optional remote MLflow tracking server can be supplied through Streamlit secrets/environment.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB.as_posix()}")
MLFLOW_ARTIFACT_ROOT = ARTIFACT_DIR / "mlflow"
REGISTERED_MODEL_NAME = "cancellation-risk-classifier"

RANDOM_STATE = 42
FEATURE_SCHEMA_VERSION = "order_features_v1"
TARGET = "is_canceled"
TIME_COL = "created_at"
ORDER_ID_CANDIDATES = ["increment_id", "order_id", "Order ID", "item_id"]
CUSTOMER_ID_CANDIDATES = ["Customer ID", "customer_id"]

LEAKAGE_COLUMNS = {
    "status", "BI Status", "bi_status", "refund_status", "refund", "is_canceled",
    "final_status", "completed_at", "cancelled_at", "canceled_at"
}

FEATURES = [
    "price", "qty_ordered", "grand_total", "discount_amount", "discount_ratio",
    "is_cod", "has_discount", "day_of_week", "hour", "customer_cancel_rate",
    "category_name_1", "payment_method"
]
NUMERIC_FEATURES = [
    "price", "qty_ordered", "grand_total", "discount_amount", "discount_ratio",
    "is_cod", "has_discount", "day_of_week", "hour", "customer_cancel_rate"
]
CATEGORICAL_FEATURES = ["category_name_1", "payment_method"]

for p in [RAW_DIR, PROCESSED_DIR, QUARANTINE_DIR, REPORTS_DIR, HOLDOUT_DIR, ARTIFACT_DIR, MLFLOW_ARTIFACT_ROOT]:
    p.mkdir(parents=True, exist_ok=True)
