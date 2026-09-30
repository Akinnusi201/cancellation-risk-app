import json
from datetime import datetime, timezone

import duckdb

from src.config import DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS data_batches (
    batch_id VARCHAR PRIMARY KEY,
    filename VARCHAR,
    file_hash VARCHAR UNIQUE,
    uploaded_at TIMESTAMP,
    raw_rows BIGINT,
    valid_rows BIGINT,
    quarantined_rows BIGINT,
    status VARCHAR,
    dataset_version VARCHAR,
    notes VARCHAR
);
CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_version VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    source_batch_id VARCHAR,
    row_count BIGINT,
    order_count BIGINT,
    processed_path VARCHAR,
    quarantine_path VARCHAR,
    report_path VARCHAR,
    active BOOLEAN DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS validation_results (
    batch_id VARCHAR,
    check_name VARCHAR,
    status VARCHAR,
    affected_rows BIGINT,
    details VARCHAR,
    checked_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id VARCHAR PRIMARY KEY,
    order_id VARCHAR,
    predicted_at TIMESTAMP,
    model_name VARCHAR,
    model_version VARCHAR,
    dataset_version VARCHAR,
    probability DOUBLE,
    threshold DOUBLE,
    recommendation VARCHAR,
    actual_outcome VARCHAR,
    expected_avoidable_cost DOUBLE,
    expected_false_positive_cost DOUBLE,
    net_expected_savings DOUBLE,
    avoidable_fulfillment_cost DOUBLE,
    intervention_effectiveness DOUBLE,
    intervention_cost DOUBLE,
    false_positive_friction_cost DOUBLE,
    latency_ms DOUBLE,
    scoring_mode VARCHAR,
    feature_json VARCHAR
);
CREATE TABLE IF NOT EXISTS manager_decisions (
    decision_id VARCHAR PRIMARY KEY,
    prediction_id VARCHAR,
    order_id VARCHAR,
    decided_at TIMESTAMP,
    manager_decision VARCHAR,
    recommendation VARCHAR,
    probability DOUBLE,
    threshold DOUBLE,
    actual_outcome VARCHAR,
    model_name VARCHAR,
    model_version VARCHAR,
    net_expected_savings DOUBLE,
    avoidable_fulfillment_cost DOUBLE,
    intervention_effectiveness DOUBLE,
    intervention_cost DOUBLE,
    false_positive_friction_cost DOUBLE
);
CREATE TABLE IF NOT EXISTS system_events (
    event_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    event_type VARCHAR,
    status VARCHAR,
    message VARCHAR,
    metadata_json VARCHAR
);
"""


def connect():
    con = duckdb.connect(str(DB_PATH))
    con.execute(SCHEMA_SQL)
    # Lightweight forward migrations for databases created by earlier prototype versions.
    migrations = [
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS expected_avoidable_cost DOUBLE",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS expected_false_positive_cost DOUBLE",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS net_expected_savings DOUBLE",
        "ALTER TABLE manager_decisions ADD COLUMN IF NOT EXISTS net_expected_savings DOUBLE",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS avoidable_fulfillment_cost DOUBLE",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS intervention_effectiveness DOUBLE",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS intervention_cost DOUBLE",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS false_positive_friction_cost DOUBLE",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS latency_ms DOUBLE",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS scoring_mode VARCHAR",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS feature_json VARCHAR",
        "ALTER TABLE manager_decisions ADD COLUMN IF NOT EXISTS avoidable_fulfillment_cost DOUBLE",
        "ALTER TABLE manager_decisions ADD COLUMN IF NOT EXISTS intervention_effectiveness DOUBLE",
        "ALTER TABLE manager_decisions ADD COLUMN IF NOT EXISTS intervention_cost DOUBLE",
        "ALTER TABLE manager_decisions ADD COLUMN IF NOT EXISTS false_positive_friction_cost DOUBLE",
    ]
    for sql in migrations:
        con.execute(sql)
    return con


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def log_event(event_id, event_type, status, message, metadata=None):
    with connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO system_events VALUES (?, ?, ?, ?, ?, ?)",
            [event_id, now(), event_type, status, message, json.dumps(metadata or {}, default=str)],
        )


def dataframe(query, params=None):
    with connect() as con:
        return con.execute(query, params or []).df()


def table_exists(name):
    with connect() as con:
        return con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [name]
        ).fetchone()[0] > 0
