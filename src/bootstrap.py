import json
from pathlib import Path

import pandas as pd

from src.data.io import read_snapshot

from src.config import ROOT
from src.database.duckdb_manager import connect, now

SEED_VERSION = "pakistan_seed_v1"
SEED_PATH = ROOT / "data" / "seed" / "pakistan_orders_v1.csv.gz"
SEED_REPORT = ROOT / "data" / "seed" / "pakistan_seed_validation.json"
SEED_QUARANTINE = ROOT / "data" / "seed" / "pakistan_seed_quarantine.csv"


def initialize_runtime() -> None:
    """Register the packaged baseline dataset in a fresh runtime database.

    This is metadata bootstrapping only. It never retrains the production model.
    """
    if not SEED_PATH.exists():
        return

    with connect() as con:
        exists = con.execute(
            "SELECT COUNT(*) FROM dataset_versions WHERE dataset_version = ?", [SEED_VERSION]
        ).fetchone()[0]
        if exists:
            return

    df = read_snapshot(SEED_PATH, columns=["order_id"])
    row_count = len(df)
    order_count = df["order_id"].astype("string").nunique()

    if not SEED_REPORT.exists():
        SEED_REPORT.write_text(json.dumps({
            "version": SEED_VERSION,
            "source": "Packaged Pakistan e-commerce baseline",
            "note": "Prepared offline from complete/canceled orders using the project DataOps rules.",
        }, indent=2))
    if not SEED_QUARANTINE.exists():
        SEED_QUARANTINE.write_text("quarantine_reason\n")

    with connect() as con:
        con.execute("UPDATE dataset_versions SET active = FALSE")
        con.execute(
            """
            INSERT INTO dataset_versions
            (dataset_version, created_at, source_batch_id, row_count, order_count,
             processed_path, quarantine_path, report_path, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            """,
            [
                SEED_VERSION, now(), "seed_batch", row_count, order_count,
                str(SEED_PATH), str(SEED_QUARANTINE), str(SEED_REPORT), True,
            ],
        )
        con.execute(
            """
            INSERT INTO data_batches
            (batch_id, filename, file_hash, uploaded_at, raw_rows, valid_rows,
             quarantined_rows, status, dataset_version, notes)
            VALUES ('seed_batch', 'Pakistan Largest Ecommerce Dataset.csv', 'packaged-seed', ?, ?, ?, 0,
                    'SEED', ?, 'Packaged baseline dataset; raw 100+ MB CSV intentionally excluded from Git.')
            ON CONFLICT DO NOTHING
            """,
            [now(), row_count, row_count, SEED_VERSION],
        )
