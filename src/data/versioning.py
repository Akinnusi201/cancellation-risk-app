import hashlib, json, uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
from src.config import RAW_DIR, PROCESSED_DIR, QUARANTINE_DIR, REPORTS_DIR
from src.database.duckdb_manager import connect, now
from src.data.schema import normalize_snapshot_schema


def sha256_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()


def next_version():
    with connect() as con:
        n = con.execute("SELECT COUNT(*) FROM dataset_versions").fetchone()[0] + 1
    return f"v{n:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def duplicate_hash_exists(file_hash):
    with connect() as con:
        row = con.execute("SELECT batch_id, dataset_version FROM data_batches WHERE file_hash = ?", [file_hash]).fetchone()
    return row


def save_raw(file_bytes, filename, batch_id):
    path = RAW_DIR / f"{batch_id}_{Path(filename).name}"
    path.write_bytes(file_bytes)
    return path


def build_cumulative_snapshot(new_orders, version):
    new_orders = normalize_snapshot_schema(new_orders)
    with connect() as con:
        prev = con.execute("SELECT processed_path FROM dataset_versions WHERE active = TRUE ORDER BY created_at DESC LIMIT 1").fetchone()
    if prev and Path(prev[0]).exists():
        old = normalize_snapshot_schema(pd.read_parquet(prev[0]))
        combined = pd.concat([old, new_orders], ignore_index=True)
        combined = normalize_snapshot_schema(combined)
        combined = combined.sort_values(["created_at", "order_id"]).drop_duplicates(subset=["order_id"], keep="last")
    else:
        combined = new_orders.copy()
    return normalize_snapshot_schema(combined.reset_index(drop=True))


def persist_version(batch_id, version, snapshot, quarantine, checks):
    snapshot = normalize_snapshot_schema(snapshot)
    processed_path = PROCESSED_DIR / f"orders_clean_{version}.parquet"
    quarantine_path = QUARANTINE_DIR / f"quarantine_{version}.csv"
    report_path = REPORTS_DIR / f"validation_{version}.json"
    snapshot.to_parquet(processed_path, index=False)
    quarantine.to_csv(quarantine_path, index=False)
    report = {"version": version, "batch_id": batch_id, "checks": [dict(check_name=a,status=b,affected_rows=c,details=d) for a,b,c,d in checks]}
    report_path.write_text(json.dumps(report, indent=2, default=str))
    with connect() as con:
        con.execute("UPDATE dataset_versions SET active = FALSE")
        con.execute("INSERT INTO dataset_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)", [
            version, now(), batch_id, len(snapshot), snapshot["order_id"].nunique(), str(processed_path), str(quarantine_path), str(report_path)
        ])
    return processed_path, quarantine_path, report_path
