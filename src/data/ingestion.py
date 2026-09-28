import io, uuid
from pathlib import Path
import pandas as pd
from src.config import ORDER_ID_CANDIDATES
from src.data.validation import validate_raw
from src.data.aggregation import aggregate_to_orders, first_existing
from src.data.versioning import sha256_bytes, save_raw, next_version, build_cumulative_snapshot, persist_version
from src.database.duckdb_manager import connect, now, log_event


def _append_quarantine(quarantine, rows, reason):
    if rows.empty:
        return quarantine
    q = rows.copy()
    q["quarantine_reason"] = reason
    if quarantine.empty:
        return q
    return pd.concat([quarantine, q], ignore_index=True, sort=False)


def _reserve_batch(file_hash: str, filename: str):
    """Atomically reserve a file hash so Streamlit reruns cannot ingest it twice."""
    batch_id = f"batch_{uuid.uuid4().hex[:10]}"
    with connect() as con:
        row = con.execute(
            """
            INSERT INTO data_batches
            (batch_id, filename, file_hash, uploaded_at, raw_rows, valid_rows,
             quarantined_rows, status, dataset_version, notes)
            VALUES (?, ?, ?, ?, 0, 0, 0, 'PROCESSING', NULL, NULL)
            ON CONFLICT(file_hash) DO NOTHING
            RETURNING batch_id
            """,
            [batch_id, filename, file_hash, now()],
        ).fetchone()
        if row:
            return batch_id, None
        existing = con.execute(
            "SELECT batch_id, dataset_version, status FROM data_batches WHERE file_hash = ?",
            [file_hash],
        ).fetchone()
    return None, existing


def _update_batch(batch_id, *, raw_rows=0, valid_rows=0, quarantined_rows=0,
                  status, dataset_version=None, notes=None):
    with connect() as con:
        con.execute(
            """
            UPDATE data_batches
            SET raw_rows=?, valid_rows=?, quarantined_rows=?, status=?, dataset_version=?, notes=?
            WHERE batch_id=?
            """,
            [raw_rows, valid_rows, quarantined_rows, status, dataset_version, notes, batch_id],
        )


def ingest_batch(file_bytes: bytes, filename: str):
    file_hash = sha256_bytes(file_bytes)
    batch_id, existing = _reserve_batch(file_hash, filename)
    if existing:
        existing_batch, existing_version, existing_status = existing
        return {
            "status": "duplicate",
            "message": f"Duplicate batch detected. This exact file was already received as {existing_batch} ({existing_status}). No data was added and retraining was not triggered.",
            "dataset_version": existing_version,
            "batch_id": existing_batch,
        }

    save_raw(file_bytes, filename, batch_id)
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    except Exception as e:
        _update_batch(batch_id, status="FAILED", notes=str(e))
        raise

    valid, quarantine, checks, fatal = validate_raw(df)
    if fatal:
        _update_batch(
            batch_id, raw_rows=len(df), valid_rows=0, quarantined_rows=len(df),
            status="FAILED", notes="Missing required columns"
        )
        with connect() as con:
            for a, b, c, d in checks:
                con.execute("INSERT INTO validation_results VALUES (?, ?, ?, ?, ?, ?)", [batch_id, a, b, c, d, now()])
        return {"status": "failed", "batch_id": batch_id, "checks": checks, "message": "Required schema validation failed"}

    order_col = first_existing(valid.columns, ORDER_ID_CANDIDATES)
    if order_col:
        inconsistent_ids = valid.groupby(order_col, dropna=False)["status"].nunique()
        inconsistent_ids = set(inconsistent_ids[inconsistent_ids > 1].index.astype(str))
        if inconsistent_ids:
            mask = valid[order_col].astype("string").isin(inconsistent_ids)
            quarantine = _append_quarantine(quarantine, valid.loc[mask], "inconsistent_order_status;")
            valid = valid.loc[~mask].copy()
        checks.append(("order_status_consistency", "PASS" if not inconsistent_ids else "WARN", len(inconsistent_ids), "Orders with multiple final statuses quarantined"))

        with connect() as con:
            prev = con.execute("SELECT processed_path FROM dataset_versions WHERE active = TRUE ORDER BY created_at DESC LIMIT 1").fetchone()
        existing_ids = set()
        if prev and Path(prev[0]).exists():
            existing_ids = set(pd.read_parquet(prev[0], columns=["order_id"])["order_id"].astype("string").dropna())
        overlap = set(valid[order_col].astype("string").dropna()) & existing_ids
        if overlap:
            mask = valid[order_col].astype("string").isin(overlap)
            quarantine = _append_quarantine(quarantine, valid.loc[mask], "existing_order_id;")
            valid = valid.loc[~mask].copy()
        checks.append(("existing_order_duplicates", "PASS" if not overlap else "WARN", len(overlap), "Previously ingested order IDs excluded and quarantined"))

    if valid.empty:
        _update_batch(
            batch_id, raw_rows=len(df), valid_rows=0, quarantined_rows=len(quarantine),
            status="NO_NEW_DATA", notes="No new valid records after validation/deduplication"
        )
        with connect() as con:
            for a, b, c, d in checks:
                con.execute("INSERT INTO validation_results VALUES (?, ?, ?, ?, ?, ?)", [batch_id, a, b, c, d, now()])
        return {
            "status": "duplicate", "batch_id": batch_id,
            "message": "No new valid records remained after validation and duplicate-order checks. Retraining was not triggered.",
            "dataset_version": None,
        }

    orders = aggregate_to_orders(valid)
    version = next_version()
    snapshot = build_cumulative_snapshot(orders, version)
    processed_path, quarantine_path, report_path = persist_version(batch_id, version, snapshot, quarantine, checks)

    _update_batch(
        batch_id, raw_rows=len(df), valid_rows=len(valid), quarantined_rows=len(quarantine),
        status="SUCCESS", dataset_version=version
    )
    with connect() as con:
        for a, b, c, d in checks:
            con.execute("INSERT INTO validation_results VALUES (?, ?, ?, ?, ?, ?)", [batch_id, a, b, c, d, now()])
    log_event(f"event_{uuid.uuid4().hex[:10]}", "DATA_INGESTION", "SUCCESS", f"Created dataset {version}", {"batch_id": batch_id})
    return {
        "status": "success", "batch_id": batch_id, "dataset_version": version,
        "raw_rows": len(df), "valid_rows": len(valid), "quarantined_rows": len(quarantine),
        "orders_in_batch": len(orders), "snapshot_orders": len(snapshot), "checks": checks,
        "processed_path": str(processed_path), "quarantine_path": str(quarantine_path), "report_path": str(report_path),
    }
