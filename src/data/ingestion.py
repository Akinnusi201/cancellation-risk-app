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


def ingest_batch(file_bytes: bytes, filename: str, progress_callback=None, run_mode="full", sample_orders=5000):
    def progress(step, message):
        if progress_callback:
            progress_callback(step, message)

    progress(2, "Calculating file fingerprint")
    # Include the processing mode in the deduplication key so the same source CSV
    # can be used once in Demo Sample mode and once in Full Dataset mode.
    mode_token = f"|mode={run_mode}|sample_orders={sample_orders if run_mode == 'demo' else 'all'}".encode()
    file_hash = sha256_bytes(file_bytes + mode_token)
    progress(5, "Checking for duplicate uploads")
    batch_id, existing = _reserve_batch(file_hash, filename)
    if existing:
        existing_batch, existing_version, existing_status = existing
        return {
            "status": "duplicate",
            "message": f"Duplicate batch detected. This exact file was already received as {existing_batch} ({existing_status}). No data was added and retraining was not triggered.",
            "dataset_version": existing_version,
            "batch_id": existing_batch,
        }

    progress(8, "Reserving new batch")
    save_raw(file_bytes, filename, batch_id)
    progress(12, "Saving raw batch")
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
        progress(15, "Reading CSV")

        # Demo mode samples COMPLETE ORDERS, not arbitrary rows, so item-level
        # baskets remain intact before order-level aggregation.
        original_rows = len(df)
        sampled_orders = None
        if run_mode == "demo":
            progress(18, "Sampling complete orders for fast demo")
            order_col_for_sample = first_existing(df.columns, ORDER_ID_CANDIDATES)
            if order_col_for_sample:
                ids = df[order_col_for_sample].dropna().astype("string").drop_duplicates()
                n = min(int(sample_orders), len(ids))
                chosen = ids.sample(n=n, random_state=42) if n < len(ids) else ids
                chosen_set = set(chosen.tolist())
                df = df[df[order_col_for_sample].astype("string").isin(chosen_set)].copy()
                sampled_orders = n
            else:
                df = df.sample(n=min(len(df), int(sample_orders)), random_state=42).copy()
                sampled_orders = None
            progress(20, "Demo sample ready")
    except Exception as e:
        _update_batch(batch_id, status="FAILED", notes=str(e))
        raise

    progress(22, "Validating schema and data quality")
    valid, quarantine, checks, fatal = validate_raw(df)
    progress(30, "Validation complete")
    if fatal:
        _update_batch(
            batch_id, raw_rows=len(df), valid_rows=0, quarantined_rows=len(df),
            status="FAILED", notes="Missing required columns"
        )
        with connect() as con:
            for a, b, c, d in checks:
                con.execute("INSERT INTO validation_results VALUES (?, ?, ?, ?, ?, ?)", [batch_id, a, b, c, d, now()])
        return {"status": "failed", "batch_id": batch_id, "checks": checks, "message": "Required schema validation failed"}

    progress(34, "Checking order consistency")
    order_col = first_existing(valid.columns, ORDER_ID_CANDIDATES)
    if order_col:
        inconsistent_ids = valid.groupby(order_col, dropna=False)["status"].nunique()
        inconsistent_ids = set(inconsistent_ids[inconsistent_ids > 1].index.astype(str))
        if inconsistent_ids:
            mask = valid[order_col].astype("string").isin(inconsistent_ids)
            quarantine = _append_quarantine(quarantine, valid.loc[mask], "inconsistent_order_status;")
            valid = valid.loc[~mask].copy()
        checks.append(("order_status_consistency", "PASS" if not inconsistent_ids else "WARN", len(inconsistent_ids), "Orders with multiple final statuses quarantined"))

        overlap = set()
        if run_mode == "full":
            with connect() as con:
                prev = con.execute("SELECT processed_path FROM dataset_versions WHERE active = TRUE ORDER BY created_at DESC LIMIT 1").fetchone()
            progress(40, "Checking for previously ingested orders")
            existing_ids = set()
            if prev and Path(prev[0]).exists():
                existing_ids = set(pd.read_parquet(prev[0], columns=["order_id"])["order_id"].astype("string").dropna())
            overlap = set(valid[order_col].astype("string").dropna()) & existing_ids
            if overlap:
                mask = valid[order_col].astype("string").isin(overlap)
                quarantine = _append_quarantine(quarantine, valid.loc[mask], "existing_order_id;")
                valid = valid.loc[~mask].copy()
            checks.append(("existing_order_duplicates", "PASS" if not overlap else "WARN", len(overlap), "Previously ingested order IDs excluded and quarantined"))
        else:
            progress(40, "Demo mode uses an isolated sample")
            checks.append(("existing_order_duplicates", "SKIP", 0, "Skipped in Demo Sample mode so the sample stays isolated and fast"))

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

    progress(48, "Aggregating item rows to order level")
    orders = aggregate_to_orders(valid)
    progress(55, "Order aggregation complete")
    progress(58, "Creating next dataset version")
    version = next_version()
    progress(62, "Building cumulative training snapshot")
    snapshot = build_cumulative_snapshot(orders, version, cumulative=(run_mode == "full"))
    progress(68, "Saving cleaned, quarantined, and validation artifacts")
    processed_path, quarantine_path, report_path = persist_version(batch_id, version, snapshot, quarantine, checks)
    progress(72, "Dataset version stored")

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
        "raw_rows": len(df), "source_rows": original_rows, "sampled_orders": sampled_orders, "run_mode": run_mode, "valid_rows": len(valid), "quarantined_rows": len(quarantine),
        "orders_in_batch": len(orders), "snapshot_orders": len(snapshot), "checks": checks,
        "processed_path": str(processed_path), "quarantine_path": str(quarantine_path), "report_path": str(report_path),
    }
