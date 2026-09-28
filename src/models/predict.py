import uuid
import pandas as pd
from src.config import FEATURES
from src.database.duckdb_manager import connect, now
from src.models.registry import load_active_model


def explain_by_baseline(model, row: pd.DataFrame, reference: pd.DataFrame, top_k=3):
    base_prob=float(model.predict_proba(row[FEATURES])[:,1][0])
    impacts=[]
    for f in FEATURES:
        alt=row.copy()
        if pd.api.types.is_numeric_dtype(reference[f]): alt[f]=reference[f].median()
        else:
            m=reference[f].mode(); alt[f]=m.iat[0] if len(m) else "unknown"
        p=float(model.predict_proba(alt[FEATURES])[:,1][0])
        impacts.append((f,base_prob-p))
    return sorted([x for x in impacts if x[1]>0],key=lambda x:x[1],reverse=True)[:top_k]


def score_order(row: pd.DataFrame, reference: pd.DataFrame):
    model,meta=load_active_model()
    if model is None: raise RuntimeError("No active model. Upload/train a dataset first.")
    prob=float(model.predict_proba(row[FEATURES])[:,1][0]); threshold=float(meta["threshold"])
    recommendation="Hold for Verification" if prob>=threshold else "Approve for Fulfillment"
    reasons=explain_by_baseline(model,row,reference)
    prediction_id=f"pred_{uuid.uuid4().hex[:12]}"
    actual="Canceled" if int(row.iloc[0]["is_canceled"])==1 else "Completed"
    with connect() as con:
        con.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",[
            prediction_id,str(row.iloc[0]["order_id"]),now(),meta["model_name"],meta["model_version"],meta["dataset_version"],prob,threshold,recommendation,actual])
    return {"prediction_id":prediction_id,"probability":prob,"threshold":threshold,"recommendation":recommendation,"reasons":reasons,"actual_outcome":actual,"meta":meta}


def record_decision(scored, order_id, decision):
    decision_id=f"dec_{uuid.uuid4().hex[:12]}"
    with connect() as con:
        con.execute("INSERT INTO manager_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",[
            decision_id,scored["prediction_id"],str(order_id),now(),decision,scored["recommendation"],scored["probability"],scored["threshold"],scored["actual_outcome"],scored["meta"]["model_name"],scored["meta"]["model_version"]])
    return decision_id
