import pandas as pd
from src.data.schema import normalize_snapshot_schema


def test_old_integer_ids_and_new_string_ids_become_one_parquet_safe_type(tmp_path):
    old = pd.DataFrame({"order_id":[100252830], "customer_id":[1], "created_at":["2020-01-01"]})
    new = pd.DataFrame({"order_id":["100252831"], "customer_id":["2"], "created_at":["2020-01-02"]})
    combined = pd.concat([normalize_snapshot_schema(old), normalize_snapshot_schema(new)], ignore_index=True)
    combined = normalize_snapshot_schema(combined)
    assert str(combined["order_id"].dtype) == "string"
    assert combined["order_id"].tolist() == ["100252830", "100252831"]
