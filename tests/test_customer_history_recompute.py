import pandas as pd

from src.data.aggregation import recompute_customer_history


def test_recompute_customer_history_spans_old_and_new_batches():
    combined = pd.DataFrame({
        "order_id": ["old-1", "new-1"],
        "customer_id": ["c1", "c1"],
        "created_at": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "is_canceled": [1, 0],
    })
    out = recompute_customer_history(combined)
    assert out.loc[out.order_id == "old-1", "customer_cancel_rate"].iat[0] == 0
    assert out.loc[out.order_id == "new-1", "customer_cancel_rate"].iat[0] == 1
