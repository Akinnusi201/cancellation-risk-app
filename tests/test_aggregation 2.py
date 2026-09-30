import pandas as pd
from src.data.aggregation import aggregate_to_orders

def test_customer_history_uses_only_prior_orders():
    df=pd.DataFrame({
        "item_id":[1,2],"Customer ID":[10,10],"created_at":pd.to_datetime(["2020-01-01","2020-01-02"]),
        "price":[10,10],"qty_ordered":[1,1],"grand_total":[10,10],"discount_amount":[0,0],
        "category_name_1":["A","A"],"payment_method":["cod","cod"],"status":["canceled","complete"]})
    out=aggregate_to_orders(df)
    assert out.iloc[0].customer_cancel_rate==0
    assert out.iloc[1].customer_cancel_rate==1

def test_mixed_numeric_and_string_order_ids_are_normalized():
    df=pd.DataFrame({
        "increment_id":[100252830,"100252831"],"Customer ID":[10,"11"],
        "created_at":pd.to_datetime(["2020-01-01","2020-01-02"]),
        "price":[10,20],"qty_ordered":[1,1],"grand_total":[10,20],"discount_amount":[0,0],
        "category_name_1":["A","B"],"payment_method":["cod","cod"],"status":["complete","canceled"]})
    out=aggregate_to_orders(df)
    assert str(out["order_id"].dtype) == "string"
    assert out["order_id"].tolist() == ["100252830", "100252831"]
