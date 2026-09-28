import pandas as pd
from src.data.validation import validate_raw

def test_bad_quantity_is_quarantined():
    df=pd.DataFrame({
        "created_at":["2026-01-01"],"price":[10],"qty_ordered":[0],"grand_total":[10],
        "discount_amount":[0],"payment_method":["cod"],"category_name_1":["x"],"status":["complete"]})
    valid,q,checks,fatal=validate_raw(df)
    assert not fatal and len(valid)==0 and len(q)==1
