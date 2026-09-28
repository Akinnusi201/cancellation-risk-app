"""Generate a small synthetic CSV with the same fields expected by the prototype."""
from pathlib import Path
import numpy as np
import pandas as pd

rng=np.random.default_rng(42)
n=2500
start=pd.Timestamp("2024-01-01")
dates=start+pd.to_timedelta(np.arange(n)*3,unit="h")
payment=rng.choice(["cod","easypay_voucher","Payaxis","jazzwallet"],n,p=[.48,.18,.18,.16])
category=rng.choice(["Mobiles & Tablets","Men's Fashion","Women's Fashion","Appliances","Beauty & Grooming"],n)
price=np.round(rng.lognormal(7.0,.75,n),2)
qty=rng.integers(1,5,n)
discount=np.round(np.where(rng.random(n)<.35,rng.uniform(0,800,n),0),2)
grand=np.maximum(price*qty-discount,0)
customer=rng.integers(1,700,n)
base=-1.2 + .7*(payment=="cod") + .00004*grand + .3*(discount>500)
prob=1/(1+np.exp(-base))
status=np.where(rng.random(n)<prob,"canceled","complete")
df=pd.DataFrame({
    "increment_id":[f"ORD{i:07d}" for i in range(n)],
    "item_id":np.arange(1,n+1),
    "Customer ID":customer,
    "created_at":dates,
    "price":price,
    "qty_ordered":qty,
    "grand_total":np.round(grand,2),
    "discount_amount":discount,
    "payment_method":payment,
    "category_name_1":category,
    "status":status,
})
out=Path(__file__).resolve().parents[1]/"demo_batch.csv"
df.to_csv(out,index=False)
print(out)
