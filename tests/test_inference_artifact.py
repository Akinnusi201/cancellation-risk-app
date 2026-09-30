import pandas as pd

from src.config import FEATURES
from src.models.registry import active_metadata, load_active_model


def test_packaged_production_model_scores_without_training():
    model, meta = load_active_model()
    assert model is not None
    assert meta["dataset_version"] == "pakistan_seed_v1"
    sample = pd.read_csv("artifacts/demo_orders.csv.gz", low_memory=False).head(3)
    probs = model.predict_proba(sample[FEATURES])[:, 1]
    assert len(probs) == 3
    assert ((probs >= 0) & (probs <= 1)).all()
