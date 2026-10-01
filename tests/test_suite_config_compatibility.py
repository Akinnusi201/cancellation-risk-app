from pathlib import Path


def test_suite_has_feature_schema_version_fallback():
    text = (Path(__file__).resolve().parents[1] / "src" / "models" / "suite.py").read_text()
    assert "except ImportError" in text
    assert 'FEATURE_SCHEMA_VERSION = "order_features_v1"' in text
    assert "from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, RANDOM_STATE" in text
