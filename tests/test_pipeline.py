"""
Basic data / model sanity tests.

Run with: pytest tests/
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.mark.skipif(not (DATA_DIR / "application_train.csv").exists(),
                     reason="Dataset not downloaded")
def test_target_is_binary():
    df = pd.read_csv(DATA_DIR / "application_train.csv", usecols=["TARGET"])
    assert set(df["TARGET"].unique()).issubset({0, 1})


@pytest.mark.skipif(not (DATA_DIR / "application_train.csv").exists(),
                     reason="Dataset not downloaded")
def test_no_duplicate_applicants():
    df = pd.read_csv(DATA_DIR / "application_train.csv", usecols=["SK_ID_CURR"])
    assert df["SK_ID_CURR"].is_unique


@pytest.mark.skipif(not (MODEL_DIR / "credit_risk_pipeline.joblib").exists(),
                     reason="Model not trained yet")
def test_model_loads_and_predicts():
    pipeline = joblib.load(MODEL_DIR / "credit_risk_pipeline.joblib")
    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    # Build a single all-NaN row (pipeline should impute) to smoke-test predict
    row = {col: None for col in numeric_features + categorical_features}
    df = pd.DataFrame([row])

    proba = pipeline.predict_proba(df)[:, 1]
    assert 0.0 <= proba[0] <= 1.0


@pytest.mark.skipif(not (MODEL_DIR / "metadata.json").exists(),
                     reason="Metadata not generated yet")
def test_metadata_has_required_fields():
    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    for key in ["model_name", "trained_at", "threshold", "roc_auc_val"]:
        assert key in metadata
    assert 0.0 <= metadata["threshold"] <= 1.0
