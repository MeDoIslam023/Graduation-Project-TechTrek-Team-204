from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

numeric_features = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "credit_income_ratio",
    "annuity_income_ratio",
    "employed_years",
    "age_years",
]

categorical_features = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "OCCUPATION_TYPE",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["credit_income_ratio"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    out["annuity_income_ratio"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    out["employed_years"] = -out["DAYS_EMPLOYED"].replace(365243, np.nan) / 365
    out["age_years"] = -out["DAYS_BIRTH"] / 365
    return out


def main() -> None:
    df = pd.read_csv(DATA_DIR / "application_train.csv")
    df = build_features(df)

    target = "TARGET"
    selected = numeric_features + categorical_features
    X = df[selected]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="auc",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = float(np.median(val_proba))

    joblib.dump(pipeline, MODEL_DIR / "credit_risk_pipeline.joblib")

    metadata = {
        "model_name": "XGBoost (retrained)",
        "trained_at": datetime.utcnow().isoformat(),
        "threshold": threshold,
        "roc_auc_val": float(np.round(np.mean((val_proba >= 0.5).astype(int)), 4)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved pipeline to {MODEL_DIR / 'credit_risk_pipeline.joblib'}")
    print(f"Saved metadata to {MODEL_DIR / 'metadata.json'}")
    print(f"Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
