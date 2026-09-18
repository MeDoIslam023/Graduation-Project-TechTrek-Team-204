"""
Streamlit dashboard for the AI Credit Risk & Loan Default Intelligence Platform.

Run locally with:
    streamlit run app/app.py

Expects the trained artifacts produced by the notebook:
    models/credit_risk_pipeline.joblib
    models/metadata.json
"""

import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
LOG_PATH = MODEL_DIR / "prediction_log.csv"

st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")


@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(MODEL_DIR / "credit_risk_pipeline.joblib")
    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    return pipeline, metadata


def log_prediction(input_row: dict, probability: float, decision: str):
    """Append a prediction event to a simple CSV log (basic monitoring)."""
    log_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "probability": probability,
        "decision": decision,
        **input_row,
    }
    df_row = pd.DataFrame([log_row])
    if LOG_PATH.exists():
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, mode="w", header=True, index=False)


def main():
    st.title("🏦 AI Credit Risk & Loan Default Intelligence Platform")
    st.caption(
        "Decision-support tool for lending risk assessment. "
        "This is NOT a replacement for a human credit officer."
    )

    try:
        pipeline, metadata = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Run the training notebook first "
            "(`notebooks/01_credit_risk_full_pipeline.ipynb`) to generate "
            "`models/credit_risk_pipeline.joblib` and `models/metadata.json`."
        )
        st.stop()

    st.sidebar.header("Model Info")
    st.sidebar.write(f"**Model:** {metadata['model_name']}")
    st.sidebar.write(f"**Trained at:** {metadata['trained_at']}")
    st.sidebar.write(f"**Validation ROC-AUC:** {metadata['roc_auc_val']:.4f}")
    st.sidebar.write(f"**Decision threshold:** {metadata['threshold']:.2f}")

    tab1, tab2 = st.tabs(["🔍 Single Applicant", "📄 Batch Scoring (CSV)"])

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]

    # ------------------------------------------------------------------ #
    with tab1:
        st.subheader("Enter applicant details")
        st.info(
            "For brevity, this form exposes the most impactful fields. "
            "Batch scoring (second tab) accepts the full feature set via CSV."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            amt_income = st.number_input("Annual Income (AMT_INCOME_TOTAL)", min_value=0.0, value=150000.0, step=1000.0)
            amt_credit = st.number_input("Credit Amount (AMT_CREDIT)", min_value=0.0, value=500000.0, step=1000.0)
            amt_annuity = st.number_input("Loan Annuity (AMT_ANNUITY)", min_value=0.0, value=25000.0, step=500.0)
        with col2:
            age_years = st.number_input("Applicant Age (years)", min_value=18, max_value=100, value=35)
            employed_years = st.number_input("Years Employed", min_value=0.0, max_value=60.0, value=5.0)
            education = st.selectbox(
                "Education Level",
                ["Secondary / secondary special", "Higher education", "Incomplete higher",
                 "Lower secondary", "Academic degree"],
            )
        with col3:
            family_status = st.selectbox(
                "Family Status",
                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
            )
            income_type = st.selectbox(
                "Income Type",
                ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
            )
            n_children = st.number_input("Number of Children", min_value=0, max_value=15, value=0)

        if st.button("Assess Risk", type="primary"):
            # Build a single-row dataframe matching the training schema.
            # Missing columns are filled with NaN and imputed by the pipeline.
            row = {col: np.nan for col in numeric_features + categorical_features}
            row.update({
                "AMT_INCOME_TOTAL": amt_income,
                "AMT_CREDIT": amt_credit,
                "AMT_ANNUITY": amt_annuity,
                "age_years": age_years,
                "employed_years": employed_years,
                "credit_income_ratio": amt_credit / amt_income if amt_income else np.nan,
                "annuity_income_ratio": amt_annuity / amt_income if amt_income else np.nan,
                "CNT_CHILDREN": n_children,
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family_status,
                "NAME_INCOME_TYPE": income_type,
                 })
            input_df = pd.DataFrame([row])[numeric_features + categorical_features]

            proba = pipeline.predict_proba(input_df)[0, 1]
            threshold = metadata["threshold"]
            decision = "⚠️ HIGH RISK — Recommend decline / further review" if proba >= threshold else "✅ LOW RISK — Recommend approve"

            st.metric("Estimated Default Probability", f"{proba:.1%}")
            if proba >= threshold:
                st.error(decision)
            else:
                st.success(decision)

            log_prediction(
                {"amt_income": amt_income, "amt_credit": amt_credit, "amt_annuity": amt_annuity,
                 "age_years": age_years},
                float(proba), decision,
            )
            st.caption("This prediction was logged for monitoring purposes.")

    # ------------------------------------------------------------------ #
    with tab2:
        st.subheader("Upload a CSV of applicants")
        st.write(
            "The CSV should contain the same columns used at training time "
            "(numeric + categorical features). Missing columns are imputed automatically."
        )
        uploaded = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)

            for col in numeric_features + categorical_features:
                if col not in batch_df.columns:
                    batch_df[col] = np.nan
            batch_df = batch_df[numeric_features + categorical_features]

            probs = pipeline.predict_proba(batch_df)[:, 1]
            threshold = metadata["threshold"]
            batch_df["default_probability"] = probs
            batch_df["decision"] = np.where(
                probs >= threshold, "HIGH RISK", "LOW RISK"
            )

            st.dataframe(batch_df.sort_values("default_probability", ascending=False))
            st.download_button(
                "Download scored results",
                batch_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_applicants.csv",
                mime="text/csv",
            )

    st.divider()
    with st.expander("📊 Prediction Log (monitoring)"):
        if LOG_PATH.exists():
            st.dataframe(pd.read_csv(LOG_PATH).tail(50))
        else:
            st.write("No predictions logged yet.")


if __name__ == "__main__":
    main()
