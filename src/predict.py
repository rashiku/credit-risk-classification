import pandas as pd
import joblib

from src.feature_engineering import create_features

# -----------------------------
# Load trained pipeline
# -----------------------------
pipeline = joblib.load(
    "models/xgboost_pipeline.pkl"
)

# -----------------------------
# Prediction function
# -----------------------------
def predict_risk(input_data):

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Apply feature engineering
    df = create_features(df)

    # Predict probability
    probability = (
        pipeline
        .predict_proba(df)[0][1]
    )

    # Decision threshold
    decision = (
        "reject"
        if probability > 0.45
        else "approve"
    )

    return {
        "probability_bad": float(probability),
        "decision": decision
    }