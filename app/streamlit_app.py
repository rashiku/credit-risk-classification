import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import pandas as pd

from src.predict import predict_risk

# -----------------------------
# App title
# -----------------------------
st.title("Credit Risk Classification System")

st.write(
    """
    Predict whether a loan applicant
    is likely to default.
    """
)

# -----------------------------
# User Inputs
# -----------------------------
age = st.slider(
    "Age",
    18,
    75,
    35
)

credit_amount = st.number_input(
    "Credit Amount",
    min_value=0,
    value=5000
)

duration = st.slider(
    "Loan Duration (months)",
    1,
    72,
    24
)

job = st.selectbox(
    "Job Level",
    [0, 1, 2, 3]
)

housing = st.selectbox(
    "Housing",
    ["own", "rent", "free"]
)

saving_accounts = st.selectbox(
    "Saving Accounts",
    [
        "little",
        "moderate",
        "quite rich",
        "rich",
        "unknown"
    ]
)

checking_account = st.selectbox(
    "Checking Account",
    [
        "little",
        "moderate",
        "rich",
        "unknown"
    ]
)

purpose = st.selectbox(
    "Loan Purpose",
    [
        "car",
        "furniture/equipment",
        "radio/TV",
        "education",
        "business",
        "vacation/others"
    ]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict Risk"):

    input_data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }

    result = predict_risk(input_data)

    probability = result["probability_bad"]

    decision = result["decision"]

    st.subheader("Prediction Result")

    st.write(
        f"Probability of Default: {probability:.2%}"
    )

    if decision == "reject":
        st.error("High Risk Applicant")
    else:
        st.success("Low Risk Applicant")