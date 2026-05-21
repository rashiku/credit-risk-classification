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

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_risk

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Credit Risk API"
)

# -----------------------------
# Input schema
# -----------------------------
class Applicant(BaseModel):

    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str
    Checking_account: str
    Credit_amount: float
    Duration: int
    Purpose: str

# -----------------------------
# Home route
# -----------------------------
@app.get("/")
def home():

    return {
        "message":
        "Credit Risk API Running"
    }

# -----------------------------
# Prediction route
# -----------------------------
@app.post("/predict")
def predict(applicant: Applicant):

    input_data = {
        "Age": applicant.Age,
        "Sex": applicant.Sex,
        "Job": applicant.Job,
        "Housing": applicant.Housing,
        "Saving accounts":
            applicant.Saving_accounts,
        "Checking account":
            applicant.Checking_account,
        "Credit amount":
            applicant.Credit_amount,
        "Duration":
            applicant.Duration,
        "Purpose":
            applicant.Purpose
    }

    result = predict_risk(input_data)

    return result