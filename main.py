from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

print("Loading model...")
model = joblib.load("logistic_regression_model.joblib")
print("Model loaded.")
scaler = joblib.load("scaler.joblib")
print("Scaler loaded.")

feature_names = ["ratio_loan", "ratio_savings"]

class LoanRequest(BaseModel):
    loan_amount: float
    capital_share: float
    savings: float

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_loan(req: LoanRequest):
    ratio_loan = req.loan_amount / req.capital_share
    ratio_savings = req.savings / req.capital_share
    X = pd.DataFrame([[ratio_loan, ratio_savings]], columns=feature_names)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0,1]
    pred = int(model.predict(X_scaled)[0])
    return {
        "prob_approval": prob,
        "prediction": pred,
        "ratios": {"ratio_loan": ratio_loan, "ratio_savings": ratio_savings},
    }