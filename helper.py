import joblib
import numpy as np
import pandas as pd

class LoanApprovalModel:
    def __init__(self, model_path='loan_assesser/logistic_regression_model.joblib', scaler_path='loan_assesser/scaler.joblib'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = ["ratio_loan", "ratio_savings"]

    def predict(self, loan_amount, capital_share, savings):
        # Compute ratios as used in training
        ratio_loan = loan_amount / capital_share
        ratio_savings = savings / capital_share
        X = pd.DataFrame([[ratio_loan, ratio_savings]], columns=self.feature_names)
        X_scaled = self.scaler.transform(X)
        prob_approval = self.model.predict_proba(X_scaled)[:, 1][0]
        class_pred = self.model.predict(X_scaled)[0]
        return {
            "input": {
                "loan_amount": loan_amount,
                "capital_share": capital_share,
                "savings": savings,
                "ratio_loan": ratio_loan,
                "ratio_savings": ratio_savings
            },
            "prob_approval": prob_approval,
            "prediction": int(class_pred)
        }

# Usage example:
if __name__ == "__main__":
    helps = LoanApprovalModel()
    loan = float(input("Enter loan amount: "))
    capital = float(input("Enter capital share: "))
    savings = float(input("Enter savings: "))
    result = helps.predict(loan_amount=loan, capital_share=capital, savings=savings)
    print("Input:", result["input"])
    print(f"Predicted approval probability: {result['prob_approval']:.2f}")
    print("Prediction:", "Approved" if result["prediction"] == 1 else "Not Approved")