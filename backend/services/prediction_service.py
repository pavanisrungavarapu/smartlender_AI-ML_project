from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import current_app

from train_model import MODEL_PATH, train


class PredictionService:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or MODEL_PATH

    def load_model_package(self) -> dict[str, Any]:
        if not self.model_path.exists():
            train()

        try:
            return joblib.load(self.model_path)
        except ModuleNotFoundError as error:
            if error.name != "xgboost":
                raise
            train()
            return joblib.load(self.model_path)

    def predict(self, applicant: dict[str, Any]) -> dict[str, Any]:
        model_package = self.load_model_package()
        input_df = pd.DataFrame([applicant])
        pipeline = model_package["pipeline"]
        prediction = int(pipeline.predict(input_df)[0])

        confidence = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_df)[0]
            confidence = round(float(probabilities[prediction]) * 100, 2)

        result = "Loan Approved" if prediction == 1 else "Loan Rejected"
        status = "approved" if prediction == 1 else "rejected"

        return {
            "result": result,
            "status": status,
            "confidence": confidence,
            "model_name": model_package["model_name"],
        }

    def build_applicant(self, form: Any) -> dict[str, Any]:
        def to_float(value: str | None, fallback: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        return {
            "Gender": form.get("gender", "Male"),
            "Married": form.get("married", "No"),
            "Dependents": form.get("dependents", "0"),
            "Education": form.get("education", "Graduate"),
            "Self_Employed": form.get("self_employed", "No"),
            "ApplicantIncome": to_float(form.get("applicant_income"), 0),
            "CoapplicantIncome": to_float(form.get("coapplicant_income"), 0),
            "LoanAmount": to_float(form.get("loan_amount"), 0),
            "Loan_Amount_Term": to_float(form.get("loan_amount_term"), 360),
            "Credit_History": to_float(form.get("credit_history"), 1),
            "Property_Area": form.get("property_area", "Urban"),
        }
