from fastapi import FastAPI, HTTPException
from .pydantic_models import TransactionInput
import joblib
import pandas as pd
import os

app = FastAPI(title="Credit Risk Scoring API")

# Path to the model saved by train.py
# Note: We ensure the model exists or handle the error
MODEL_PATH = "src/api/model.pkl"


@app.get("/")
def read_root():
    return {"status": "alive", "service": "Credit Risk Scoring"}


@app.post("/predict")
def predict_risk(data: TransactionInput):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=500,
            detail="Model file not found. Please train the model first.",
        )

    # Load model (lazy loading or load at startup)
    model = joblib.load(MODEL_PATH)

    # Prepare input dataframe
    input_data = data.dict()
    df = pd.DataFrame([input_data])

    # Make prediction
    try:
        prob = model.predict_proba(df)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # Business Logic: Threshold at 0.5
    risk_label = "High Risk" if prob > 0.5 else "Low Risk"

    return {"risk_probability": float(prob), "risk_label": risk_label}
