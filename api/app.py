from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import logging

app = FastAPI(title="Fraud Detection API")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    model = joblib.load("models/isolation_forest.pkl")
except Exception as e:
    logging.error("Model loading failed")
    raise RuntimeError("Model not available") from e


class Transaction(BaseModel):
    amount: float = Field(..., gt=0)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    distance_from_home: float = Field(..., ge=0)


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        features = np.array([[
            transaction.amount,
            transaction.hour,
            transaction.day_of_week,
            transaction.distance_from_home
        ]])

        prediction = model.predict(features)[0]
        is_fraud = 1 if prediction == -1 else 0

        return {
            "fraud_probability": 0.85 if is_fraud else 0.15,
            "fraud_detected": bool(is_fraud),
            "reasoning": (
                "Transaction deviates from normal behavior patterns"
                if is_fraud else
                "Transaction aligns with normal behavior"
            )
        }

    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Prediction error")
