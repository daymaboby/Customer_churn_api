#student_name : Shalini gupta
#student_id : iitp_aiml_2506898

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)

# Input schema
class Customer(BaseModel):
    recency: float
    frequency: float
    monetary: float
    ticket_count: float
    refund_rate: float
    web_activity: float
    campaign_engagement: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: Customer):
    features = np.array([[
        data.recency,
        data.frequency,
        data.monetary,
        data.ticket_count,
        data.refund_rate,
        data.web_activity,
        data.campaign_engagement
    ]])

    prob = model.predict_proba(features)[0][1]
    pred = int(prob >= 0.4)

    if prob < 0.3:
        risk = "low"
        msg = "Customer is stable with low churn risk."
    elif prob < 0.7:
        risk = "medium"
        msg = "Moderate churn risk detected."
    else:
        risk = "high"
        msg = "High churn risk due to low engagement or high complaints."

    return {
        "churn_probability": float(prob),
        "predicted_class": pred,
        "risk_level": risk,
        "risk_explanation": msg
    }


@app.post("/batch_predict")
def batch_predict(data: list[Customer]):
    results = []

    for d in data:
        features = np.array([[
            d.recency,
            d.frequency,
            d.monetary,
            d.ticket_count,
            d.refund_rate,
            d.web_activity,
            d.campaign_engagement
        ]])

        prob = model.predict_proba(features)[0][1]
        pred = int(prob >= 0.4)

        results.append({
            "churn_probability": float(prob),
            "predicted_class": pred
        })

    return {"results": results}
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(model_path)
