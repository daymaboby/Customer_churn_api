from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict():
    response = client.post("/predict", json={
        "recency": 10,
        "frequency": 5,
        "monetary": 2000,
        "ticket_count": 1,
        "refund_rate": 0.1,
        "web_activity": 20,
        "campaign_engagement": 2
    })

    assert response.status_code == 200
    assert "churn_probability" in response.json()