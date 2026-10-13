import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Investment Customer Churn API is running"


def test_predict():
    payload = {
        "age": 35,
        "tenure_months": 24,
        "monthly_income": 80000,
        "total_investment": 500000,
        "monthly_sip": 10000,
        "num_investments": 4,
        "portfolio_return_pct": 12,
        "last_investment_days": 60,
        "last_login_days": 20,
        "advisor_interactions": 3,
        "withdrawal_count_6m": 0,
        "support_tickets_6m": 1,
        "risk_profile": "Moderate"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert result["churn_prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1