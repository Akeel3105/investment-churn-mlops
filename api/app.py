from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


from pathlib import Path
import joblib


# --------------------------------------------------
# Paths / MLflow configuration
# --------------------------------------------------


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"

# Load saved model
model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Investment Customer Churn Prediction API",
    version="1.0.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class CustomerInput(BaseModel):
    age: int
    tenure_months: int
    monthly_income: float
    total_investment: float
    monthly_sip: float
    num_investments: int
    portfolio_return_pct: float
    last_investment_days: int
    last_login_days: int
    advisor_interactions: int
    withdrawal_count_6m: int
    support_tickets_6m: int
    risk_profile: str


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Investment Customer Churn API is running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerInput):

    input_data = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(input_data)[0])

    # Probability
    probabilities = model.predict_proba(input_data)

    churn_probability = float(probabilities[0][1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(churn_probability, 4)
    }