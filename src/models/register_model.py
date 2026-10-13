import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

run_id = "3330dac336734f8db1a9abba8cbaee85"

model_name = "InvestmentCustomerChurnModel"

client = MlflowClient()

model_uri = f"runs:/{run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("\n========== MODEL REGISTERED ==========")
print(f"Model Name : {registered_model.name}")
print(f"Version    : {registered_model.version}")
print(f"Run ID     : {run_id}")