
import mlflow.pyfunc
import mlflow
import os
from app.auth import acquire_access_token

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://ca-dev-mlflowserving-3wqoecnvn.orangecliff-532b4075.northeurope.azurecontainerapps.io")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

access_token = acquire_access_token()
os.environ["MLFLOW_TRACKING_TOKEN"] = access_token

_loaded_models = {}

def load_model(name: str, stage: str = "Production"):
    key = f"{name}:{stage}"
    if key not in _loaded_models:
        model_uri = f"models:/{name}/{stage}"
        _loaded_models[key] = mlflow.pyfunc.load_model(model_uri)
    return _loaded_models[key]
