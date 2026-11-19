
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.auth import acquire_access_token
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier


FEATURE_COLUMNS = ["age", "sexe", "symptomes"]
TARGET_COLUMNS = ["specialite", "maladie"]
MODEL_NAME = "specialty_disease_classifier"

tracking_uri = "https://ca-dev-mlflowtracking-afcpry.agreeablecliff-300d9ef7.francecentral.azurecontainerapps.io"
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-ai-registry")

#access_token = acquire_access_token()
#os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.set_experiment(experiment_name)
mlflow_client = MlflowClient()
mlflow_client.set_experiment_tag(experiment.experiment_id, "release", "latest")

data = pd.read_csv(PROJECT_DIR / "data" / "symptomes_maladies_icd.csv")
test_data = pd.DataFrame(
    json.loads((PROJECT_DIR / "data" / "icd_medical_dataset_fr_complet.json").read_text(encoding="utf-8"))
)
missing_columns = set(FEATURE_COLUMNS + TARGET_COLUMNS) - set(test_data.columns)
if missing_columns:
    raise ValueError(f"Colonnes manquantes dans le jeu de test JSON: {sorted(missing_columns)}")

def prepare_features(frame):
    features = frame[FEATURE_COLUMNS].copy()
    features["age"] = pd.to_numeric(features["age"], errors="coerce")
    features["sexe"] = features["sexe"].fillna("inconnu").astype(str)
    features["symptomes"] = features["symptomes"].fillna("").astype(str)
    return features


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "age",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["age"],
            ),
            (
                "sexe",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["sexe"],
            ),
            (
                "symptomes",
                TfidfVectorizer(token_pattern=r"(?u)[^;\s]+"),
                "symptomes",
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", MultiOutputClassifier(LogisticRegression(max_iter=1000))),
        ]
    )


def evaluate_predictions(y_true, predictions, prefix):
    metrics = {}
    reports = {}
    for index, target in enumerate(TARGET_COLUMNS):
        truth = y_true[target]
        predicted = predictions[:, index]
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, average="macro", zero_division=0
        )
        target_metrics = {
            "accuracy": accuracy_score(truth, predicted),
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
        }
        metrics.update({f"{prefix}_{target}_{key}": value for key, value in target_metrics.items()})
        reports[target] = classification_report(
            truth, predicted, output_dict=True, zero_division=0
        )
    metrics[f"{prefix}_exact_match_accuracy"] = (predictions == y_true[TARGET_COLUMNS].to_numpy()).all(axis=1).mean()
    return metrics, reports


X = prepare_features(data)
y = data[TARGET_COLUMNS].fillna("inconnu").astype(str)
X_test = prepare_features(test_data)
y_test = test_data[TARGET_COLUMNS].fillna("inconnu").astype(str)

# The supplied JSON is heavily duplicated against training inputs; an internal
# holdout provides a separate estimate of generalization performance.
input_keys = pd.MultiIndex.from_frame(X.astype("string").fillna(""))
test_keys = pd.MultiIndex.from_frame(X_test.astype("string").fillna(""))
overlap_mask = test_keys.isin(input_keys)
overlap_rows = int(overlap_mask.sum())
overlap_rate = overlap_rows / len(X_test) if len(X_test) else 0.0

X_fit, X_validation, y_fit, y_validation = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y["specialite"],
)
validation_model = build_model().fit(X_fit, y_fit)
validation_predictions = validation_model.predict(X_validation)
validation_metrics, validation_reports = evaluate_predictions(
    y_validation, validation_predictions, "validation"
)

model = build_model().fit(X, y)
test_predictions = model.predict(X_test)
json_test_metrics, json_test_reports = evaluate_predictions(y_test, test_predictions, "json_test")
metrics = {
    **validation_metrics,
    **json_test_metrics,
    "json_test_feature_overlap_rows": overlap_rows,
    "json_test_feature_overlap_rate": overlap_rate,
}
reports = {
    "validation": validation_reports,
    "json_test": json_test_reports,
    "json_test_data_quality": {
        "rows": len(test_data),
        "training_rows": len(data),
        "feature_overlap_rows": overlap_rows,
        "feature_overlap_rate": overlap_rate,
        "warning": "JSON test examples overlap with training inputs; json_test scores are not an independent generalization estimate.",
    },
}

with mlflow.start_run(run_name=f"{MODEL_NAME}-latest"):
    mlflow.set_tags(
        {
            "release": "latest",
            "test_dataset": "icd_medical_dataset_fr_complet.json",
            "json_test_feature_overlap_rate": str(overlap_rate),
        }
    )
    mlflow.log_params(
        {
            "training_dataset": "symptomes_maladies_icd.csv",
            "test_dataset": "icd_medical_dataset_fr_complet.json",
            "training_rows": len(data),
            "validation_rows": len(X_validation),
            "json_test_rows": len(test_data),
            "output_targets": ",".join(TARGET_COLUMNS),
            "random_state": 42,
        }
    )
    mlflow.log_metrics(metrics)
    mlflow.log_dict(reports, "classification_reports.json")
    model_info = mlflow.sklearn.log_model(
        model,
        name=MODEL_NAME,
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["numpy.dtype"],
    )
    registered_version = model_info.registered_model_version
    if registered_version is not None:
        mlflow_client.set_model_version_tag(MODEL_NAME, registered_version, "release", "latest")
