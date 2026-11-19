import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import shutil
import tempfile
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from app.auth import acquire_access_token

os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "3000"      # 5 min timeout
os.environ["CURL_CA_BUNDLE"] = ""                       # désactive vérif SSL
os.environ["REQUESTS_CA_BUNDLE"] = ""

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-ai-registry")

access_token = acquire_access_token()
os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

EMBEDDER_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDER_LOCAL_PATH = "data/embedder_model"  # sauvegarde locale des poids HuggingFace


class MedicalSimilarityModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        # Chargement du dataset
        with open(context.artifacts["dataset_path"], "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.df = pd.DataFrame(self.data)

        # Encodeur sexe — fitté sur .values pour éviter le warning feature names
        self.sex_encoder = OneHotEncoder(sparse_output=False)
        self.sex_encoder.fit(self.df[["sexe"]].values)

        # Chargement du SentenceTransformer depuis l'artifact MLflow (pas HuggingFace)
        self.embedder = SentenceTransformer(context.artifacts["embedder_path"])

        # Précalcul des embeddings du dataset
        self.dataset_embeddings = []
        for _, row in self.df.iterrows():
            age_norm = row["age"] / 100.0
            sex_encoded = self.sex_encoder.transform([[row["sexe"]]])[0]
            symptoms_emb = self.embedder.encode(row["symptomes"])
            combined_emb = np.concatenate([np.array([age_norm]), sex_encoded, symptoms_emb])
            self.dataset_embeddings.append(combined_emb)

        self.dataset_embeddings = np.array(self.dataset_embeddings)

    def predict(self, context, model_input):
        if isinstance(model_input, dict):
            df_input = pd.DataFrame([model_input])
        else:
            df_input = model_input

        results = []
        for _, row in df_input.iterrows():
            age_norm = row["age"] / 100.0
            sex_encoded = self.sex_encoder.transform([[row["sexe"]]])[0]
            symptoms_emb = self.embedder.encode(row["symptomes"])
            query_emb = np.concatenate([np.array([age_norm]), sex_encoded, symptoms_emb])

            similarities = cosine_similarity([query_emb], self.dataset_embeddings)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            best_match = self.df.iloc[best_idx]
            results.append({
                "maladie": best_match["maladie"],
                "specialite": best_match["specialite"],
                "similarity_score": float(best_score),
                "code_cim": best_match["code_cim"]
            })

        return pd.DataFrame(results)


class DummyContext:
    def __init__(self, artifacts):
        self.artifacts = artifacts


# ─────────────────────────────────────────────
# 1. Sauvegarde locale du SentenceTransformer
#    (téléchargement unique depuis HuggingFace)
# ─────────────────────────────────────────────
print(f"Téléchargement et sauvegarde du SentenceTransformer → {EMBEDDER_LOCAL_PATH}")
embedder = SentenceTransformer(EMBEDDER_MODEL_NAME)
embedder.save(EMBEDDER_LOCAL_PATH)

# ─────────────────────────────────────────────
# 2. Chemins des datasets
# ─────────────────────────────────────────────
train_data_path = "data/icd_medical_dataset_fr_complet.json"
test_data_path  = "data/symptomes_maladies_icd.csv"

with open(train_data_path, "r", encoding="utf-8") as f:
    train_data = json.load(f)

train_df = pd.DataFrame(train_data)
train_dataset_size   = len(train_df)
unique_specialites   = train_df["specialite"].nunique()
avg_symptoms_length  = train_df["symptomes"].str.len().mean()

# ─────────────────────────────────────────────
# 3. Évaluation sur le dataset de test
# ─────────────────────────────────────────────
model = MedicalSimilarityModel()
model.load_context(DummyContext({
    "dataset_path": train_data_path,
    "embedder_path": EMBEDDER_LOCAL_PATH      # ← poids locaux, pas HuggingFace
}))

test_df = pd.read_csv(test_data_path)
input_columns = ["age", "sexe", "symptomes"]
if not all(col in test_df.columns for col in input_columns):
    raise ValueError(f"Le fichier de test doit contenir les colonnes : {input_columns}")

predictions  = model.predict(None, test_df[input_columns])

test_results = test_df.copy()
test_results["pred_maladie"]          = predictions["maladie"]
test_results["pred_specialite"]       = predictions["specialite"]
test_results["pred_similarity_score"] = predictions["similarity_score"]

if "specialite" not in test_results.columns:
    raise ValueError("La colonne 'specialite' est requise dans les données de test.")

true_labels  = test_results["specialite"]
pred_labels  = test_results["pred_specialite"]

train_accuracy          = 1.0
train_loss              = 0.0
test_accuracy           = accuracy_score(true_labels, pred_labels)
test_recall             = recall_score(true_labels, pred_labels, average="macro")
test_loss               = float(np.mean(1.0 - test_results["pred_similarity_score"]))
classification_report_str = classification_report(true_labels, pred_labels, zero_division=0)

predictions_path           = "data/symptomes_maladies_icd_predictions.csv"
classification_report_path = "data/classification_report.txt"
test_results.to_csv(predictions_path, index=False)
with open(classification_report_path, "w", encoding="utf-8") as f:
    f.write(classification_report_str)

# ─────────────────────────────────────────────
# 4. Run MLflow — log datasets + modèle + embedder
# ─────────────────────────────────────────────
pip_requirements = [
    "mlflow",
    "pandas",
    "numpy",
    "scikit-learn",
    "sentence-transformers",
]

with mlflow.start_run():

    # Métriques
    mlflow.log_metric("train_accuracy",    train_accuracy)
    mlflow.log_metric("train_loss",        train_loss)
    mlflow.log_metric("test_accuracy",     float(test_accuracy))
    mlflow.log_metric("test_recall",       float(test_recall))
    mlflow.log_metric("test_loss",         test_loss)
    mlflow.log_metric("train_dataset_size", train_dataset_size)
    mlflow.log_metric("unique_specialites", unique_specialites)
    mlflow.log_metric("avg_symptoms_length", avg_symptoms_length)
    mlflow.log_metric("test_dataset_size", len(test_results))

    # ── Datasets rattachés au run ──────────────────
    # Train
    mlflow.log_artifact(train_data_path, artifact_path="datasets/train")
    # Test
    mlflow.log_artifact(test_data_path,  artifact_path="datasets/test")
    # Évaluation
    mlflow.log_artifact(predictions_path,           artifact_path="evaluation")
    mlflow.log_artifact(classification_report_path, artifact_path="evaluation")

    # ── SentenceTransformer sauvegardé dans le run ─
    mlflow.log_artifacts(EMBEDDER_LOCAL_PATH, artifact_path="embedder_model")

    # ── Modèle enregistré avec les deux datasets
    #    et l'embedder comme artifacts liés ──────────
    mlflow.pyfunc.log_model(
        artifact_path="sickness_similarity_model",
        python_model=MedicalSimilarityModel(),
        artifacts={
            # Dataset d'entraînement rattaché au modèle
            "dataset_path":  train_data_path,
            # Poids du SentenceTransformer — plus de téléchargement HuggingFace
            "embedder_path": EMBEDDER_LOCAL_PATH,
        },
        registered_model_name="sickness_similarity_model",
        pip_requirements=pip_requirements,
    )

    print(f"Run terminé — test_accuracy={test_accuracy:.4f} | test_recall={test_recall:.4f}")