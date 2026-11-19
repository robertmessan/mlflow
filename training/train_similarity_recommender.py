
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.auth import acquire_access_token
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import login


tracking_uri = "https://ca-dev-mlflowserving-3wqoecnvn.orangecliff-532b4075.northeurope.azurecontainerapps.io"
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-tracking")

#access_token = acquire_access_token()
#os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    
class CosineRecommender(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus = context.artifacts["corpus"]

    def predict(self, context, model_input):
        embeddings_query = self.model.encode(model_input)
        embeddings_corpus = self.model.encode(self.corpus)
        scores = cosine_similarity(embeddings_query, embeddings_corpus)
        top_idx = scores.argsort()[0][-5:][::-1]
        return [self.corpus[i] for i in top_idx]

corpus = ["Cardiologist", "Neurologist", "Dermatologist", "General Practitioner"]

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="cosine_recommender",
        python_model=CosineRecommender(),
        artifacts={"corpus": corpus},
        registered_model_name="cosine_recommender"
    )
