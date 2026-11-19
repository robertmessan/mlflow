import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["FLAGS_use_mkldnn"] = "0"

import mlflow.pyfunc
import mlflow
import pandas as pd
import shutil
from app.auth import acquire_access_token
from paddleocr import PaddleOCR
from pathlib import Path

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-ai-registry")

access_token = acquire_access_token()
os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# ─────────────────────────────────────────────────────────────
# 1. Téléchargement unique — PaddleX stocke dans ~/.paddlex
#    On laisse PaddleOCR télécharger dans son cache par défaut
#    puis on copie les modèles dans notre dossier local
# ─────────────────────────────────────────────────────────────
def to_file_uri(path: str) -> str:
    """Convertit un chemin Windows absolu en URI file:/// compatible MLflow."""
    return Path(os.path.abspath(path)).as_uri()

PADDLE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".paddlex", "official_models")
PADDLE_MODELS_DIR = os.path.abspath("data/paddle_models")

# Noms des modèles utilisés par PaddleOCR 3.5.0 en français
DET_MODEL_NAME = "PP-OCRv5_server_det"  #Détection
REC_MODEL_NAME = "latin_PP-OCRv5_mobile_rec"  #Reconnaissance
CLS_MODEL_NAME = "PP-LCNet_x1_0_textline_ori"  # orientation

DET_CACHE = os.path.join(PADDLE_CACHE_DIR, DET_MODEL_NAME)
REC_CACHE = os.path.join(PADDLE_CACHE_DIR, REC_MODEL_NAME)
CLS_CACHE = os.path.join(PADDLE_CACHE_DIR, CLS_MODEL_NAME)

DET_LOCAL = os.path.join(PADDLE_MODELS_DIR, DET_MODEL_NAME)
REC_LOCAL = os.path.join(PADDLE_MODELS_DIR, REC_MODEL_NAME)
CLS_LOCAL = os.path.join(PADDLE_MODELS_DIR, CLS_MODEL_NAME)

print("Initialisation PaddleOCR — téléchargement dans le cache par défaut...")
_init_ocr = PaddleOCR(
    use_textline_orientation=True,
    enable_mkldnn=False,
    lang="fr",
    # Pas de chemins custom ici — on laisse PaddleX gérer son cache
)
del _init_ocr
print("Téléchargement terminé.")

# Copie depuis le cache ~/.paddlex vers notre dossier local
print(f"Copie des modèles vers {PADDLE_MODELS_DIR}...")
for src, dst in [(DET_CACHE, DET_LOCAL), (REC_CACHE, REC_LOCAL), (CLS_CACHE, CLS_LOCAL)]:
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  ✅ {os.path.basename(src)} copié")
    else:
        print(f"  ⚠️  Modèle introuvable dans le cache : {src}")
        print(f"      Vérifiez le nom exact dans ~/.paddlex/official_models/")


class PaddleOCRWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        os.environ["FLAGS_use_mkldnn"] = "0"

        # ✅ Chargement depuis les artifacts MLflow via les nouveaux paramètres 3.5.0
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            enable_mkldnn=False,
            lang="fr",
            text_detection_model_dir=context.artifacts["det_model_dir"],
            text_recognition_model_dir=context.artifacts["rec_model_dir"],
            textline_orientation_model_dir=context.artifacts["cls_model_dir"],
        )

    def predict(self, context, model_input):
        result = self.ocr.predict(model_input)

        documents = []
        for res in result:
            lines = []
            confidences = []
            for item in res:
                try:
                    if isinstance(item, dict):
                        lines.append(item.get("rec_text", ""))
                        confidences.append(item.get("rec_score", 0.0))
                    elif item and len(item) > 1 and len(item[1]) > 1:
                        lines.append(item[1][0])
                        confidences.append(item[1][1])
                except Exception:
                    continue

            avg_confidence = (
                float(sum(confidences) / len(confidences)) if confidences else 0.0
            )
            documents.append({
                "text": "\n".join(lines),
                "avg_confidence": avg_confidence,
                "n_segments": len(confidences),
            })

        return pd.DataFrame(documents)


pip_requirements = [
    "mlflow",
    "paddleocr==3.5.0",
    "paddlepaddle",
    "opencv-python-headless",
]

# ─────────────────────────────────────────────
# 2. Enregistrement MLflow avec modèles figés
# ─────────────────────────────────────────────
with mlflow.start_run():

    mlflow.log_artifacts(DET_LOCAL, artifact_path=f"paddle_models/{DET_MODEL_NAME}")
    mlflow.log_artifacts(REC_LOCAL, artifact_path=f"paddle_models/{REC_MODEL_NAME}")
    mlflow.log_artifacts(CLS_LOCAL, artifact_path=f"paddle_models/{CLS_MODEL_NAME}")

    mlflow.pyfunc.log_model(
        artifact_path="paddleocr_model",
        python_model=PaddleOCRWrapper(),
        artifacts={
            "det_model_dir": to_file_uri(DET_LOCAL),
            "rec_model_dir": to_file_uri(REC_LOCAL),
            "cls_model_dir": to_file_uri(CLS_LOCAL),
        },
        registered_model_name="paddleocr_model",
        pip_requirements=pip_requirements,
    )

    print("✅ Modèle PaddleOCR enregistré sur MLflow avec modèles figés.")