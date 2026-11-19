
#import sys
#import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#os.environ["FLAGS_use_mkldnn"] = "0"
#import mlflow.pyfunc
#import mlflow
#import pandas as pd
#from app.auth import acquire_access_token
#from paddleocr import PaddleOCR
#
#tracking_uri = "https://ca-dev-mlflowserving-3wqoecnvn.orangecliff-532b4075.northeurope.azurecontainerapps.io"
#experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-ai-registry")
#
#access_token = acquire_access_token()
#os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
#mlflow.set_tracking_uri(tracking_uri)
#mlflow.set_experiment(experiment_name)
#
#class PaddleOCRWrapper(mlflow.pyfunc.PythonModel):
#
#    def load_context(self, context):
#        self.ocr = PaddleOCR(use_textline_orientation=True, enable_mkldnn=False, lang='fr')
#
#    def predict(self, context, model_input):
#        result = self.ocr.predict(model_input)
#
#        # PaddleOCR retourne normalement une liste de pages / lignes.
#        pages = result
#        if result:
#            is_page_list = isinstance(result[0], list) and result[0] and isinstance(result[0][0], list)
#            if not is_page_list:
#                pages = [result]
#
#        documents = []
#        for page in pages:
#            text = "\n".join([line[1][0] for line in page])
#            confidences = [line[1][1] for line in page if line and len(line) > 1 and len(line[1]) > 1]
#            avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
#            documents.append({
#                "text": text,
#                "avg_confidence": avg_confidence,
#                "n_segments": len(confidences),
#            })
#
#        if mlflow.active_run() and documents:
#            overall_avg = float(sum(d["avg_confidence"] for d in documents) / len(documents))
#            mlflow.log_metric("document_avg_confidence", overall_avg)
#            mlflow.log_metric("document_count", len(documents))
#
#        return pd.DataFrame(documents)
#
#pip_requirements = [
#    "mlflow",
#    "paddleocr==3.5.0",
#    "paddlepaddle",
#    "opencv-python-headless",
#]
#
#with mlflow.start_run():
#    mlflow.pyfunc.log_model(
#        artifact_path="paddleocr_model",
#        python_model=PaddleOCRWrapper(),
#        registered_model_name="paddleocr_model",
#        pip_requirements=pip_requirements,
#    )


import sys
import os
import mlflow
import mlflow.pyfunc
import pandas as pd

from paddleocr import PaddleOCR
from mlflow.models.signature import infer_signature

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["FLAGS_use_mkldnn"] = "0"

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-ai-registry")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)


# ─────────────────────────────────────────────
# MODEL WRAPPER
# ─────────────────────────────────────────────
class PaddleOCRWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):

        base = context.artifacts["base_path"]
        base = os.path.abspath(base)

        det_path = os.path.join(base,"PP-OCRv5_server_det")
        rec_path = os.path.join(base,"latin_PP-OCRv5_mobile_rec")
        cls_path = os.path.join(base,"PP-LCNet_x1_0_textline_ori")

        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            enable_mkldnn=False,
            lang="fr",

            det_model_dir=det_path,
            rec_model_dir=rec_path,
            cls_model_dir=cls_path
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


# ─────────────────────────────────────────────
# INPUT / OUTPUT EXAMPLE
# ─────────────────────────────────────────────
input_example = pd.DataFrame({
    "image_path": ["dummy_path.png"]
})


output_example = pd.DataFrame({
    "image_path": ["dummy_path.png"],
    "text": ["exemple"],
    "avg_confidence": [0.99],
    "n_segments": [5]
})

signature = infer_signature(input_example, output_example)


# ─────────────────────────────────────────────
# MLflow LOG
# ─────────────────────────────────────────────
with mlflow.start_run():

    mlflow.pyfunc.log_model(
        artifact_path="paddleocr_model",
        python_model=PaddleOCRWrapper(),

        artifacts={
            "base_path": os.path.join("data", "paddle_models")
        },

        registered_model_name="paddleocr_model",

        pip_requirements=[
            "mlflow",
            "paddleocr==3.5.0",
            "paddlepaddle",
            "opencv-python-headless",
            "pandas",
            "numpy"
        ],

        signature=signature,
        #input_example=input_example
    )

    print("✔ PaddleOCR model logged correctly")