import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import mlflow
import mlflow.pyfunc
import subprocess
import shutil
from pathlib import Path
from app.auth import acquire_access_token
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ─────────────────────────────────────────────
# Configuration MLflow
# ─────────────────────────────────────────────
tracking_uri   = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-tracking")

access_token = acquire_access_token()
os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# ─────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────
MODEL_NAME        = "Qwen/Qwen3-1.7B"
OLLAMA_MODEL_TAG  = "medisync-slm:latest"
OUTPUT_DIR        = "./results"
BEST_CHECKPOINT   = "./results/checkpoint-2600"   # ← dernier checkpoint visible
SLM_MODEL_DIR     = "./slm_model_merged"          # modèle fusionné LoRA + base
GGUF_F16_PATH     = "./slm_model_merged/model_f16.gguf"
GGUF_PATH         = "./slm_model_merged/model.gguf"
MODELFILE_PATH    = "./slm_model_merged/Modelfile"
LLAMA_CPP_DIR     = "./llama.cpp"

HF_TOKEN          = os.getenv("HF_TOKEN")
HF_REPO_ID        = os.getenv("HF_REPO_ID", "Brave1/medisync-slm-gguf")

# ─────────────────────────────────────────────
# 1. Fusion LoRA + modèle de base → modèle complet
#    (si pas déjà fait)
# ─────────────────────────────────────────────
if not os.path.exists(os.path.join(SLM_MODEL_DIR, "config.json")):
    print(f"Fusion LoRA depuis {BEST_CHECKPOINT}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BEST_CHECKPOINT, trust_remote_code=True)

    # Charger les poids LoRA depuis le checkpoint
    model = PeftModel.from_pretrained(base_model, BEST_CHECKPOINT)
    model = model.merge_and_unload()

    os.makedirs(SLM_MODEL_DIR, exist_ok=True)
    model.save_pretrained(SLM_MODEL_DIR)
    tokenizer.save_pretrained(SLM_MODEL_DIR)
    print(f"✅ Modèle fusionné sauvegardé dans {SLM_MODEL_DIR}")
    del model, base_model
else:
    print(f"✅ Modèle fusionné déjà présent : {SLM_MODEL_DIR}")

# ─────────────────────────────────────────────
# 2. Conversion GGUF (si pas déjà fait)
# ─────────────────────────────────────────────
if not os.path.exists(GGUF_PATH):

    # Cloner llama.cpp si absent
    if not os.path.exists(LLAMA_CPP_DIR):
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            LLAMA_CPP_DIR
        ], check=True)

    # Dépendances Python de llama.cpp (sans écraser torch)
    subprocess.run([
        "pip", "install", "--quiet", "--no-deps",
        "-r", f"{LLAMA_CPP_DIR}/requirements.txt"
    ], check=True)
    subprocess.run([
        "pip", "install", "--quiet", "gguf", "sentencepiece"
    ], check=True)

    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")

    # Étape 1 : HF → GGUF f16
    print("Étape 1 : Conversion HF → GGUF f16...")
    subprocess.run([
        "python", convert_script,
        SLM_MODEL_DIR,
        "--outfile", GGUF_F16_PATH,
        "--outtype", "f16",
    ], check=True)
    print(f"✅ GGUF f16 : {GGUF_F16_PATH}")

    # Étape 2 : Quantisation f16 → Q4_K_M
    print("Étape 2 : Quantisation → Q4_K_M...")
    quantize_bin = os.path.join(LLAMA_CPP_DIR, "llama-quantize")
    if not os.path.exists(quantize_bin):
        subprocess.run([
            "cmake", "-B", f"{LLAMA_CPP_DIR}/build", LLAMA_CPP_DIR
        ], check=True)
        subprocess.run([
            "cmake", "--build", f"{LLAMA_CPP_DIR}/build",
            "--config", "Release", "--target", "llama-quantize",
            "-j", str(os.cpu_count())
        ], check=True)
        for root, dirs, files in os.walk(f"{LLAMA_CPP_DIR}/build"):
            for f in files:
                if f == "llama-quantize":
                    quantize_bin = os.path.join(root, f)
                    break

    subprocess.run([
        quantize_bin, GGUF_F16_PATH, GGUF_PATH, "Q4_K_M"
    ], check=True)

    if os.path.exists(GGUF_F16_PATH):
        os.remove(GGUF_F16_PATH)

    print(f"✅ GGUF Q4_K_M : {GGUF_PATH}")
else:
    print(f"✅ GGUF déjà présent : {GGUF_PATH}")

# ─────────────────────────────────────────────
# 3. Upload GGUF sur HuggingFace Hub
#    (stockage gratuit, pas de limite de taille)
# ─────────────────────────────────────────────
from huggingface_hub import HfApi, login, repo_exists

if HF_TOKEN:
    login(token=HF_TOKEN)

api = HfApi()

# Vérifier si le repo existe déjà
if not repo_exists(repo_id=HF_REPO_ID, repo_type="model"):
    print(f"Création du repo HuggingFace : {HF_REPO_ID}")
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        private=True,
        exist_ok=True       # ✅ pas d'erreur si existe déjà
    )
    print(f"✅ Repo créé : {HF_REPO_ID}")
else:
    print(f"✅ Repo existant : {HF_REPO_ID}")

print(f"Upload du GGUF sur HuggingFace Hub ({HF_REPO_ID})...")
api.upload_file(
    path_or_fileobj=GGUF_PATH,
    path_in_repo="model.gguf",
    repo_id=HF_REPO_ID,
    repo_type="model",
)
gguf_hf_url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/model.gguf"
print(f"✅ GGUF uploadé : {gguf_hf_url}")

# ─────────────────────────────────────────────
# 4. Création du Modelfile + enregistrement Ollama
# ─────────────────────────────────────────────
os.makedirs(os.path.dirname(MODELFILE_PATH), exist_ok=True)
modelfile_content = f"""FROM {os.path.abspath(GGUF_PATH)}

SYSTEM Tu es un médecin expert et bienveillant. Tu analyses les symptômes décrits par le patient et fournis un diagnostic clair, des conseils médicaux précis et une orientation vers la spécialité appropriée.

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
"""
with open(MODELFILE_PATH, "w") as f:
    f.write(modelfile_content)

if shutil.which("ollama") is None:
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    subprocess.Popen(["ollama", "serve"])
    import time; time.sleep(5)

subprocess.run(["ollama", "create", OLLAMA_MODEL_TAG, "-f", MODELFILE_PATH], check=True)
print("✅ Modèle Ollama créé.")

# ─────────────────────────────────────────────
# 5. Récupération des métriques depuis trainer_state.json
# ─────────────────────────────────────────────
def get_metrics(results_dir: str) -> dict:
    metrics = {}
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f == "trainer_state.json":
                with open(os.path.join(root, f)) as fp:
                    state = json.load(fp)
                log_history = state.get("log_history", [])
                eval_entries  = [e for e in log_history if "eval_loss" in e]
                train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
                if eval_entries:
                    best = min(eval_entries, key=lambda x: x["eval_loss"])
                    metrics["eval_loss"] = best["eval_loss"]
                    metrics["best_step"] = best.get("step", 0)
                if train_entries:
                    metrics["train_loss"] = train_entries[-1].get("loss", 0)
                metrics["num_epochs"] = state.get("num_train_epochs", 0)
                print(f"✅ Métriques lues depuis : {os.path.join(root, f)}")
                return metrics
    print("⚠️  Aucun trainer_state.json trouvé")
    return metrics

metrics = get_metrics(OUTPUT_DIR)
print(f"   Train loss : {metrics.get('train_loss', 'N/A')}")
print(f"   Eval loss  : {metrics.get('eval_loss', 'N/A')}")
print(f"   Best step  : {metrics.get('best_step', 'N/A')}")

# ─────────────────────────────────────────────
# 6. Wrapper MLflow
# ─────────────────────────────────────────────
class MedicalConsultationModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import subprocess, time, urllib.request, shutil

        #gguf_url   = context.artifacts["gguf_hf_url"]
        gguf_url   = os.getenv("GGUF_HF_URL")
        local_gguf = "/tmp/model.gguf"

        if not os.path.exists(local_gguf):
            print("Téléchargement du GGUF depuis HuggingFace...")
            urllib.request.urlretrieve(gguf_url, local_gguf)
            print(f"✅ GGUF téléchargé : {local_gguf}")

        local_modelfile = "/tmp/Modelfile"
        with open(local_modelfile, "w") as f:
            f.write(f"""FROM {local_gguf}

SYSTEM Tu es un médecin expert et bienveillant.

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|system|>"
""")

        if shutil.which("ollama") is None:
            subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True, check=True
            )
            subprocess.Popen(["ollama", "serve"])
            time.sleep(5)

        subprocess.run(
            ["ollama", "create", "medisync-slm:latest", "-f", local_modelfile],
            check=True
        )
        self.model_tag = "medisync-slm:latest"

    def predict(self, context, model_input):
        import requests

        if isinstance(model_input, dict):
            questions = [model_input.get("symptomes", "")]
        elif hasattr(model_input, "iterrows"):
            questions = model_input["symptomes"].tolist()
        else:
            questions = list(model_input)

        results = []
        for question in questions:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_tag,
                    "prompt": f"<|user|>{question}<|end|>\n<|assistant|>",
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=120
            )
            resp.raise_for_status()
            results.append({
                "reponse_medecin": resp.json().get("response", ""),
                "model": self.model_tag,
            })
        return results

# ─────────────────────────────────────────────
# 7. Log MLflow
# ─────────────────────────────────────────────
print("Enregistrement sur MLflow...")

with mlflow.start_run(run_name="medisync-slm-healthcaremagic"):

    mlflow.log_params({
        "base_model":    MODEL_NAME,
        "ollama_tag":    OLLAMA_MODEL_TAG,
        "dataset":       "lavita/ChatDoctor-HealthCareMagic-100k",
        "quantization":  "q4_k_m",
        "lora_r":        16,
        "lora_alpha":    32,
        "max_length":    512,
        "batch_size":    8,
        "best_checkpoint": BEST_CHECKPOINT,
    })

    if metrics:
        mlflow.log_metric("train_loss", metrics.get("train_loss", 0))
        mlflow.log_metric("eval_loss",  metrics.get("eval_loss",  0))
        mlflow.log_metric("best_step",  metrics.get("best_step",  0))

    mlflow.set_tag("gguf_hf_url",  gguf_hf_url)
    mlflow.set_tag("gguf_size_mb", f"{os.path.getsize(GGUF_PATH)/1024/1024:.1f} MB")
    mlflow.set_tag("hf_repo",      HF_REPO_ID)

    mlflow.log_artifact(MODELFILE_PATH, artifact_path="ollama")

    mlflow.pyfunc.log_model(
        artifact_path="medisync_slm",
        python_model=MedicalConsultationModel(),
        artifacts={
            #"gguf_hf_url": gguf_hf_url,
            "modelfile":   Path(os.path.abspath(MODELFILE_PATH)).as_uri(),
        },
        registered_model_name="medisync_slm",
        pip_requirements=[
            "transformers", "torch", "peft",
            "accelerate", "requests", "huggingface_hub",
        ],
    )

    print("✅ Run MLflow terminé.")
    print(f"   Train loss : {metrics.get('train_loss', 'N/A')}")
    print(f"   Eval loss  : {metrics.get('eval_loss',  'N/A')}")
    print(f"   GGUF URL   : {gguf_hf_url}")