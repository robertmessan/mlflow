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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN) 


# ─────────────────────────────────────────────
# Configuration MLflow
# ─────────────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-tracking")

access_token = acquire_access_token()
os.environ["MLFLOW_TRACKING_TOKEN"] = access_token
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# ─────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen3-1.7B"      # SLM performant CPU/GPU
OLLAMA_MODEL_TAG = "medisync-slm:latest" # nom du modèle dans Ollama
OUTPUT_DIR       = "./results"
SLM_MODEL_DIR    = "./slm_model"
GGUF_PATH        = "./slm_model/model.gguf"
MODELFILE_PATH   = "./slm_model/Modelfile"
MAX_LENGTH       = 512
BATCH_SIZE       = 8
NUM_EPOCHS       = 3

# ─────────────────────────────────────────────
# 1. Chargement du dataset HealthCareMagic-100k
# ─────────────────────────────────────────────
print("Chargement de HealthCareMagic-100k...")
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

def format_consultation(example):
    """
    Formate chaque entrée en format consultation :
    Patient décrit ses symptômes → Médecin répond
    """
    prompt = (
        f"<|system|>Tu es un médecin expert. Réponds de façon précise et bienveillante.<|end|>\n"
        f"<|user|>{example['input']}<|end|>\n"
        f"<|assistant|>{example['output']}<|end|>"
    )
    return {"text": prompt}

dataset = dataset.map(format_consultation, remove_columns=dataset.column_names)

# Split train / eval
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]

print(f"Train : {len(train_dataset)} exemples | Eval : {len(eval_dataset)} exemples")

# ─────────────────────────────────────────────
# 2. Tokenizer + Modèle + LoRA
# ─────────────────────────────────────────────
print(f"Chargement du modèle {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    trust_remote_code=True,
)

# LoRA — réduit drastiquement la mémoire nécessaire au finetuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # rang LoRA
    lora_alpha=32,           # scaling
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # couches ciblées
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# 3. Tokenisation du dataset
# ─────────────────────────────────────────────
def tokenize(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_tokenized = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
eval_tokenized  = eval_dataset.map(tokenize,  batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ─────────────────────────────────────────────
# 4. Arguments d'entraînement
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    per_device_eval_batch_size=8,
    prediction_loss_only=True,
    eval_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=True,
    report_to="none",                  # on gère MLflow manuellement
    push_to_hub=False,
)

# ─────────────────────────────────────────────
# 5. Métriques custom
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Perplexité = exp(cross-entropy loss)
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        torch.tensor(shift_logits).view(-1, shift_logits.shape[-1]),
        torch.tensor(shift_labels).view(-1),
    )
    perplexity = float(np.exp(loss.item()))
    return {"perplexity": perplexity}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,   # arrêt si pas d'amélioration après 3 évals
            early_stopping_threshold=0.008 # amélioration minimale requise
        )
    ]
)

# ─────────────────────────────────────────────
# 6. Entraînement
# ─────────────────────────────────────────────
print("Démarrage du finetuning...")
train_result = trainer.train()

# Fusion LoRA → modèle complet pour l'export
print("Fusion des poids LoRA dans le modèle base...")
model = model.merge_and_unload()

# Sauvegarde locale
os.makedirs(SLM_MODEL_DIR, exist_ok=True)
model.save_pretrained(SLM_MODEL_DIR)
tokenizer.save_pretrained(SLM_MODEL_DIR)
print(f"Modèle sauvegardé dans {SLM_MODEL_DIR}")

# ─────────────────────────────────────────────
# 7. Conversion GGUF pour Ollama (via llama.cpp)
# ─────────────────────────────────────────────
print("Conversion en GGUF pour Ollama...")

LLAMA_CPP_DIR  = "./llama.cpp"
GGUF_F16_PATH  = "./slm_model/model_f16.gguf"   # intermédiaire
GGUF_PATH      = "./slm_model/model.gguf"        # final quantisé

# Cloner llama.cpp si absent
if not os.path.exists(LLAMA_CPP_DIR):
    print("Clonage de llama.cpp...")
    subprocess.run([
        "git", "clone", "--depth", "1",
        "https://github.com/ggerganov/llama.cpp.git",
        LLAMA_CPP_DIR
    ], check=True)

# Installer les dépendances Python de llama.cpp
subprocess.run([
    "pip", "install", "--quiet",
    "--no-deps",                          # ✅ n'installe pas les dépendances transitives
    "-r", f"{LLAMA_CPP_DIR}/requirements.txt"
], check=True)

convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")

# ── Étape 1 : Conversion HF → GGUF f16 ──────────────────────
print("Étape 1 : Conversion HF → GGUF f16...")
subprocess.run([
    "python", convert_script,
    SLM_MODEL_DIR,
    "--outfile", GGUF_F16_PATH,
    "--outtype", "f16",          # ✅ type supporté par convert_hf_to_gguf.py
], check=True)
print(f"✅ GGUF f16 généré : {GGUF_F16_PATH}")

# ── Étape 2 : Quantisation f16 → q4_k_m ─────────────────────
print("Étape 2 : Quantisation f16 → Q4_K_M...")

# Compiler llama-quantize si absent
quantize_bin = os.path.join(LLAMA_CPP_DIR, "llama-quantize")
if not os.path.exists(quantize_bin):
    print("Compilation de llama-quantize...")
    subprocess.run([
        "cmake", "-B", f"{LLAMA_CPP_DIR}/build", LLAMA_CPP_DIR
    ], check=True)
    subprocess.run([
        "cmake", "--build", f"{LLAMA_CPP_DIR}/build",
        "--config", "Release",
        "--target", "llama-quantize",
        "-j", str(os.cpu_count())
    ], check=True)
    # Chercher le binaire compilé
    for root, dirs, files in os.walk(f"{LLAMA_CPP_DIR}/build"):
        for f in files:
            if f == "llama-quantize":
                quantize_bin = os.path.join(root, f)
                break

subprocess.run([
    quantize_bin,
    GGUF_F16_PATH,   # input  : GGUF f16
    GGUF_PATH,       # output : GGUF q4_k_m
    "Q4_K_M",        # type de quantisation
], check=True)

print(f"✅ GGUF Q4_K_M généré : {GGUF_PATH}")

# Nettoyage du fichier intermédiaire f16
if os.path.exists(GGUF_F16_PATH):
    os.remove(GGUF_F16_PATH)
    print("Fichier intermédiaire f16 supprimé.")

# ─────────────────────────────────────────────
# 8. Création du Modelfile Ollama
# ─────────────────────────────────────────────
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

with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
    f.write(modelfile_content)

if shutil.which("ollama") is None:
    print("Installation d'Ollama...")
    subprocess.run(
        "curl -fsSL https://ollama.com/install.sh | sh",
        shell=True, check=True
    )
    subprocess.Popen(["ollama", "serve"])
    import time
    time.sleep(5)
    print("Ollama démarré.")

# Enregistrement dans Ollama
print(f"Enregistrement du modèle dans Ollama sous le tag '{OLLAMA_MODEL_TAG}'...")
subprocess.run(["ollama", "create", OLLAMA_MODEL_TAG, "-f", MODELFILE_PATH], check=True)
print("Modèle Ollama créé avec succès.")




# ─────────────────────────────────────────────
# 9. Wrapper MLflow pour l'inférence via Ollama
# ─────────────────────────────────────────────
class MedicalConsultationModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import subprocess
        import json

        # Recharge le modèle dans Ollama depuis le GGUF stocké en artifact
        gguf_path     = context.artifacts["gguf_model"]
        modelfile_path = context.artifacts["modelfile"]

        # Recréer le modèle Ollama depuis les artifacts MLflow
        subprocess.run(
            ["ollama", "create", OLLAMA_MODEL_TAG, "-f", modelfile_path],
            check=True
        )
        self.model_tag = OLLAMA_MODEL_TAG

    def predict(self, context, model_input):
        import requests
        import pandas as pd

        if isinstance(model_input, dict):
            questions = [model_input.get("symptomes", "")]
        elif hasattr(model_input, "iterrows"):
            questions = model_input["symptomes"].tolist()
        else:
            questions = list(model_input)

        results = []
        for question in questions:
            payload = {
                "model": self.model_tag,
                "prompt": f"<|user|>{question}<|end|>\n<|assistant|>",
                "stream": False,
                "options": {"temperature": 0.3}
            }
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120
            )
            resp.raise_for_status()
            results.append({
                "reponse_medecin": resp.json().get("response", ""),
                "model": self.model_tag,
            })

        return results

# ─────────────────────────────────────────────
# 10. Log MLflow — tout en un seul run
# ─────────────────────────────────────────────
pip_requirements = [
    "transformers",
    "torch",
    "peft",
    "datasets",
    "accelerate",
    "requests",
]

print("Enregistrement sur MLflow...")

with mlflow.start_run(run_name="medisync-slm-healthcaremagic"):

    # ── Paramètres d'entraînement ────────────
    mlflow.log_params({
        "base_model":    MODEL_NAME,
        "ollama_tag":    OLLAMA_MODEL_TAG,
        "dataset":       "lavita/ChatDoctor-HealthCareMagic-100k",
        "train_size":    len(train_dataset),
        "eval_size":     len(eval_dataset),
        "num_epochs":    NUM_EPOCHS,
        "batch_size":    BATCH_SIZE,
        "max_length":    MAX_LENGTH,
        "learning_rate": training_args.learning_rate,
        "lora_r":        lora_config.r,
        "lora_alpha":    lora_config.lora_alpha,
        "quantization":  "q4_k_m",
    })

    # ── Métriques d'entraînement ─────────────
    mlflow.log_metric("train_loss",             train_result.training_loss)
    mlflow.log_metric("train_runtime_sec",      train_result.metrics.get("train_runtime", 0))
    mlflow.log_metric("train_samples_per_sec",  train_result.metrics.get("train_samples_per_second", 0))

    # Métriques d'évaluation finale
    eval_metrics = trainer.evaluate()
    mlflow.log_metric("eval_loss",        eval_metrics.get("eval_loss", 0))
    mlflow.log_metric("eval_perplexity",  eval_metrics.get("eval_perplexity", 0))

    # ── Fichiers nécessaires au finetuning ───
    mlflow.log_artifact(SLM_MODEL_DIR,    artifact_path="model_weights")

    # ── Fichiers nécessaires à l'inférence ───
    mlflow.log_artifact(GGUF_PATH,        artifact_path="gguf")
    mlflow.log_artifact(MODELFILE_PATH,   artifact_path="ollama")

    # ── Dataset de training loggé ────────────
    train_sample_path = "./data/train_sample.jsonl"
    os.makedirs("./data", exist_ok=True)
    with open(train_sample_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(train_dataset.select(range(min(100, len(train_dataset))))):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    mlflow.log_artifact(train_sample_path, artifact_path="dataset")

    # ── Modèle MLflow avec wrapper Ollama ────
    mlflow.pyfunc.log_model(
        artifact_path="medisync_slm",
        python_model=MedicalConsultationModel(),
        artifacts={
            "gguf_model": Path(GGUF_PATH).as_uri(),
            "modelfile":  Path(MODELFILE_PATH).as_uri(),
        },
        registered_model_name="medisync_slm",
        pip_requirements=pip_requirements,
    )

    print("✅ Run MLflow terminé.")
    print(f"   Train loss      : {train_result.training_loss:.4f}")
    print(f"   Eval loss       : {eval_metrics.get('eval_loss', 0):.4f}")
    print(f"   Eval perplexité : {eval_metrics.get('eval_perplexity', 0):.2f}")