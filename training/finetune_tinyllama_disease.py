import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR.parent / ".env")
sys.path.insert(0, str(PROJECT_DIR))

import mlflow
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from app.auth import acquire_access_token


DATASET_ID = "sajjadhadi/disease-diagnosis-dataset"
MODEL_ID = "therealcyberlord/TinyLlama-1.1B-Medical"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_DIR = PROJECT_DIR / "data" / "disease"
OUTPUT_DIR = PROJECT_DIR / "results" / "tinyllama_disease"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
METRICS_HISTORY_PATH = OUTPUT_DIR / "metrics_history.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune l'adaptateur médical TinyLlama sur le dataset de diagnostic."
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--validation-limit", type=int, default=5000)
    parser.add_argument("--generation-eval-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def build_prompt(example: dict[str, Any]) -> dict[str, str]:
    text = " ".join(str(example["text"]).split())
    diagnosis = " ".join(str(example["diagnosis"]).split())
    match = re.search(
        r"Patient reported these symptoms:\s*(.*?)\s*(?:may indicate|based on these symptoms)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Impossible d'extraire les symptômes de l'exemple: {text[:160]}")
    symptoms = match.group(1).strip(" .")
    prompt = (
        f"<|user|>Patient reported these symptoms: {symptoms}. "
        "Based on these symptoms, what disease may the patient have?"
        "<|assistant|>"
    )
    return {"prompt": prompt, "diagnosis": diagnosis}


def save_jsonl(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(path), orient="records", lines=True, force_ascii=False)


def cache_key(values: dict[str, Any]) -> str:
    serialized = json.dumps(values, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def has_saved_dataset(path: Path) -> bool:
    return (path / "dataset_dict.json").is_file()


class MetricsCallback(TrainerCallback):
    def __init__(self, history_path: Path) -> None:
        self.history_path = history_path

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        record = {key: value for key, value in logs.items() if isinstance(value, (int, float))}
        if "eval_loss" in record:
            record["eval_perplexity"] = math.exp(min(float(record["eval_loss"]), 20))
        record["step"] = state.global_step
        record["epoch"] = state.epoch
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        if mlflow.active_run():
            for key, value in record.items():
                if key != "epoch" or value is not None:
                    mlflow.log_metric(key, float(value), step=state.global_step)


def tokenize_example(example: dict[str, str], tokenizer: Any, max_length: int) -> dict[str, list[int]]:
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(
        f" {example['diagnosis']}{tokenizer.eos_token}", add_special_tokens=False
    )["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    if not any(label != -100 for label in labels):
        raise ValueError("La séquence maximale ne laisse aucun token de diagnostic à entraîner.")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class CompletionCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        labels = [feature["labels"] for feature in features]
        model_features = [
            {key: value for key, value in feature.items() if key != "labels"}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        padded_labels = [
            label + [-100] * (max_length - len(label))
            if self.tokenizer.padding_side == "right"
            else [-100] * (max_length - len(label)) + label
            for label in labels
        ]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def normalize_answer(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def generation_accuracy(
    model: Any,
    tokenizer: Any,
    test_data: Dataset,
    limit: int,
) -> tuple[float, int]:
    examples = test_data.select(range(min(limit, len(test_data))))
    correct = 0
    evaluated = 0
    skipped = 0
    batch_size = 4
    model.eval()
    device = next(model.parameters()).device
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples.select(range(start, min(start + batch_size, len(examples))))
            valid_rows = [
                (prompt, diagnosis)
                for prompt, diagnosis in zip(batch["prompt"], batch["diagnosis"])
                if isinstance(prompt, str)
                and prompt.strip()
                and isinstance(diagnosis, str)
                and diagnosis.strip()
            ]
            skipped += len(batch) - len(valid_rows)
            if not valid_rows:
                continue
            prompts, references = zip(*valid_rows)
            encoded = tokenizer(
                list(prompts), padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            continuation = generated[:, encoded["input_ids"].shape[1] :]
            predictions = tokenizer.batch_decode(continuation, skip_special_tokens=True)
            correct += sum(
                normalize_answer(prediction) == normalize_answer(reference)
                for prediction, reference in zip(predictions, references)
            )
            evaluated += len(references)
    finally:
        tokenizer.padding_side = original_padding_side
    if skipped:
        print(f"Exemples ignorés pendant l'évaluation: {skipped} (prompt ou diagnostic invalide).")
    if not evaluated and examples:
        raise ValueError("Aucun exemple de test valide pour le calcul de l'exactitude.")
    return (correct / evaluated if evaluated else 0.0), evaluated


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_HISTORY_PATH.unlink(missing_ok=True)

    local_data_files = {
        "train": DATA_DIR / "source_train.parquet",
        "test": DATA_DIR / "source_test.parquet",
    }
    has_local_source = all(path.exists() for path in local_data_files.values())
    source_identity: dict[str, Any] = {"dataset": DATASET_ID, "seed": args.seed}
    if has_local_source:
        source_identity["files"] = {
            split: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for split, path in local_data_files.items()
        }
    prepared_cache = DATA_DIR / "cache" / f"prepared-{cache_key(source_identity)}"
    if args.rebuild_cache and prepared_cache.exists():
        shutil.rmtree(prepared_cache)

    if has_saved_dataset(prepared_cache):
        print(f"Réutilisation des données préparées: {prepared_cache}")
        prepared_data = load_from_disk(str(prepared_cache))
        train_data = prepared_data["train"]
        validation_data = prepared_data["validation"]
        test_data = prepared_data["test"]
    else:
        print(f"Chargement de {DATASET_ID}...")
        if has_local_source:
            source = load_dataset(
                "parquet", data_files={split: str(path) for split, path in local_data_files.items()}
            )
        else:
            source = load_dataset(DATASET_ID)
        if isinstance(source, DatasetDict) and "test" in source:
            source_train = source["train"]
            test_raw = source["test"]
        else:
            fallback_split = source["train"].train_test_split(test_size=0.1, seed=args.seed)
            source_train = fallback_split["train"]
            test_raw = fallback_split["test"]

        train_validation = source_train.train_test_split(test_size=0.05, seed=args.seed)
        train_data = train_validation["train"].map(
            build_prompt, remove_columns=train_validation["train"].column_names
        )
        validation_data = train_validation["test"].map(
            build_prompt, remove_columns=train_validation["test"].column_names
        )
        test_data = test_raw.map(build_prompt, remove_columns=test_raw.column_names)
        prepared_cache.parent.mkdir(parents=True, exist_ok=True)
        if prepared_cache.exists():
            shutil.rmtree(prepared_cache)
        DatasetDict(
            {"train": train_data, "validation": validation_data, "test": test_data}
        ).save_to_disk(str(prepared_cache))
        save_jsonl(train_data, DATA_DIR / "train.jsonl")
        save_jsonl(validation_data, DATA_DIR / "validation.jsonl")
        save_jsonl(test_data, DATA_DIR / "test.jsonl")
        print(
            f"Données préparées et mises en cache: {len(train_data)} train, "
            f"{len(validation_data)} validation, {len(test_data)} test."
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_cuda else torch.float32)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=model_dtype)
    model = PeftModel.from_pretrained(base_model, MODEL_ID, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    token_cache_identity = {
        "prepared_cache": prepared_cache.name,
        "tokenizer": BASE_MODEL_ID,
        "max_length": args.max_length,
        "validation_limit": args.validation_limit,
    }
    token_cache = DATA_DIR / "cache" / f"tokens-{cache_key(token_cache_identity)}"
    if args.rebuild_cache and token_cache.exists():
        shutil.rmtree(token_cache)
    if has_saved_dataset(token_cache):
        print(f"Réutilisation des tokens: {token_cache}")
        tokenized_data = load_from_disk(str(token_cache))
    else:
        eval_source = validation_data.select(
            range(min(args.validation_limit, len(validation_data)))
        )
        tokenized_data = DatasetDict(
            {
                "train": train_data.map(
                    lambda example: tokenize_example(example, tokenizer, args.max_length),
                    remove_columns=train_data.column_names,
                    desc="Tokenisation train",
                ),
                "validation": eval_source.map(
                    lambda example: tokenize_example(example, tokenizer, args.max_length),
                    remove_columns=eval_source.column_names,
                    desc="Tokenisation validation",
                ),
                "test": test_data.map(
                    lambda example: tokenize_example(example, tokenizer, args.max_length),
                    remove_columns=test_data.column_names,
                    desc="Tokenisation test",
                ),
            }
        )
        token_cache.parent.mkdir(parents=True, exist_ok=True)
        if token_cache.exists():
            shutil.rmtree(token_cache)
        tokenized_data.save_to_disk(str(token_cache))
        print(f"Tokens calculés et mis en cache: {token_cache}")
    train_tokens = tokenized_data["train"]
    validation_tokens = tokenized_data["validation"]
    test_tokens = tokenized_data["test"]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_DIR / "mlruns"))
    #if tracking_uri.startswith(("http://", "https://")) and not os.getenv(
    #    "MLFLOW_TRACKING_TOKEN"
    #):
    #    os.environ["MLFLOW_TRACKING_TOKEN"] = acquire_access_token()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "medisync-ai-registry"))
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=0.03,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        save_total_limit=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=validation_tokens,
        data_collator=CompletionCollator(tokenizer),
        callbacks=[MetricsCallback(METRICS_HISTORY_PATH)],
    )
    last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR))

    with mlflow.start_run(run_name="tinyllama-disease-diagnosis"):
        mlflow.log_params(
            {
                "base_model": BASE_MODEL_ID,
                "starting_adapter": MODEL_ID,
                "dataset": DATASET_ID,
                "train_size": len(train_data),
                "validation_size": len(validation_data),
                "test_size": len(test_data),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "eval_steps": args.eval_steps,
            }
        )
        if last_checkpoint:
            print(f"Reprise de l'entraînement depuis le checkpoint: {last_checkpoint}")
        else:
            print("Aucun checkpoint trouvé; démarrage du fine-tuning...")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(str(FINAL_MODEL_DIR))
        tokenizer.save_pretrained(FINAL_MODEL_DIR)

        validation_metrics = trainer.evaluate(metric_key_prefix="validation")
        test_metrics = trainer.evaluate(
            eval_dataset=test_tokens, metric_key_prefix="test"
        )
        test_accuracy, accuracy_sample_size = generation_accuracy(
            trainer.model, tokenizer, test_data, args.generation_eval_limit
        )
        test_metrics["test_perplexity"] = math.exp(
            min(float(test_metrics["test_loss"]), 20)
        )
        test_metrics["test_generation_exact_match"] = test_accuracy
        test_metrics["test_generation_sample_size"] = accuracy_sample_size

        summary = {
            "train": train_result.metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "generation_exact_match_sample_size": accuracy_sample_size,
            "model_output_dir": str(FINAL_MODEL_DIR),
        }
        with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
            json.dump(summary, metrics_file, ensure_ascii=False, indent=2, default=float)
        for key, value in {**train_result.metrics, **validation_metrics, **test_metrics}.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value), step=trainer.state.global_step)
        mlflow.log_artifact(str(METRICS_PATH), artifact_path="metrics")
        mlflow.log_artifact(str(METRICS_HISTORY_PATH), artifact_path="metrics")

    print(f"Fine-tuning terminé. Modèle: {FINAL_MODEL_DIR}")
    print(f"Métriques finales: {METRICS_PATH}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()