from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.specialty_groups import map_to_broad_specialty, merge_rare_specialties
from src.utils.paths import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


DEFAULT_MODEL = "UBC-NLP/MARBERT"
MODEL_CHOICES = {
    "marbert": "UBC-NLP/MARBERT",
    "arabert": "aubmindlab/bert-base-arabertv2",
    "mbert": "bert-base-multilingual-cased",
}


def require_transformer_dependencies() -> None:
    try:
        import datasets  # noqa: F401
        import evaluate  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Transformer dependencies are optional. Install them with:\n"
            "  pip install -r requirements-transformers.txt\n"
            "Then rerun this script."
        ) from exc


def load_splits(label_mode: str, min_class_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv").fillna("")
    valid_df = pd.read_csv(PROCESSED_DATA_DIR / "valid.csv").fillna("")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv").fillna("")
    class_counts = train_df["specialty"].value_counts().to_dict()

    def make_label(label: object) -> str:
        if label_mode == "broad":
            return map_to_broad_specialty(label)
        if label_mode == "rare_merged":
            return merge_rare_specialties(label, class_counts, min_class_count)
        return str(label)

    for df in (train_df, valid_df, test_df):
        df["label_text"] = df["specialty"].map(make_label)

    labels = sorted(train_df["label_text"].unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    for df in (train_df, valid_df, test_df):
        df["labels"] = df["label_text"].map(label_to_id)
    return train_df, valid_df, test_df, label_to_id


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def compute_balanced_class_weights(labels: pd.Series, num_labels: int) -> list[float]:
    counts = labels.value_counts().to_dict()
    total = int(labels.shape[0])
    if total == 0:
        raise ValueError("Cannot compute class weights on an empty training split.")
    return [
        float(total / (num_labels * max(int(counts.get(label_id, 0)), 1)))
        for label_id in range(num_labels)
    ]


def limit_split(df: pd.DataFrame, rows: int | None) -> pd.DataFrame:
    if rows is None:
        return df
    if rows <= 0:
        raise ValueError("--limit-rows must be positive when provided.")
    return df.head(rows).reset_index(drop=True)


def train_transformer(
    model_name: str,
    label_mode: str = "specialty",
    min_class_count: int = 50,
    max_length: int = 160,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    class_weighted: bool = False,
    limit_rows: int | None = None,
) -> None:
    require_transformer_dependencies()
    from datasets import Dataset
    import torch
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    ensure_project_dirs()
    train_df, valid_df, test_df, label_to_id = load_splits(label_mode, min_class_count)
    train_df = limit_split(train_df, limit_rows)
    valid_df = limit_split(valid_df, limit_rows)
    test_df = limit_split(test_df, limit_rows)
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_dataset = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False).map(tokenize, batched=True)
    valid_dataset = Dataset.from_pandas(valid_df[["text", "labels"]], preserve_index=False).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df[["text", "labels"]], preserve_index=False).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    output_dir = MODELS_DIR / f"transformer_{model_slug(model_name)}_{label_mode}"
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_total_limit=2,
        seed=42,
        data_seed=42,
        report_to="none",
    )

    def compute_metrics(eval_pred: object) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
            "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        }

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_parameters = signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer_class = Trainer
    class_weights = compute_balanced_class_weights(train_df["labels"], len(label_to_id))

    if class_weighted:
        class WeightedLossTrainer(Trainer):
            def compute_loss(self, model: object, inputs: dict[str, object], return_outputs: bool = False, **kwargs: object) -> object:
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                model_config = getattr(model, "config", None)
                if model_config is None and hasattr(model, "module"):
                    model_config = model.module.config
                loss_function = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(class_weights, dtype=torch.float, device=logits.device)
                )
                loss = loss_function(logits.view(-1, model_config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer_class = WeightedLossTrainer

    trainer = trainer_class(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    prediction_output = trainer.predict(test_dataset)
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    true_labels = prediction_output.label_ids
    label_names = [id_to_label[index] for index in range(len(id_to_label))]
    report = classification_report(
        true_labels,
        predictions,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    confusion = confusion_matrix(true_labels, predictions, labels=list(range(len(label_names)))).tolist()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_metrics = {
        "modelName": model_name,
        "labelMode": label_mode,
        "classWeighted": class_weighted,
        "metrics": metrics,
        "classificationReport": report,
        "confusionMatrix": confusion,
        "labels": label_names,
    }
    training_manifest = {
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "modelName": model_name,
        "labelMode": label_mode,
        "minClassCount": min_class_count,
        "maxLength": max_length,
        "epochs": epochs,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "classWeighted": class_weighted,
        "classWeights": dict(zip(label_names, class_weights, strict=True)),
        "splitRows": {
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "labels": label_names,
        "randomSeed": 42,
        "selectionPolicy": "Best checkpoint selected by validation macro_f1; test split evaluated after training.",
    }

    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "training_manifest.json").write_text(json.dumps(training_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = METRICS_DIR / f"transformer_{model_slug(model_name)}_{label_mode}_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved transformer model to {output_dir}")
    print(f"Saved transformer metrics to {metrics_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Optional transformer comparison for Darija Health NLP.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or shortcut: marbert, arabert, mbert.")
    parser.add_argument("--label-mode", choices=["specialty", "rare_merged", "broad"], default="specialty")
    parser.add_argument("--min-class-count", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--class-weighted", action="store_true", help="Use balanced cross-entropy weights from the training split.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Tiny smoke-test mode; uses the first N rows of each split.")
    args = parser.parse_args()
    model_id = MODEL_CHOICES.get(args.model, args.model)
    train_transformer(
        model_name=model_id,
        label_mode=args.label_mode,
        min_class_count=args.min_class_count,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        class_weighted=args.class_weighted,
        limit_rows=args.limit_rows,
    )
