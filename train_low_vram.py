from __future__ import annotations

import os
import sys
import logging
import json
from pathlib import Path

# Setup robust logging to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LowVramTrainer")

# Automatically resolve the project root and add it to python path
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "requirements-transformers.txt").exists():
    # If placed in src/models/, traverse up to find the root
    for parent in [PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent]:
        if (parent / "requirements-transformers.txt").exists():
            PROJECT_ROOT = parent
            break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt imports and verify deep learning dependencies
try:
    import torch
    import pandas as pd
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:
    logger.error(
        "Missing critical packages. Make sure you have installed requirements-transformers.txt."
    )
    sys.exit(1)

from src.utils.paths import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    METRICS_DIR,
    ensure_project_dirs,
)


def load_splits(
    label_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Loads CSV splits and maps text labels to integer targets."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    valid_path = PROCESSED_DATA_DIR / "valid.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not all(p.exists() for p in [train_path, valid_path, test_path]):
        raise FileNotFoundError(
            f"Processed splits are missing in {PROCESSED_DATA_DIR}. "
            "Please run 'python src/data/build_processed_dataset.py' first."
        )

    train_df = pd.read_csv(train_path).fillna("")
    valid_df = pd.read_csv(valid_path).fillna("")
    test_df = pd.read_csv(test_path).fillna("")

    # Map labels uniformly
    labels = sorted(train_df["specialty"].unique())
    label_to_id = {label: index for index, label in enumerate(labels)}

    for df in (train_df, valid_df, test_df):
        df["labels"] = df["specialty"].map(label_to_id)

    return train_df, valid_df, test_df, label_to_id


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
    }


def main():
    ensure_project_dirs()

    # Pre-allocation cleanup for 4GB card
    if torch.cuda.is_available():
        logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        logger.warning("CUDA is not available. Training will default to CPU.")

    model_name = "UBC-NLP/MARBERT"
    label_mode = "specialty"
    max_length = 128  # Safe length constraint for 4GB VRAM. (Can be increased to 160 if VRAM permits)
    epochs = 3
    output_dir = MODELS_DIR / f"transformer_MARBERT_{label_mode}_local"

    logger.info("Loading dataset splits...")
    try:
        train_df, valid_df, test_df, label_to_id = load_splits(label_mode)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    id_to_label = {index: label for label, index in label_to_id.items()}

    logger.info(f"Initializing tokenizer and model for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    # Convert pandas dataframes to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(
        train_df[["text", "labels"]], preserve_index=False
    ).map(tokenize, batched=True)
    valid_dataset = Dataset.from_pandas(
        valid_df[["text", "labels"]], preserve_index=False
    ).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(
        test_df[["text", "labels"]], preserve_index=False
    ).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # VRAM highly-optimized Training Arguments
    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # Low batch size to fit 4GB VRAM
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # Simulates an effective batch size of 8
        gradient_checkpointing=True,  # Crucial optimization: saves activation VRAM
        fp16=True,  # Leverages FP16 tensor cores on RTX 3050 Ti
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,  # Frequent terminal output log updates
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",  # Avoid overhead of third-party logging integrations
        dataloader_num_workers=0,  # Safest configuration for local machines
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Intercept Ctrl+C KeyboardInterrupt gracefully
    try:
        logger.info(
            "Starting training loop. Press Ctrl+C at any time to halt and save weights."
        )
        trainer.train()
    except KeyboardInterrupt:
        logger.warning(
            "\n[CTRL+C DETECTED] Interruption signal received. Halting training safely..."
        )

        # Save current in-memory weights (even if incomplete) and tokenizer configuration
        logger.info(f"Saving current model parameters to: {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        logger.info("Model state saved successfully. Exiting.")
        sys.exit(0)

    # Normal successful execution path
    logger.info("Training completed. Evaluating on test set...")
    metrics = trainer.evaluate(test_dataset)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics_path = METRICS_DIR / f"transformer_MARBERT_local_{label_mode}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(f"Model successfully saved to: {output_dir}")
    logger.info(f"Test metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
