from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.specialty_groups import map_to_broad_specialty, merge_rare_specialties
from src.utils.paths import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, ensure_project_dirs


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/data/build_processed_dataset.py first.")
    return pd.read_csv(path).fillna("")


def build_models() -> dict[str, Pipeline]:
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95)
    return {
        "tfidf_logistic_regression": Pipeline(
            [
                ("tfidf", tfidf),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95)),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        ),
        "char_wb_tfidf_linear_svm": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        ),
    }


def score_model(model: Pipeline, x: pd.Series, y: pd.Series) -> dict[str, float]:
    preds = model.predict(x)
    return {
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(y, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(y, preds, average="macro", zero_division=0),
    }


def prepare_training_labels(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_mode: str,
    min_class_count: int,
) -> tuple[pd.Series, pd.Series]:
    if label_mode == "specialty":
        return train_df["specialty"], valid_df["specialty"]
    if label_mode == "broad":
        return train_df["specialty"].map(map_to_broad_specialty), valid_df["specialty"].map(map_to_broad_specialty)
    if label_mode == "rare_merged":
        class_counts = train_df["specialty"].value_counts().to_dict()
        return (
            train_df["specialty"].map(lambda label: merge_rare_specialties(label, class_counts, min_class_count)),
            valid_df["specialty"].map(lambda label: merge_rare_specialties(label, class_counts, min_class_count)),
        )
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def train(label_mode: str = "specialty", min_class_count: int = 50) -> Pipeline:
    ensure_project_dirs()
    train_df = load_split("train")
    valid_df = load_split("valid")
    y_train, y_valid = prepare_training_labels(train_df, valid_df, label_mode, min_class_count)
    models = build_models()
    results: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model: Pipeline | None = None
    best_score = -1.0

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(train_df["text"], y_train)
        metrics = score_model(model, valid_df["text"], y_valid)
        results[name] = metrics
        print(name, metrics)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_name = name
            best_model = model

    assert best_model is not None
    joblib.dump(best_model, MODELS_DIR / "specialty_classifier.joblib")
    labels = sorted(pd.Series(y_train).unique().tolist())
    (MODELS_DIR / "label_mapping.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "best_model": best_name,
                "label_mode": label_mode,
                "min_class_count": min_class_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (METRICS_DIR / "validation_model_comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved best model ({best_name}) to {MODELS_DIR / 'specialty_classifier.joblib'}")
    return best_model


if __name__ == "__main__":
    parser = ArgumentParser(description="Train V1/V2 TF-IDF specialty classifiers.")
    parser.add_argument(
        "--label-mode",
        choices=["specialty", "rare_merged", "broad"],
        default="specialty",
        help="Train on original labels, rare-category merged labels, or broad specialty groups.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=50,
        help="Minimum class count used when --label-mode rare_merged is selected.",
    )
    args = parser.parse_args()
    train(label_mode=args.label_mode, min_class_count=args.min_class_count)
