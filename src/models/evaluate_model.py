from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_project_dirs


def evaluate() -> dict[str, object]:
    ensure_project_dirs()
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv").fillna("")
    model = joblib.load(MODELS_DIR / "specialty_classifier.joblib")
    preds = model.predict(test_df["text"])
    labels = sorted(test_df["specialty"].unique())
    metrics = {
        "accuracy": accuracy_score(test_df["specialty"], preds),
        "macro_f1": f1_score(test_df["specialty"], preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(test_df["specialty"], preds, average="weighted", zero_division=0),
        "macro_precision": precision_score(test_df["specialty"], preds, average="macro", zero_division=0),
        "macro_recall": recall_score(test_df["specialty"], preds, average="macro", zero_division=0),
        "classification_report": classification_report(test_df["specialty"], preds, zero_division=0, output_dict=True),
    }
    (METRICS_DIR / "specialty_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    cm = confusion_matrix(test_df["specialty"], preds, labels=labels)
    fig, ax = plt.subplots(figsize=(14, 14))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    analysis_df = test_df[["id", "text", "language", "specialty", "source"]].copy()
    analysis_df["prediction"] = preds
    correct = analysis_df[analysis_df["specialty"] == analysis_df["prediction"]].head(20)
    wrong = analysis_df[analysis_df["specialty"] != analysis_df["prediction"]].head(50)
    wrong.to_csv(METRICS_DIR / "wrong_predictions.csv", index=False)
    correct.to_csv(METRICS_DIR / "correct_predictions.csv", index=False)
    write_error_report(metrics, correct, wrong)
    print(f"Saved metrics to {METRICS_DIR / 'specialty_metrics.json'}")
    return metrics


def write_error_report(metrics: dict[str, object], correct: pd.DataFrame, wrong: pd.DataFrame) -> None:
    lines = [
        "# Evaluation and Error Analysis",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
        "",
        "## Observations",
        "- TF-IDF is a strong transparent baseline, but it relies on surface forms and can struggle with spelling variation across Darija, Arabic, and French.",
        "- Rare custom-only specialties and Latin-script Darija examples remain higher risk than frequent Arabic-script MedQA-MA labels.",
        "- Some MedQA-MA rows are vague or contain generated assistant-like text, which can blur specialty boundaries.",
        "",
        "## Correct prediction examples",
    ]
    for _, row in correct.head(5).iterrows():
        lines.append(f"- `{row['text'][:160]}` -> {row['prediction']}")
    lines.append("")
    lines.append("## Wrong prediction examples")
    if wrong.empty:
        lines.append("- No wrong examples found in the first evaluated test pass.")
    else:
        for _, row in wrong.head(10).iterrows():
            lines.append(f"- `{row['text'][:160]}` true={row['specialty']} predicted={row['prediction']}")
    (REPORTS_DIR / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    evaluate()
