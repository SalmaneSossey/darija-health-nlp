from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_project_dirs


BEST_MODEL_NAME = "char_wb_tfidf_linear_svm"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_barh(
    labels: Iterable[str],
    values: Iterable[float],
    title: str,
    xlabel: str,
    output_path: Path,
    color: str = "#4c78a8",
) -> None:
    label_list = [str(label) for label in labels]
    value_list = list(values)
    height = max(4.5, min(12, len(label_list) * 0.45))
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(label_list, value_list, color=color)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def model_comparison_table(validation_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics in validation_metrics.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def analyze_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False, observed=False):
        if len(group) == 0:
            continue
        rows.append(
            {
                group_col: value,
                "support": int(len(group)),
                "accuracy": accuracy_score(group["specialty"], group["prediction"]),
                "macro_f1": f1_score(group["specialty"], group["prediction"], average="macro", zero_division=0),
                "error_rate": float((group["specialty"] != group["prediction"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["error_rate", "support"], ascending=[False, False]).reset_index(drop=True)


def top_confusion_pairs(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    support_by_label = dict(zip(labels, cm.sum(axis=1), strict=True))
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i == j or cm[i, j] == 0:
                continue
            support = int(support_by_label[true_label])
            rows.append(
                {
                    "true_specialty": true_label,
                    "predicted_specialty": pred_label,
                    "count": int(cm[i, j]),
                    "true_class_support": support,
                    "share_of_true_class": float(cm[i, j] / support) if support else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["count", "share_of_true_class"], ascending=False).reset_index(drop=True)


def add_text_length_bins(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["char_length"] = working["text"].astype(str).str.len()
    quantiles = working["char_length"].quantile([0.2, 0.4, 0.6, 0.8]).to_list()
    bins = [-1, *quantiles, float("inf")]
    labels = ["very_short", "short", "medium", "long", "very_long"]
    working["text_length_bin"] = pd.cut(working["char_length"], bins=bins, labels=labels, duplicates="drop")
    return working


def write_academic_report(
    metrics: dict[str, object],
    label_mapping: dict[str, object],
    comparison_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    confusions_df: pd.DataFrame,
    language_df: pd.DataFrame,
    source_df: pd.DataFrame,
    length_df: pd.DataFrame,
    wrong_examples: pd.DataFrame,
) -> None:
    best_row = comparison_df.iloc[0]
    baseline_rows = comparison_df[comparison_df["model"] != BEST_MODEL_NAME]
    best_macro = float(metrics["macro_f1"])
    best_accuracy = float(metrics["accuracy"])
    best_weighted = float(metrics["weighted_f1"])
    v1_svm = comparison_df[comparison_df["model"] == "tfidf_linear_svm"]
    macro_gain = None
    if not v1_svm.empty:
        macro_gain = float(best_row["macro_f1"] - v1_svm.iloc[0]["macro_f1"])

    weakest = per_class_df.sort_values("f1-score").head(8)
    strongest = per_class_df.sort_values("f1-score", ascending=False).head(8)

    lines = [
        "# V2 Error Analysis: Darija Health NLP",
        "",
        "## Scope",
        "",
        (
            "This report analyzes the V2 specialty classifier, focusing on the selected "
            f"`{label_mapping.get('best_model', BEST_MODEL_NAME)}` model. The model is evaluated on the held-out test split "
            "and compared against the V1 word-level TF-IDF baselines using validation-set metrics."
        ),
        "",
        "## Overall performance",
        "",
        f"- Test accuracy: {best_accuracy:.4f}",
        f"- Test macro F1: {best_macro:.4f}",
        f"- Test weighted F1: {best_weighted:.4f}",
        f"- Label mode: `{label_mapping.get('label_mode', 'specialty')}`",
        "",
        "The weighted F1 is higher than macro F1, which indicates that performance is better on frequent or easier classes than on the most difficult specialties. This gap is expected in a noisy multilingual medical-orientation dataset where several specialties share lexical cues.",
        "",
        "## V1 baseline comparison",
        "",
        "| Model | Validation accuracy | Validation macro F1 | Validation weighted F1 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in comparison_df.iterrows():
        lines.append(
            f"| `{row['model']}` | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['weighted_f1']:.4f} |"
        )
    lines.append("")
    if macro_gain is not None:
        lines.append(
            f"The character n-gram model improves validation macro F1 by {macro_gain:.4f} over the V1 word-level LinearSVC baseline. "
            "This supports the V2 hypothesis that character n-grams are better suited to Darija spelling variation, Arabizi transliteration, and mixed-script surface forms."
        )
    else:
        lines.append(
            "The character n-gram model is the best validation model among the compared baselines. "
            "This supports using subword-style lexical features for noisy Darija and mixed-script text."
        )
    if not baseline_rows.empty:
        lines.append(
            "The result does not remove the need for transformer comparisons; it establishes a stronger classical baseline that future MARBERT, AraBERT, or multilingual BERT experiments should beat."
        )

    lines.extend(
        [
            "",
            "## Most confused specialties",
            "",
            "| True specialty | Predicted specialty | Count | Share of true class |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in confusions_df.head(12).iterrows():
        lines.append(
            f"| {row['true_specialty']} | {row['predicted_specialty']} | {int(row['count'])} | {row['share_of_true_class']:.3f} |"
        )

    lines.extend(
        [
            "",
            "The highest-confusion pairs tend to involve clinically adjacent or lexically overlapping domains. In this dataset, many questions are short, ambiguous, or phrased as general advice requests, so a word or character TF-IDF model often learns specialty-associated vocabulary rather than deeper clinical intent.",
            "",
            "## Per-class behavior",
            "",
            "Lowest F1 specialties:",
        ]
    )
    for _, row in weakest.iterrows():
        lines.append(
            f"- {row['specialty']}: F1={row['f1-score']:.3f}, precision={row['precision']:.3f}, recall={row['recall']:.3f}, support={int(row['support'])}"
        )
    lines.append("")
    lines.append("Highest F1 specialties:")
    for _, row in strongest.iterrows():
        lines.append(
            f"- {row['specialty']}: F1={row['f1-score']:.3f}, precision={row['precision']:.3f}, recall={row['recall']:.3f}, support={int(row['support'])}"
        )

    lines.extend(
        [
            "",
            "## Language and script analysis",
            "",
            "| Language/script | Support | Accuracy | Macro F1 | Error rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in language_df.iterrows():
        lines.append(
            f"| {row['language']} | {int(row['support'])} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['error_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The raw MedQA-MA portion is dominated by Arabic-script Darija; therefore, language-specific conclusions for Latin Darija, French, and mixed language must be treated as preliminary. The expanded custom examples improve coverage for these scripts, but they are still not a substitute for clinically reviewed real patient messages.",
            "",
            "## Source and text-length analysis",
            "",
            "| Source | Support | Accuracy | Macro F1 | Error rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in source_df.iterrows():
        lines.append(
            f"| {row['source']} | {int(row['support'])} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['error_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Text length bin | Support | Accuracy | Error rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in length_df.iterrows():
        lines.append(
            f"| {row['text_length_bin']} | {int(row['support'])} | {row['accuracy']:.4f} | {row['error_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Qualitative error examples",
            "",
        ]
    )
    if wrong_examples.empty:
        lines.append("No errors were found in the evaluated split.")
    else:
        for _, row in wrong_examples.head(10).iterrows():
            text = str(row["text"]).replace("|", " ")[:180]
            lines.append(f"- `{text}` true={row['specialty']} predicted={row['prediction']} language={row['language']}")

    lines.extend(
        [
            "",
            "## Academic interpretation",
            "",
            "The V2 results show that a spelling-robust classical model is a meaningful baseline for Moroccan medical triage orientation. Character n-grams partially address Darija orthographic variation because they can match subword fragments across spelling variants, Arabic-script forms, Arabizi forms, and French loanwords. However, the remaining errors demonstrate that surface-form models cannot reliably disambiguate specialties when patient messages are short, vague, or contain symptoms shared by multiple departments.",
            "",
            "The model should therefore be interpreted as an orientation system, not a diagnostic system. For deployment, specialty classification should remain coupled with explicit safety rules, symptom extraction, and conservative recommendations. The next research step is to compare this stronger TF-IDF baseline with contextual multilingual encoders such as MARBERT, AraBERT, and multilingual BERT, while preserving the transparent baseline for error analysis and reproducibility.",
            "",
            "## Generated artifacts",
            "",
            "- `artifacts/metrics/v2_per_class_metrics.csv`",
            "- `artifacts/metrics/v2_top_confusions.csv`",
            "- `artifacts/metrics/v2_language_error_analysis.csv`",
            "- `artifacts/metrics/v2_source_error_analysis.csv`",
            "- `artifacts/metrics/v2_text_length_error_analysis.csv`",
            "- `artifacts/metrics/v2_wrong_predictions.csv`",
            "- `artifacts/figures/v2_model_comparison_macro_f1.png`",
            "- `artifacts/figures/v2_normalized_confusion_matrix.png`",
            "- `artifacts/figures/v2_top_confusions.png`",
            "- `artifacts/figures/v2_language_error_rates.png`",
            "- `artifacts/figures/v2_per_class_f1.png`",
            "- `artifacts/figures/v2_text_length_error_rates.png`",
        ]
    )
    report = "\n".join(lines) + "\n"
    (REPORTS_DIR / "v2_error_analysis.md").write_text(report, encoding="utf-8")
    (REPORTS_DIR / "error_analysis.md").write_text(report, encoding="utf-8")


def run_v2_error_analysis() -> dict[str, object]:
    ensure_project_dirs()
    validation_metrics = _load_json(METRICS_DIR / "validation_model_comparison.json")
    label_mapping = _load_json(MODELS_DIR / "label_mapping.json")
    model = joblib.load(MODELS_DIR / "specialty_classifier.joblib")
    test_df = add_text_length_bins(pd.read_csv(PROCESSED_DATA_DIR / "test.csv").fillna(""))
    test_df["prediction"] = model.predict(test_df["text"])
    test_df["is_correct"] = test_df["specialty"] == test_df["prediction"]

    labels = sorted(set(test_df["specialty"]) | set(test_df["prediction"]))
    metrics = {
        "best_model": label_mapping.get("best_model", BEST_MODEL_NAME),
        "accuracy": accuracy_score(test_df["specialty"], test_df["prediction"]),
        "macro_f1": f1_score(test_df["specialty"], test_df["prediction"], average="macro", zero_division=0),
        "weighted_f1": f1_score(test_df["specialty"], test_df["prediction"], average="weighted", zero_division=0),
    }
    class_report = classification_report(
        test_df["specialty"],
        test_df["prediction"],
        zero_division=0,
        output_dict=True,
    )
    metrics["classification_report"] = class_report
    (METRICS_DIR / "v2_specialty_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (METRICS_DIR / "specialty_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    comparison_df = model_comparison_table(validation_metrics)
    comparison_df.to_csv(METRICS_DIR / "v2_model_comparison.csv", index=False)

    per_class_df = (
        pd.DataFrame(class_report)
        .T.reset_index()
        .rename(columns={"index": "specialty"})
        .query("specialty not in ['accuracy', 'macro avg', 'weighted avg']")
        .sort_values("f1-score")
        .reset_index(drop=True)
    )
    per_class_df.to_csv(METRICS_DIR / "v2_per_class_metrics.csv", index=False)

    confusions_df = top_confusion_pairs(test_df["specialty"], test_df["prediction"], labels)
    confusions_df.to_csv(METRICS_DIR / "v2_top_confusions.csv", index=False)

    language_df = analyze_by_group(test_df, "language")
    source_df = analyze_by_group(test_df, "source")
    length_df = analyze_by_group(test_df, "text_length_bin")
    language_df.to_csv(METRICS_DIR / "v2_language_error_analysis.csv", index=False)
    source_df.to_csv(METRICS_DIR / "v2_source_error_analysis.csv", index=False)
    length_df.to_csv(METRICS_DIR / "v2_text_length_error_analysis.csv", index=False)

    wrong_examples = test_df.loc[~test_df["is_correct"], ["id", "text", "language", "source", "specialty", "prediction", "char_length"]]
    correct_examples = test_df.loc[test_df["is_correct"], ["id", "text", "language", "source", "specialty", "prediction", "char_length"]]
    wrong_examples.to_csv(METRICS_DIR / "v2_wrong_predictions.csv", index=False)
    correct_examples.head(100).to_csv(METRICS_DIR / "v2_correct_prediction_examples.csv", index=False)
    wrong_examples.head(100).to_csv(METRICS_DIR / "wrong_predictions.csv", index=False)
    correct_examples.head(100).to_csv(METRICS_DIR / "correct_predictions.csv", index=False)

    _save_barh(
        comparison_df["model"],
        comparison_df["macro_f1"],
        "Validation Macro F1 by Model",
        "Macro F1",
        FIGURES_DIR / "v2_model_comparison_macro_f1.png",
        color="#2f7f6f",
    )
    _save_barh(
        [f"{row.true_specialty} -> {row.predicted_specialty}" for row in confusions_df.head(15).itertuples()],
        confusions_df.head(15)["count"],
        "Top Specialty Confusion Pairs",
        "Misclassified examples",
        FIGURES_DIR / "v2_top_confusions.png",
        color="#b75d69",
    )
    _save_barh(
        language_df["language"],
        language_df["error_rate"],
        "Error Rate by Language/Script",
        "Error rate",
        FIGURES_DIR / "v2_language_error_rates.png",
        color="#6b5b95",
    )
    _save_barh(
        length_df["text_length_bin"],
        length_df["error_rate"],
        "Error Rate by Text Length Bin",
        "Error rate",
        FIGURES_DIR / "v2_text_length_error_rates.png",
        color="#d18f3b",
    )
    strongest_and_weakest = pd.concat([per_class_df.head(10), per_class_df.tail(10)]).drop_duplicates("specialty")
    _save_barh(
        strongest_and_weakest["specialty"],
        strongest_and_weakest["f1-score"],
        "Per-Class F1: Weakest and Strongest Specialties",
        "F1 score",
        FIGURES_DIR / "v2_per_class_f1.png",
        color="#4c78a8",
    )

    cm = confusion_matrix(test_df["specialty"], test_df["prediction"], labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(16, 16))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=True, values_format=".2f")
    ax.set_title("Normalized Confusion Matrix: V2 Character N-gram LinearSVC")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "v2_normalized_confusion_matrix.png", dpi=170)
    plt.close(fig)

    write_academic_report(
        metrics,
        label_mapping,
        comparison_df,
        per_class_df,
        confusions_df,
        language_df,
        source_df,
        length_df,
        wrong_examples,
    )
    print(f"Saved V2 error analysis report to {REPORTS_DIR / 'v2_error_analysis.md'}")
    return metrics


if __name__ == "__main__":
    run_v2_error_analysis()
