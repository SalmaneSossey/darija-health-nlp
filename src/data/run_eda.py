from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.text_normalization import has_arabic, has_latin, normalize_text
from src.utils.paths import FIGURES_DIR, RAW_DATA_DIR, REPORTS_DIR, ensure_project_dirs


MASTER_FILE = RAW_DATA_DIR / "Dataset" / "MedQA_Ma dataset" / "MedQA_MA.csv"
FRENCH_WORDS = {
    "douleur",
    "mal",
    "tete",
    "tête",
    "ventre",
    "fièvre",
    "fievre",
    "toux",
    "respirer",
    "poitrine",
    "depuis",
}
LATIN_DARIJA_TOKENS = {
    "3ndi",
    "andi",
    "sda3",
    "dwakha",
    "skhana",
    "k7a",
    "wje3",
    "wja3",
    "kerchi",
    "sedri",
    "di9",
    "nefs",
    "haboub",
}


def find_main_dataset() -> Path:
    if MASTER_FILE.exists():
        return MASTER_FILE
    csv_files = sorted(RAW_DATA_DIR.rglob("*.csv"), key=lambda path: path.stat().st_size, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {RAW_DATA_DIR}")
    return csv_files[0]


def detect_language(text: object) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return "unknown"
    arabic = has_arabic(normalized)
    latin = has_latin(normalized)
    tokens = set(re.findall(r"[\wÀ-ÿ]+", normalized.lower()))
    has_french = bool(tokens & FRENCH_WORDS)
    has_darija_latin = bool(tokens & LATIN_DARIJA_TOKENS) or bool(re.search(r"[379]", normalized))
    if arabic and latin:
        return "mixed"
    if arabic:
        return "arabic_darija"
    if latin and has_darija_latin and has_french:
        return "mixed"
    if latin and has_darija_latin:
        return "latin_darija"
    if latin and has_french:
        return "french"
    if latin:
        return "french"
    return "unknown"


def load_dataset() -> pd.DataFrame:
    path = find_main_dataset()
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    print(f"Loaded {path} with shape {df.shape}")
    return df


def useful_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    lowered = {col.lower(): col for col in df.columns}
    text_col = (
        lowered.get("question")
        or lowered.get("question_darija")
        or lowered.get("question_darija_processesd")
        or lowered.get("question_darija_processed")
        or lowered.get("text")
    )
    label_col = lowered.get("category") or lowered.get("specialty") or lowered.get("label")
    if not text_col:
        raise ValueError(f"Could not identify a question/text column from {list(df.columns)}")
    return text_col, label_col


def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path, top_n: int | None = None) -> None:
    values = series.head(top_n) if top_n else series
    plt.figure(figsize=(12, max(5, min(12, len(values) * 0.35))))
    plt.barh([str(index) for index in values.index], values.values)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_eda() -> dict[str, object]:
    ensure_project_dirs()
    df = load_dataset()
    text_col, label_col = useful_columns(df)

    working = df.copy()
    working[text_col] = working[text_col].fillna("").astype(str)
    working["normalized_text"] = working[text_col].map(normalize_text)
    working["language"] = working[text_col].map(detect_language)
    working["char_len"] = working["normalized_text"].str.len()
    working["word_count"] = working["normalized_text"].str.split().map(len)

    duplicates = int(working.duplicated().sum())
    repeated_questions = int(working.duplicated(subset=[text_col]).sum())
    missing_values = working.isna().sum().sort_values(ascending=False)
    empty_questions = int((working["normalized_text"] == "").sum())
    empty_labels = int(working[label_col].isna().sum()) if label_col else 0

    label_counts = working[label_col].fillna("unknown").astype(str).str.strip().value_counts() if label_col else pd.Series(dtype=int)
    language_counts = working["language"].value_counts()
    rare_classes = label_counts[label_counts < 50]

    token_counter: Counter[str] = Counter()
    arabic_token_counter: Counter[str] = Counter()
    for text in working["normalized_text"]:
        tokens = re.findall(r"[\wÀ-ÿ\u0600-\u06FF]+", text)
        token_counter.update(token for token in tokens if len(token) > 1)
        arabic_token_counter.update(token for token in tokens if has_arabic(token) and len(token) > 1)

    top_tokens = pd.Series(dict(token_counter.most_common(30)))
    top_arabic_tokens = pd.Series(dict(arabic_token_counter.most_common(30)))

    save_bar(label_counts, "Class Distribution", "Rows", "Specialty", FIGURES_DIR / "class_distribution.png", top_n=30)
    plt.figure(figsize=(10, 6))
    plt.hist(working["char_len"], bins=60)
    plt.title("Text Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=160)
    plt.close()
    save_bar(top_tokens, "Top Tokens", "Frequency", "Token", FIGURES_DIR / "top_tokens.png")
    save_bar(missing_values[missing_values > 0] if (missing_values > 0).any() else pd.Series({"none": 0}), "Missing Values", "Missing Rows", "Column", FIGURES_DIR / "missing_values.png")
    save_bar(language_counts, "Language Distribution", "Rows", "Language", FIGURES_DIR / "language_distribution.png")

    very_short = working.nsmallest(10, "char_len")[[text_col, "char_len", "language"]].to_dict("records")
    very_long = working.nlargest(10, "char_len")[[text_col, "char_len", "language"]].to_dict("records")

    summary = {
        "dataset_size": {"rows": int(len(working)), "columns": int(len(df.columns))},
        "columns": list(df.columns),
        "useful_columns": {"text": text_col, "label": label_col},
        "duplicates": duplicates,
        "repeated_questions": repeated_questions,
        "empty_questions": empty_questions,
        "empty_labels": empty_labels,
        "text_length": {
            "average_chars": float(working["char_len"].mean()),
            "min_chars": int(working["char_len"].min()),
            "max_chars": int(working["char_len"].max()),
            "average_words": float(working["word_count"].mean()),
        },
        "num_specialties": int(label_counts.size),
        "top_specialties": label_counts.head(15).to_dict(),
        "least_frequent_specialties": label_counts.tail(15).to_dict(),
        "rare_classes_under_50": rare_classes.to_dict(),
        "language_distribution": language_counts.to_dict(),
        "top_tokens": top_tokens.to_dict(),
        "top_arabic_tokens": top_arabic_tokens.to_dict(),
        "very_short_examples": very_short,
        "very_long_examples": very_long,
    }

    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(summary)
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved EDA report to {REPORTS_DIR / 'eda_summary.md'}")
    return summary


def write_markdown_report(summary: dict[str, object]) -> None:
    top_specialties = summary["top_specialties"]
    language_distribution = summary["language_distribution"]
    rare_classes = summary["rare_classes_under_50"]

    lines = [
        "# EDA Summary",
        "",
        "## Dataset size",
        f"- Rows: {summary['dataset_size']['rows']}",
        f"- Columns: {summary['dataset_size']['columns']}",
        f"- Column names: {', '.join(summary['columns'])}",
        "",
        "## Useful columns",
        f"- Text column: `{summary['useful_columns']['text']}`",
        f"- Specialty label column: `{summary['useful_columns']['label']}`",
        "- The answer column is useful for qualitative context, but V1 specialty classification uses patient questions as model input.",
        "",
        "## Main labels",
        f"- Number of specialties: {summary['num_specialties']}",
        "- Most frequent specialties:",
    ]
    lines.extend(f"  - {label}: {count}" for label, count in list(top_specialties.items())[:10])
    lines.extend(
        [
            "",
            "## Class imbalance observations",
            f"- Rare classes with fewer than 50 rows: {len(rare_classes)}",
            "- The dataset is imbalanced, so baseline models should use class weighting and macro F1 in addition to accuracy.",
            "",
            "## Text quality observations",
            f"- Duplicate full rows: {summary['duplicates']}",
            f"- Repeated questions: {summary['repeated_questions']}",
            f"- Empty questions after normalization: {summary['empty_questions']}",
            f"- Average character length: {summary['text_length']['average_chars']:.2f}",
            f"- Maximum character length: {summary['text_length']['max_chars']}",
            "- Several rows contain vague prompts, unclear questions, or answer-like/generated text. V1 keeps the dataset but removes empty texts and exact duplicate text-label pairs.",
            "",
            "## Language and script observations",
        ]
    )
    lines.extend(f"- {label}: {count}" for label, count in language_distribution.items())
    lines.extend(
        [
            "",
            "## Preprocessing decisions justified by EDA",
            "- Use the master `MedQA_MA.csv` file as the canonical source to avoid double-counting per-specialty slice files.",
            "- Normalize Arabic letter variants and whitespace because the data is mostly Arabic-script Darija with inconsistent spelling.",
            "- Preserve Arabizi digits such as `3`, `7`, and `9` because custom and real Moroccan Latin-script messages depend on them.",
            "- Remove empty questions and exact duplicate text-specialty pairs because EDA found repeated and low-information rows.",
            "- Add a small custom triage dataset because MedQA-MA provides specialty labels but not reliable urgency or symptom labels.",
        ]
    )
    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_eda()
