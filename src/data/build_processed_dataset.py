from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.create_custom_triage_examples import create_custom_examples
from src.data.run_eda import detect_language, find_main_dataset
from src.features.symptom_extractor import extract_symptoms
from src.features.text_normalization import normalize_text
from src.utils.paths import PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, ensure_project_dirs


SPECIALTY_NORMALIZATION = {
    "gynecologists": "Obstetrics and Gynecology",
    "ophthalmologist": "Ophthalmology",
    "Cosmetic dermatologist": "Cosmetic Dermatology",
    "Cosmetic dermatologist ": "Cosmetic Dermatology",
    "Diabetes mellitus": "Endocrinology",
    "Dietetics / Nutrition": "Dietetics and Nutrition",
    "General Medicine": "General Practice",
    "Respiratory diseases": "Pulmonology",
}


def find_columns(df: pd.DataFrame) -> tuple[str, str]:
    lowered = {col.lower().strip(): col for col in df.columns}
    text_col = lowered.get("question") or lowered.get("question_darija") or lowered.get("question_darija_processesd")
    label_col = lowered.get("category") or lowered.get("specialty") or lowered.get("label")
    if not text_col or not label_col:
        raise ValueError(f"Could not identify text/label columns from {list(df.columns)}")
    return text_col, label_col


def normalize_specialty(label: object) -> str:
    label_text = str(label).strip()
    return SPECIALTY_NORMALIZATION.get(label_text, label_text)


def build_medqa_dataset() -> pd.DataFrame:
    path = find_main_dataset()
    raw = pd.read_csv(path)
    text_col, label_col = find_columns(raw)
    df = pd.DataFrame(
        {
            "text": raw[text_col].map(normalize_text),
            "language": raw[text_col].map(detect_language),
            "specialty": raw[label_col].map(normalize_specialty),
            "urgency": "unknown",
            "symptoms": raw[text_col].map(lambda text: ";".join(extract_symptoms(text))),
            "source": "medqa_ma",
        }
    )
    df = df[(df["text"] != "") & (df["specialty"] != "")]
    df = df.drop_duplicates(subset=["text", "specialty"]).reset_index(drop=True)
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = df["specialty"] if df["specialty"].value_counts().min() >= 3 else None
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    temp_stratify = temp_df["specialty"] if temp_df["specialty"].value_counts().min() >= 2 else None
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_stratify,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_processed_dataset() -> pd.DataFrame:
    ensure_project_dirs()
    medqa = build_medqa_dataset()
    custom = create_custom_examples()
    combined = pd.concat([medqa, custom], ignore_index=True)
    combined["id"] = [f"triage_{index:06d}" for index in range(len(combined))]
    combined = combined[["id", "text", "language", "specialty", "urgency", "symptoms", "source"]]
    combined = combined.drop_duplicates(subset=["text", "specialty"], keep="last").reset_index(drop=True)
    combined["id"] = [f"triage_{index:06d}" for index in range(len(combined))]

    combined.to_csv(PROCESSED_DATA_DIR / "triage_dataset.csv", index=False)
    train_df, valid_df, test_df = split_dataset(combined)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    valid_df.to_csv(PROCESSED_DATA_DIR / "valid.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    combined.head(25).to_csv(SAMPLE_DATA_DIR / "sample_medqa_ma.csv", index=False)
    print(f"Saved processed dataset: {combined.shape}")
    print(f"Train/valid/test: {train_df.shape}, {valid_df.shape}, {test_df.shape}")
    return combined


if __name__ == "__main__":
    build_processed_dataset()
