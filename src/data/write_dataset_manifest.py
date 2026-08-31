from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR


REQUIRED_SPLITS = ("train.csv", "valid.csv", "test.csv")
MANIFEST_VERSION = "v2.1-dataset-provenance"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def split_summary(path: Path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(path).fillna("")
    required_columns = {"id", "text", "language", "specialty", "urgency", "symptoms", "source"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    text_series = df["text"].astype(str)
    return {
        "file": str(path.relative_to(path.parents[2])),
        "sha256": file_sha256(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "labelDistribution": df["specialty"].astype(str).value_counts().sort_index().to_dict(),
        "languageDistribution": df["language"].astype(str).value_counts().sort_index().to_dict(),
        "sourceDistribution": df["source"].astype(str).value_counts().sort_index().to_dict(),
        "duplicateIds": int(df["id"].duplicated().sum()),
        "duplicateTextSpecialtyRows": int(df.duplicated(subset=["text", "specialty"]).sum()),
        "emptyTextRows": int((text_series.str.strip() == "").sum()),
        "avgTextChars": round(float(text_series.str.len().mean()), 2) if len(df) else 0.0,
    }


def leakage_report(splits: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    names = sorted(splits)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = splits[left_name]
            right = splits[right_name]
            text_overlap = set(left["text"].astype(str)) & set(right["text"].astype(str))
            id_overlap = set(left["id"].astype(str)) & set(right["id"].astype(str))
            report[f"{left_name}_vs_{right_name}"] = {
                "textOverlap": len(text_overlap),
                "idOverlap": len(id_overlap),
            }
    return report


def build_manifest() -> dict[str, Any]:
    missing = [name for name in REQUIRED_SPLITS if not (PROCESSED_DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed split files: "
            + ", ".join(missing)
            + ". Restore MedQA-MA raw data and run `python src/data/build_processed_dataset.py` first."
        )

    import pandas as pd

    split_frames = {
        name.removesuffix(".csv"): pd.read_csv(PROCESSED_DATA_DIR / name).fillna("")
        for name in REQUIRED_SPLITS
    }
    split_summaries = {
        name.removesuffix(".csv"): split_summary(PROCESSED_DATA_DIR / name)
        for name in REQUIRED_SPLITS
    }
    all_sources = sorted(
        {
            source
            for df in split_frames.values()
            for source in df["source"].astype(str).unique().tolist()
            if source
        }
    )
    all_labels = sorted(
        {
            label
            for df in split_frames.values()
            for label in df["specialty"].astype(str).unique().tolist()
            if label
        }
    )
    return {
        "manifestVersion": MANIFEST_VERSION,
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "gitRevision": git_revision(),
        "randomSeed": 42,
        "splitStrategy": "80/10/10 sklearn train_test_split with specialty stratification when class counts allow it",
        "schema": "id,text,language,specialty,urgency,symptoms,source",
        "sources": all_sources,
        "labels": all_labels,
        "splits": split_summaries,
        "leakageChecks": leakage_report(split_frames),
        "notes": [
            "Raw and processed medical data are intentionally not committed to Git.",
            "This manifest is the reproducibility contract for Kaggle/Colab/VPS handoff.",
            "The final test split must not be used during model selection.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducible manifest for Darija Health NLP splits.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "reports" / "dataset_manifest.json",
        help="Manifest output path.",
    )
    args = parser.parse_args()

    try:
        manifest = build_manifest()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote dataset manifest to {args.output}")


if __name__ == "__main__":
    main()
