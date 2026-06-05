from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import RAW_DATA_DIR, DATA_DIR, ensure_project_dirs


LIKELY_COLUMNS = {
    "question",
    "questions",
    "answer",
    "answers",
    "specialty",
    "category",
    "text",
    "message",
    "label",
    "class",
}


def detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    if suffix == ".json":
        return "json"
    if suffix in {".txt", ".md"}:
        return "text"
    return suffix.lstrip(".") or "unknown"


def load_preview(path: Path) -> pd.DataFrame | None:
    file_type = detect_file_type(path)
    try:
        if file_type == "csv":
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            return pd.read_csv(path, sep=sep, nrows=5)
        if file_type == "excel":
            return pd.read_excel(path, nrows=5)
        if file_type == "json":
            try:
                return pd.read_json(path, lines=True, nrows=5)
            except ValueError:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return pd.DataFrame(data[:5])
                if isinstance(data, dict):
                    return pd.json_normalize(data).head()
        if file_type == "text":
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
            return pd.DataFrame({"line": lines})
    except Exception as exc:
        print(f"  Could not load preview: {exc}")
    return None


def inspect_dataset(raw_dir: Path = RAW_DATA_DIR) -> None:
    ensure_project_dirs()
    print(f"Raw data directory: {raw_dir}")
    if not raw_dir.exists():
        alternative = next(DATA_DIR.glob("MedQA-MA*"), None)
        if alternative and alternative.exists():
            raw_dir = alternative
            print(f"Using alternative raw data directory: {raw_dir}")
        else:
            print("Raw data directory does not exist.")
            return

    files = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".tsv", ".json", ".xlsx", ".xls", ".txt"}
    )
    if not files:
        print("No indexable files found.")
        return

    for path in files:
        rel_path = path.relative_to(raw_dir)
        size_kb = path.stat().st_size / 1024
        file_type = detect_file_type(path)
        print("\n" + "=" * 80)
        print(f"File: {rel_path}")
        print(f"Size: {size_kb:,.2f} KB")
        print(f"Detected type: {file_type}")

        preview = load_preview(path)
        if preview is None:
            print("No tabular/text preview available.")
            continue

        print(f"Columns: {list(preview.columns)}")
        likely = [
            col for col in preview.columns if str(col).strip().lower() in LIKELY_COLUMNS
        ]
        print(f"Likely useful columns: {likely or 'none detected'}")
        print("Preview:")
        print(preview.head().to_string(index=False))


if __name__ == "__main__":
    inspect_dataset()
