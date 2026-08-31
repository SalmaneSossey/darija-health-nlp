from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.write_dataset_manifest import REQUIRED_SPLITS, build_manifest
from src.utils.paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR


DEFAULT_OUTPUT = ARTIFACTS_DIR / "kaggle" / "darija_model_training_package.zip"
PACKAGE_README = """# Darija Health NLP Kaggle Training Package

Use this package in a Kaggle notebook with GPU enabled.

Recommended command:

```bash
python src/models/train_transformer_specialty_classifier.py \\
  --model marbert \\
  --epochs 4 \\
  --batch-size 16 \\
  --max-length 160 \\
  --class-weighted
```

Rules:
- Use `valid.csv` for model selection.
- Use `test.csv` only for final reporting.
- Export the resulting model folder with `model.safetensors`, tokenizer files, `training_manifest.json`, and `test_metrics.json`.
- Do not use OpenAI or paid APIs for this experiment; the goal is a local deployable model.
"""


def add_file(zip_file: ZipFile, path: Path, arcname: str | None = None) -> None:
    if path.exists() and path.is_file():
        zip_file.write(path, arcname=arcname or str(path.relative_to(PROJECT_ROOT)))


def add_tree(zip_file: ZipFile, folder: Path, patterns: tuple[str, ...]) -> None:
    if not folder.exists():
        return
    for pattern in patterns:
        for path in sorted(folder.rglob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                zip_file.write(path, arcname=str(path.relative_to(PROJECT_ROOT)))


def create_package(output_path: Path, include_data: bool, allow_missing_data: bool) -> Path:
    missing = [name for name in REQUIRED_SPLITS if not (PROCESSED_DATA_DIR / name).exists()]
    if include_data and missing and not allow_missing_data:
        raise FileNotFoundError(
            "Cannot create Kaggle package with data. Missing: "
            + ", ".join(missing)
            + ". Rebuild splits first or use --allow-missing-data for a code-only package."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr("README_KAGGLE.md", PACKAGE_README)
        add_file(zip_file, PROJECT_ROOT / "requirements-transformers.txt")
        add_file(zip_file, PROJECT_ROOT / "requirements-dev.txt")
        add_file(zip_file, PROJECT_ROOT / "README.md")
        add_tree(zip_file, PROJECT_ROOT / "docs", ("*.md",))
        add_tree(zip_file, PROJECT_ROOT / "src", ("*.py",))
        add_tree(zip_file, PROJECT_ROOT / "backend" / "app", ("*.py",))
        add_tree(zip_file, PROJECT_ROOT / "tests", ("*.py",))
        add_file(zip_file, PROJECT_ROOT / "train_low_vram.py")

        if include_data and not missing:
            manifest = build_manifest()
            zip_file.writestr("artifacts/reports/dataset_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            for name in REQUIRED_SPLITS:
                add_file(zip_file, PROCESSED_DATA_DIR / name, f"data/processed/{name}")
        else:
            zip_file.writestr(
                "DATA_MISSING.txt",
                "Processed splits were not included. Restore/build data/processed/train.csv, valid.csv, and test.csv before training.\n",
            )

    print(f"Wrote Kaggle training package to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Kaggle-ready Darija Health NLP training package.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-data", action="store_true", help="Create a code-only package even if splits exist.")
    parser.add_argument("--allow-missing-data", action="store_true", help="Create a code-only package if splits are missing.")
    args = parser.parse_args()

    create_package(
        output_path=args.output,
        include_data=not args.no_data,
        allow_missing_data=args.allow_missing_data,
    )


if __name__ == "__main__":
    main()
