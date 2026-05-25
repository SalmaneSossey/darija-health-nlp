from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.paths import PROCESSED_DATA_DIR


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "colab"
OUTPUT_ZIP = OUTPUT_DIR / "darija_health_processed_splits.zip"
REQUIRED_SPLITS = ["train.csv", "valid.csv", "test.csv"]


def export_colab_training_data() -> Path:
    missing = [name for name in REQUIRED_SPLITS if not (PROCESSED_DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed split files: "
            + ", ".join(missing)
            + ". Run python src/data/build_processed_dataset.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED) as zip_file:
        for name in REQUIRED_SPLITS:
            source = PROCESSED_DATA_DIR / name
            zip_file.write(source, arcname=f"data/processed/{name}")

    print(f"Saved Colab training data package to {OUTPUT_ZIP}")
    print(f"Size: {OUTPUT_ZIP.stat().st_size / (1024 * 1024):.2f} MB")
    return OUTPUT_ZIP


if __name__ == "__main__":
    export_colab_training_data()
