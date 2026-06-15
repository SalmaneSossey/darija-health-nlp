from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


medqa_folders = list(DATA_DIR.glob("MedQA-MA*"))
if medqa_folders:
    RAW_DATA_DIR = medqa_folders[0]
else:
    RAW_DATA_DIR = DATA_DIR / "raw" / "medqa_ma"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_project_dirs() -> None:

    dirs_to_create = [
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        MODELS_DIR,
    ]
    if not RAW_DATA_DIR.exists():
        dirs_to_create.append(RAW_DATA_DIR)

    for path in dirs_to_create:
        path.mkdir(parents=True, exist_ok=True)
