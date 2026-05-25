from __future__ import annotations

import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.text_normalization import normalize_text
from src.utils.paths import MODELS_DIR


def predict_specialty(text: str) -> str:
    model = joblib.load(MODELS_DIR / "specialty_classifier.joblib")
    return str(model.predict([normalize_text(text)])[0])


if __name__ == "__main__":
    message = " ".join(sys.argv[1:]) or "3ndi wje3 f sedri w di9 f nefs"
    print(predict_specialty(message))
