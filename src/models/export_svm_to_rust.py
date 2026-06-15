from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib

from src.utils.paths import MODELS_DIR


def export_to_json() -> None:
    model_path = MODELS_DIR / "specialty_classifier.joblib"
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run specialty training first.")
        return

    pipeline = joblib.load(model_path)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    vocabulary = {str(k): int(v) for k, v in tfidf.vocabulary_.items()}
    idf = tfidf.idf_.tolist()

    classes = [str(c) for c in clf.classes_.tolist()]

    if hasattr(clf, "coef_"):
        coef = clf.coef_
        if hasattr(coef, "toarray"):
            coef = coef.toarray()
        weights = coef.astype(float).tolist()
    else:
        raise ValueError(
            "Classifier does not have coefficients (coef_). Only LinearSVC is supported."
        )

    intercepts = [float(x) for x in clf.intercept_.tolist()]

    data = {
        "vocabulary": vocabulary,
        "idf": idf,
        "classes": classes,
        "weights": weights,
        "intercepts": intercepts,
    }

    output_path = MODELS_DIR / "svm_weights.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Successfully exported SVM weights to {output_path}")


if __name__ == "__main__":
    export_to_json()
