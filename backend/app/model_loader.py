from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

SYMPTOM_SPECIALTY_PRIORITY = [
    ("chest_pain", "Cardiology"),
    ("pregnancy_bleeding", "Obstetrics and Gynecology"),
    ("skin_rash", "Dermatology"),
    ("stomach_pain", "Gastroenterology"),
    ("headache", "Neurology"),
    ("dizziness", "Neurology"),
    ("fever", "General Practice"),
    ("cough", "Pulmonology"),
    ("shortness_of_breath", "Pulmonology"),
]


class SpecialtyModel:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.is_transformer = False
        self.rust_classifier: Any | None = None

        models_dir = Path(__file__).resolve().parents[2] / "models"
        self.joblib_model_path = Path(
            os.getenv("MODEL_PATH", models_dir / "specialty_classifier.joblib")
        )
        self.json_model_path = Path(
            os.getenv("JSON_MODEL_PATH", models_dir / "svm_weights.json")
        )
        self.transformer_path = Path(
            os.getenv(
                "TRANSFORMER_PATH", models_dir / "transformer_MARBERT_specialty"
            )
        )

        self.load()

    def load(self) -> None:
        # Tier 1: Load Transformer model if files are present and transformer dependency exists
        if self.transformer_path.exists() and pipeline is not None:
            import logging

            logging.info(
                f"Loading MARBERT Transformer model from {self.transformer_path} on CPU"
            )
            self.model = pipeline(
                "text-classification",
                model=str(self.transformer_path),
                top_k=1,
                device=-1,
            )
            self.is_transformer = True
        else:
            # Tier 2: Try to import and run Rust compiled SVM
            try:
                from rust_inference import RustClassifier

                if self.json_model_path.exists():
                    import logging

                    logging.info(
                        f"Loading optimized Rust-SVM classifier from {self.json_model_path}"
                    )
                    self.rust_classifier = RustClassifier(
                        str(self.json_model_path)
                    )
                else:
                    import logging

                    logging.warning(
                        f"Rust module loaded, but JSON weights not found at {self.json_model_path}"
                    )
            except ImportError:
                pass

            # Tier 3: Standard fallback to pure Python/Joblib model
            if self.rust_classifier is None and self.joblib_model_path.exists():
                import logging

                logging.info(
                    f"Loading baseline Joblib model from {self.joblib_model_path}"
                )
                self.model = joblib.load(self.joblib_model_path)
                self.is_transformer = False

    def predict(
        self, normalized_text: str, symptoms: list[str]
    ) -> tuple[str, float | None]:
        if symptoms:
            for symptom, specialty in SYMPTOM_SPECIALTY_PRIORITY:
                if symptom in symptoms:
                    return specialty, None

        # 1. Evaluate with Transformer (if active)
        if self.is_transformer and self.model is not None:
            outs = self.model(normalized_text)
            best = outs[0][0] if isinstance(outs[0], list) else outs[0]
            return best["label"], float(best["score"])

        # 2. Evaluate with compiled Rust classical engine (if active)
        if self.rust_classifier is not None:
            prediction = self.rust_classifier.predict(normalized_text)
            return prediction, None

        # 3. Evaluate with standard Python Joblib engine (if active)
        if self.model is not None:
            prediction = str(self.model.predict([normalized_text])[0])
            confidence = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba([normalized_text])[0]
                confidence = float(max(probabilities))
            return prediction, confidence

        return "General Practice", None


specialty_model = SpecialtyModel()
