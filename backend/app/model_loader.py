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
    ("bleeding", "Emergency Medicine"),
    ("loss_of_consciousness", "Emergency Medicine"),
    ("skin_rash", "Dermatology"),
    ("stomach_pain", "Gastroenterology"),
    ("headache", "Neurology"),
    ("dizziness", "Neurology"),
    ("cough", "Pulmonology"),
    ("shortness_of_breath", "Pulmonology"),
]
CRITICAL_SYMPTOM_HINTS = {
    "chest_pain",
    "pregnancy_bleeding",
    "bleeding",
    "loss_of_consciousness",
    "shortness_of_breath",
}
GENERAL_SPECIALTIES = {
    "General Practice",
    "Internal Medicine",
    "general practitioner",
}
LOW_CONFIDENCE_THRESHOLD = 0.55


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
        self.transformer_top_k = int(os.getenv("TRANSFORMER_TOP_K", "3"))

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
                top_k=max(1, self.transformer_top_k),
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
    ) -> tuple[str, float | None, list[dict[str, str | float]]]:
        symptom_hint = self._symptom_specialty_hint(symptoms)

        # 1. Evaluate with Transformer (if active)
        if self.is_transformer and self.model is not None:
            outs = self.model(normalized_text)
            ranked = self._normalize_transformer_output(outs)
            if ranked:
                best = ranked[0]
                label = str(best["label"])
                score = float(best["score"])
                label = self._apply_symptom_refinement(label, score, symptoms, symptom_hint)
                return label, score, ranked

        # 2. Evaluate with compiled Rust classical engine (if active)
        if self.rust_classifier is not None:
            prediction = self.rust_classifier.predict(normalized_text)
            return self._apply_symptom_refinement(prediction, None, symptoms, symptom_hint), None, []

        # 3. Evaluate with standard Python Joblib engine (if active)
        if self.model is not None:
            prediction = str(self.model.predict([normalized_text])[0])
            confidence = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba([normalized_text])[0]
                confidence = float(max(probabilities))
            return self._apply_symptom_refinement(prediction, confidence, symptoms, symptom_hint), confidence, []

        return symptom_hint or "General Practice", None, []

    def _symptom_specialty_hint(self, symptoms: list[str]) -> str | None:
        for symptom, specialty in SYMPTOM_SPECIALTY_PRIORITY:
            if symptom in symptoms:
                return specialty
        return None

    def _normalize_transformer_output(self, outs: Any) -> list[dict[str, str | float]]:
        candidates = outs[0] if isinstance(outs, list) and outs and isinstance(outs[0], list) else outs
        if isinstance(candidates, dict):
            candidates = [candidates]
        if not isinstance(candidates, list):
            return []
        ranked = [
            {"label": str(item["label"]), "score": float(item["score"])}
            for item in candidates
            if isinstance(item, dict) and "label" in item and "score" in item
        ]
        return sorted(ranked, key=lambda item: float(item["score"]), reverse=True)

    def _apply_symptom_refinement(
        self,
        predicted: str,
        confidence: float | None,
        symptoms: list[str],
        symptom_hint: str | None,
    ) -> str:
        if symptom_hint is None:
            return predicted
        if not symptoms:
            return predicted

        low_confidence = confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD
        critical_signal = any(symptom in CRITICAL_SYMPTOM_HINTS for symptom in symptoms)
        general_prediction = predicted in GENERAL_SPECIALTIES

        if critical_signal and (low_confidence or general_prediction):
            return symptom_hint
        if low_confidence and predicted in GENERAL_SPECIALTIES:
            return symptom_hint
        return predicted


specialty_model = SpecialtyModel()
