from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib


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
        self.model_path = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[2] / "models" / "specialty_classifier.joblib"))
        self.load()

    def load(self) -> None:
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def predict(self, normalized_text: str, symptoms: list[str]) -> tuple[str, float | None]:
        if symptoms:
            for symptom, specialty in SYMPTOM_SPECIALTY_PRIORITY:
                if symptom in symptoms:
                    return specialty, None
        if self.model is None:
            return "General Practice", None
        prediction = str(self.model.predict([normalized_text])[0])
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba([normalized_text])[0]
            confidence = float(max(probabilities))
        return prediction, confidence


specialty_model = SpecialtyModel()
