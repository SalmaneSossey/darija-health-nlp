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
        
        models_dir = Path(__file__).resolve().parents[2] / "models"
        self.joblib_model_path = Path(os.getenv("MODEL_PATH", models_dir / "specialty_classifier.joblib"))
        
        self.transformer_path = Path(os.getenv("TRANSFORMER_PATH", models_dir / "transformer_MARBERT_specialty"))
        
        self.load()

    def load(self) -> None:
        if self.transformer_path.exists() and pipeline is not None:
            import logging
            logging.info(f"Loading MARBERT Transformer model from {self.transformer_path} on CPU")
            # Ensure it works in newer versions of transformers depending on the return defaults
            # device=-1 explicitly forces CPU evaluation
            self.model = pipeline("text-classification", model=str(self.transformer_path), top_k=1, device=-1)
            self.is_transformer = True
        elif self.joblib_model_path.exists():
            import logging
            logging.info(f"Loading baseline Joblib model from {self.joblib_model_path}")
            self.model = joblib.load(self.joblib_model_path)
            self.is_transformer = False

    def predict(self, normalized_text: str, symptoms: list[str]) -> tuple[str, float | None]:
        if symptoms:
            for symptom, specialty in SYMPTOM_SPECIALTY_PRIORITY:
                if symptom in symptoms:
                    return specialty, None
        if self.model is None:
            return "General Practice", None
            
        if self.is_transformer:
            outs = self.model(normalized_text)
            # outs might be [[{'label': 'Cardiology', 'score': 0.95}]] when top_k=1 is specified
            best = outs[0][0] if isinstance(outs[0], list) else outs[0]
            prediction = best["label"]
            confidence = float(best["score"])
        else:
            prediction = str(self.model.predict([normalized_text])[0])
            confidence = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba([normalized_text])[0]
                confidence = float(max(probabilities))
                
        return prediction, confidence


specialty_model = SpecialtyModel()
