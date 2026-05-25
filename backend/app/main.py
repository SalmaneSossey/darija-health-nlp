from __future__ import annotations

from fastapi import FastAPI

from .model_loader import specialty_model
from .preprocessing import normalize_text
from .recommendation import DISCLAIMER, build_recommendation
from .schemas import PredictRequest, PredictResponse
from .symptom_extractor import extract_symptoms
from .urgency_rules import classify_urgency


app = FastAPI(
    title="Darija Health NLP",
    description="Moroccan medical triage orientation API. This API does not diagnose.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    normalized = normalize_text(request.message)
    symptoms = extract_symptoms(normalized)
    specialty, confidence = specialty_model.predict(normalized, symptoms)
    urgency = classify_urgency(normalized, symptoms)
    return PredictResponse(
        input_text=request.message,
        normalized_text=normalized,
        predicted_specialty=specialty,
        specialty_confidence=confidence,
        urgency=urgency["urgency"],
        urgency_reason=urgency["reason"],
        symptoms=symptoms,
        recommendation=build_recommendation(urgency["urgency"], specialty),
        disclaimer=DISCLAIMER,
    )
