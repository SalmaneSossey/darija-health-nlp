from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    input_text: str
    normalized_text: str
    predicted_specialty: str
    specialty_confidence: float | None
    top_predictions: list[dict[str, float | str]] = Field(default_factory=list)
    urgency: str
    urgency_reason: str
    symptoms: list[str]
    recommendation: str
    disclaimer: str
