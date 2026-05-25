from __future__ import annotations

from src.features.symptom_patterns import SYMPTOM_PATTERNS
from src.features.text_normalization import normalize_text


def extract_symptoms(text: object) -> list[str]:
    normalized = normalize_text(text)
    found: list[str] = []
    for symptom, patterns in SYMPTOM_PATTERNS.items():
        normalized_patterns = [normalize_text(pattern) for pattern in patterns]
        if any(pattern and pattern in normalized for pattern in normalized_patterns):
            found.append(symptom)
    return found
