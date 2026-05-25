from backend.app.symptom_extractor import extract_symptoms
from backend.app.urgency_rules import classify_urgency


def test_extract_chest_pain() -> None:
    symptoms = extract_symptoms("3ndi wje3 f sedri")
    assert "chest_pain" in symptoms


def test_high_urgency_breathing() -> None:
    text = "عندي ضيق في التنفس"
    symptoms = extract_symptoms(text)
    result = classify_urgency(text, symptoms)
    assert result["urgency"] == "high"
