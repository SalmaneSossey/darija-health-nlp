from __future__ import annotations

from .preprocessing import normalize_text


HIGH_URGENCY_SYMPTOMS = {
    "chest_pain",
    "shortness_of_breath",
    "bleeding",
    "loss_of_consciousness",
    "pregnancy_bleeding",
}
HIGH_URGENCY_PATTERNS = [
    "douleur thoracique",
    "mal a la poitrine",
    "ألم في الصدر",
    "الم في الصدر",
    "ضيق في التنفس",
    "صعوبة في التنفس",
    "نزيف",
    "فقدان الوعي",
    "غمي عليا",
    "suicide",
    "انتحار",
    "شلل",
    "نص وجهي",
]
MEDIUM_URGENCY_PATTERNS = [
    "ثلاثة أيام",
    "3 jours",
    "depuis 3 jours",
    "vomissements",
    "ترجيع",
    "قيء",
    "سخانة",
    "fièvre",
    "fievre",
    "infection",
    "كيزيد",
]


def classify_urgency(text: str, symptoms: list[str] | None = None) -> dict[str, str]:
    symptoms = symptoms or []
    normalized = normalize_text(text)
    symptom_set = set(symptoms)
    if symptom_set & HIGH_URGENCY_SYMPTOMS or any(normalize_text(pattern) in normalized for pattern in HIGH_URGENCY_PATTERNS):
        return {
            "urgency": "high",
            "reason": "Possible chest pain, breathing difficulty, bleeding, loss of consciousness, or another emergency sign detected.",
        }
    if {"fever", "cough"} <= symptom_set or "vomiting" in symptom_set or any(
        normalize_text(pattern) in normalized for pattern in MEDIUM_URGENCY_PATTERNS
    ):
        return {
            "urgency": "medium",
            "reason": "Symptoms may need timely medical advice, especially if they persist or worsen.",
        }
    if symptom_set:
        return {
            "urgency": "low",
            "reason": "No emergency warning sign was detected from the available text.",
        }
    return {
        "urgency": "unknown",
        "reason": "No clear urgency signal was detected. A clinician should assess persistent or concerning symptoms.",
    }
