from __future__ import annotations

from .preprocessing import normalize_text


SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "headache": ["sda3", "صداع", "وجع الراس", "mal à la tête", "mal a la tete", "headache"],
    "dizziness": ["dwakha", "دوخة", "vertige", "dizziness"],
    "fever": ["skhana", "سخانة", "fièvre", "fievre", "fever"],
    "cough": ["k7a", "كحة", "toux", "cough"],
    "chest_pain": [
        "wje3 f sedri",
        "wja3 f sedri",
        "وجع فصدري",
        "وجع في صدري",
        "ألم في الصدر",
        "الم في الصدر",
        "douleur thoracique",
        "mal à la poitrine",
        "mal a la poitrine",
        "chest pain",
    ],
    "shortness_of_breath": [
        "di9 f nefs",
        "di9 f nefas",
        "ضيق في التنفس",
        "صعوبة في التنفس",
        "difficulté à respirer",
        "difficulte a respirer",
        "shortness of breath",
    ],
    "stomach_pain": [
        "wje3 f kerchi",
        "wja3 f kerchi",
        "وجع فكرشي",
        "وجع في كرشي",
        "ألم في البطن",
        "الم في البطن",
        "mal au ventre",
        "stomach pain",
    ],
    "skin_rash": ["7boub", "haboub", "حبوب", "طفح", "rash", "boutons"],
    "vomiting": ["t9ya", "ترجيع", "قيء", "vomissements", "vomiting"],
    "bleeding": ["nziif", "نزيف", "دم بزاف", "saignement", "bleeding"],
    "loss_of_consciousness": ["t7t lard", "غمي عليا", "فقدان الوعي", "perte de conscience", "loss of consciousness"],
    "pregnancy_bleeding": ["7amla w kayn nziif", "حامل وعندي نزيف", "saignement grossesse", "pregnancy bleeding"],
}


def extract_symptoms(text: object) -> list[str]:
    normalized = normalize_text(text)
    symptoms: list[str] = []
    for symptom, patterns in SYMPTOM_PATTERNS.items():
        if any((pattern_norm := normalize_text(pattern)) and pattern_norm in normalized for pattern in patterns):
            symptoms.append(symptom)
    return symptoms
