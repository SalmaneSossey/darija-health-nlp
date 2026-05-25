from __future__ import annotations


SYMPTOM_PATTERNS: dict[str, list[str]] = {
    "headache": [
        "sda3",
        "صداع",
        "وجع الراس",
        "mal à la tête",
        "mal a la tete",
        "headache",
    ],
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
    "loss_of_consciousness": [
        "t7t lard",
        "غمي عليا",
        "فقدان الوعي",
        "perte de conscience",
        "loss of consciousness",
    ],
    "pregnancy_bleeding": [
        "7amla w kayn nziif",
        "حامل وعندي نزيف",
        "saignement grossesse",
        "pregnancy bleeding",
    ],
}


SYMPTOM_TO_SPECIALTY: dict[str, str] = {
    "headache": "Neurology",
    "dizziness": "Neurology",
    "fever": "General Practice",
    "cough": "Pulmonology",
    "chest_pain": "Cardiology",
    "shortness_of_breath": "Pulmonology",
    "stomach_pain": "Gastroenterology",
    "skin_rash": "Dermatology",
    "vomiting": "Gastroenterology",
    "bleeding": "Emergency Medicine",
    "loss_of_consciousness": "Emergency Medicine",
    "pregnancy_bleeding": "Obstetrics and Gynecology",
}
