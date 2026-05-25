from __future__ import annotations


BROAD_SPECIALTY_MAP: dict[str, str] = {
    "Allergy and Immunology": "Medicine",
    "Anesthesiology": "Medicine",
    "Cardiology": "Cardiopulmonary",
    "Cosmetic Dermatology": "Dermatology",
    "Dentistry": "Dentistry",
    "Dermatology": "Dermatology",
    "Dietetics and Nutrition": "General and Preventive Care",
    "Emergency Medicine": "Emergency and Urgent Care",
    "Endocrinology": "Medicine",
    "Gastroenterology": "Medicine",
    "General Practice": "General and Preventive Care",
    "Hematology": "Medicine",
    "Infectious Diseases": "Medicine",
    "Internal Medicine": "Medicine",
    "Mental Health": "Mental Health",
    "Neurology": "Neurology and Pain",
    "Obstetrics and Gynecology": "Women and Child Health",
    "Oncology": "Medicine",
    "Ophthalmology": "Eye and ENT",
    "Otorhinolaryngology": "Eye and ENT",
    "Pediatric Medicine": "Women and Child Health",
    "Psychiatry": "Mental Health",
    "Pulmonology": "Cardiopulmonary",
    "Rheumatology and Orthopedics": "Neurology and Pain",
}


def merge_rare_specialties(labels: object, class_counts: dict[str, int], min_count: int = 50) -> str:
    label = str(labels).strip()
    if class_counts.get(label, 0) >= min_count:
        return label
    return BROAD_SPECIALTY_MAP.get(label, "Other")


def map_to_broad_specialty(label: object) -> str:
    label_text = str(label).strip()
    return BROAD_SPECIALTY_MAP.get(label_text, label_text)
