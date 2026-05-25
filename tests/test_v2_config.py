from src.data.create_custom_triage_examples import create_custom_examples
from src.models.specialty_groups import map_to_broad_specialty, merge_rare_specialties
from src.models.train_specialty_classifier import build_models


def test_custom_examples_v2_scale() -> None:
    df = create_custom_examples()
    assert 500 <= len(df) <= 1000
    assert {"latin_darija", "french", "mixed", "arabic_darija"} <= set(df["language"])


def test_char_ngram_model_available() -> None:
    models = build_models()
    assert "char_wb_tfidf_linear_svm" in models


def test_specialty_grouping() -> None:
    counts = {"Emergency Medicine": 10, "Cardiology": 100}
    assert merge_rare_specialties("Emergency Medicine", counts, min_count=50) == "Emergency and Urgent Care"
    assert map_to_broad_specialty("Cardiology") == "Cardiopulmonary"
