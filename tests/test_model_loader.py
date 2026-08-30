from backend.app.model_loader import SpecialtyModel


def make_unloaded_model() -> SpecialtyModel:
    model = SpecialtyModel.__new__(SpecialtyModel)
    model.model = None
    model.is_transformer = False
    model.rust_classifier = None
    model.transformer_top_k = 3
    return model


def test_transformer_output_is_normalized_to_top_predictions() -> None:
    model = make_unloaded_model()

    ranked = model._normalize_transformer_output(
        [[
            {"label": "General Practice", "score": 0.20},
            {"label": "Cardiology", "score": 0.87},
        ]]
    )

    assert ranked == [
        {"label": "Cardiology", "score": 0.87},
        {"label": "General Practice", "score": 0.20},
    ]


def test_symptoms_do_not_override_high_confidence_specific_transformer_prediction() -> None:
    model = make_unloaded_model()

    refined = model._apply_symptom_refinement(
        "Pediatric Medicine",
        0.91,
        ["fever", "cough"],
        "Pulmonology",
    )

    assert refined == "Pediatric Medicine"


def test_critical_symptom_refines_low_confidence_general_prediction() -> None:
    model = make_unloaded_model()

    refined = model._apply_symptom_refinement(
        "General Practice",
        0.42,
        ["chest_pain"],
        "Cardiology",
    )

    assert refined == "Cardiology"
