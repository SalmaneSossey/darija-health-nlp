from __future__ import annotations

import pandas as pd

from src.models.train_transformer_specialty_classifier import compute_balanced_class_weights, model_slug


def test_model_slug_is_filesystem_friendly() -> None:
    assert model_slug("UBC-NLP/MARBERT") == "UBC-NLP_MARBERT"


def test_balanced_class_weights_raise_rare_classes() -> None:
    weights = compute_balanced_class_weights(pd.Series([0, 0, 0, 1]), num_labels=2)

    assert weights[0] < weights[1]
    assert weights == [4 / (2 * 3), 4 / (2 * 1)]
