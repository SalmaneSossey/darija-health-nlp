from __future__ import annotations

import pandas as pd

from src.data import write_dataset_manifest


def write_split(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_dataset_manifest_reports_counts_and_leakage(tmp_path, monkeypatch) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    base_rows = [
        {
            "id": "row_1",
            "text": "3ndi wje3 f sedri",
            "language": "latin_darija",
            "specialty": "Cardiology",
            "urgency": "high",
            "symptoms": "chest_pain",
            "source": "medqa_ma",
        },
        {
            "id": "row_2",
            "text": "haboub f wejhi",
            "language": "latin_darija",
            "specialty": "Dermatology",
            "urgency": "low",
            "symptoms": "skin_rash",
            "source": "custom",
        },
    ]
    write_split(processed / "train.csv", base_rows)
    write_split(processed / "valid.csv", [{**base_rows[0], "id": "row_3", "text": "sda3 qwi"}])
    write_split(processed / "test.csv", [{**base_rows[1], "id": "row_4", "text": "k7a w skhana"}])
    monkeypatch.setattr(write_dataset_manifest, "PROCESSED_DATA_DIR", processed)

    manifest = write_dataset_manifest.build_manifest()

    assert manifest["randomSeed"] == 42
    assert manifest["splits"]["train"]["rows"] == 2
    assert manifest["splits"]["train"]["labelDistribution"] == {
        "Cardiology": 1,
        "Dermatology": 1,
    }
    assert manifest["leakageChecks"]["train_vs_valid"] == {
        "textOverlap": 0,
        "idOverlap": 0,
    }


def test_dataset_manifest_fails_when_splits_are_missing(tmp_path, monkeypatch) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    monkeypatch.setattr(write_dataset_manifest, "PROCESSED_DATA_DIR", processed)

    try:
        write_dataset_manifest.build_manifest()
    except FileNotFoundError as exc:
        assert "Missing processed split files" in str(exc)
    else:
        raise AssertionError("Expected missing splits to fail loudly.")
