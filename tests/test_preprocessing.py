from backend.app.preprocessing import normalize_text


def test_arabic_normalization() -> None:
    assert normalize_text("ألم   في  الصدر!!!") == "الم في الصدر!"


def test_preserves_arabizi_digits() -> None:
    assert "3ndi" in normalize_text("3ndi sda3")
