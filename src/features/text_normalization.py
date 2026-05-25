from __future__ import annotations

import re
import unicodedata


ARABIC_NORMALIZATION_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
)

ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
REPEATED_PUNCT_RE = re.compile(r"([!?؟.,،؛:]){2,}")
SPACE_RE = re.compile(r"\s+")
NOISE_RE = re.compile(r"[^\w\s\u0600-\u06FF!?؟.,،؛:;'/+-]", re.UNICODE)


def normalize_arabic(text: str) -> str:
    text = text.translate(ARABIC_NORMALIZATION_MAP)
    text = ARABIC_DIACRITICS_RE.sub("", text)
    return text


def normalize_text(text: object) -> str:
    """Normalize lightly while preserving Darija Arabizi digits such as 3, 7, and 9."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.strip().lower()
    normalized = normalize_arabic(normalized)
    normalized = REPEATED_PUNCT_RE.sub(r"\1", normalized)
    normalized = NOISE_RE.sub(" ", normalized)
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized


def has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÿ]", text or ""))
