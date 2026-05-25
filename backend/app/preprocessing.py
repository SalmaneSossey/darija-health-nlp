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


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).strip().lower()
    normalized = normalized.translate(ARABIC_NORMALIZATION_MAP)
    normalized = ARABIC_DIACRITICS_RE.sub("", normalized)
    normalized = REPEATED_PUNCT_RE.sub(r"\1", normalized)
    normalized = NOISE_RE.sub(" ", normalized)
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized
