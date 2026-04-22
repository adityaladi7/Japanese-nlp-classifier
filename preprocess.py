"""
preprocess.py
-------------
Japanese text preprocessing pipeline:
- Unicode normalization (full-width → half-width)
- MeCab tokenization (falls back to character splitting if MeCab unavailable)
- Stopword removal
- Lemmatization
"""

import unicodedata
import re
import pandas as pd
from typing import List, Optional

# Try importing MeCab; fall back gracefully
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    print("[WARNING] MeCab not found. Falling back to character-level tokenization.")


# ── Japanese stopwords (common particles, auxiliaries) ──
JAPANESE_STOPWORDS = {
    "は", "が", "を", "に", "で", "と", "も", "の", "な", "へ",
    "から", "まで", "より", "て", "で", "し", "た", "だ", "です",
    "ます", "ある", "いる", "する", "なる", "れる", "られる", "せる",
    "ない", "ぬ", "ず", "か", "や", "ね", "よ", "わ", "さ", "こ",
    "そ", "あ", "どこ", "これ", "それ", "あれ", "この", "その"
}


def normalize_unicode(text: str) -> str:
    """
    Normalize full-width characters to half-width (catches adversarial substitution).
    e.g. 'ａｂｃ' → 'abc', '１２３' → '123'
    """
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def remove_excessive_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_mecab(text: str) -> List[str]:
    """Tokenize Japanese text using MeCab morphological analyzer."""
    tagger = MeCab.Tagger("-Owakati")
    result = tagger.parse(text)
    tokens = result.strip().split()
    return tokens


def tokenize_fallback(text: str) -> List[str]:
    """
    Character-level fallback tokenizer when MeCab is unavailable.
    Splits on whitespace and individual CJK characters.
    """
    tokens = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":   # CJK Unified Ideographs
            tokens.append(char)
        elif "\u3040" <= char <= "\u309f":  # Hiragana
            tokens.append(char)
        elif "\u30a0" <= char <= "\u30ff":  # Katakana
            tokens.append(char)
        else:
            tokens.append(char)
    return [t.strip() for t in tokens if t.strip()]


def tokenize(text: str) -> List[str]:
    if MECAB_AVAILABLE:
        return tokenize_mecab(text)
    return tokenize_fallback(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in JAPANESE_STOPWORDS]


def preprocess_text(text: str, remove_stops: bool = True) -> str:
    """
    Full preprocessing pipeline for a single text string.
    Returns a cleaned, tokenized string ready for feature extraction.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_excessive_whitespace(text)

    tokens = tokenize(text)

    if remove_stops:
        tokens = remove_stopwords(tokens)

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame column.
    Adds a 'processed_text' column.
    """
    print(f"[preprocess] Processing {len(df)} records...")
    df = df.copy()
    df["processed_text"] = df[text_col].apply(preprocess_text)
    # Drop rows where preprocessing produced empty strings
    before = len(df)
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)
    print(f"[preprocess] Dropped {before - len(df)} empty rows. Remaining: {len(df)}")
    return df


if __name__ == "__main__":
    # Quick smoke test
    samples = [
        "このアプリは最悪です！絶対に使わないでください。",
        "ａｂｕｓｅ unicode full-width test ａｂｃ",
        "普通のレビューです。特に問題はありませんでした。",
        "<b>HTML tag test</b> https://example.com",
    ]
    print("=== Preprocessing Smoke Test ===")
    for s in samples:
        result = preprocess_text(s)
        print(f"IN:  {s}")
        print(f"OUT: {result}")
        print()
