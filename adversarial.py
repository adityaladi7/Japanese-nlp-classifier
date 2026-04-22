"""
adversarial.py
--------------
Handles real-world text evasion techniques commonly used to bypass content classifiers:
- Full-width / homoglyph character substitution
- Symbol / punctuation insertion
- Code-switching (Japanese + English mixed abuse)
- Repeated character elongation
- Leet-speak variants
"""

import re
import unicodedata
from typing import Tuple, List, Dict

# ── Homoglyph map: visually similar characters → ASCII equivalent ──
HOMOGLYPH_MAP = {
    # Cyrillic lookalikes
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
    # Greek lookalikes
    "ο": "o", "α": "a", "ε": "e",
    # Common symbol substitutions
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "@": "a", "$": "s", "!": "i",
}

# ── Known abuse terms (sample — extend for production) ──
KNOWN_ABUSE_PATTERNS = [
    r"死[ね亡]",          # death wishes
    r"殺[すし]",          # kill
    r"馬鹿|バカ|ばか",    # idiot/fool
    r"クズ|くず",         # scum
    r"きもい|キモい|気持ち悪い",  # disgusting
    r"うざい|ウザい",     # annoying/irritating
]


def normalize_homoglyphs(text: str) -> str:
    """Replace visually similar characters with their ASCII equivalents."""
    return "".join(HOMOGLYPH_MAP.get(char, char) for char in text)


def normalize_fullwidth(text: str) -> str:
    """Convert full-width Unicode characters to half-width (NFKC normalization)."""
    return unicodedata.normalize("NFKC", text)


def remove_symbol_insertion(text: str) -> str:
    """
    Remove symbols inserted between letters to evade keyword matching.
    e.g. 'a.b.u.s.e' → 'abuse', 'k*i*l*l' → 'kill'
    """
    # Remove punctuation inserted between word characters
    return re.sub(r"(?<=\w)[.·•*_\-|/\\](?=\w)", "", text)


def normalize_elongation(text: str) -> str:
    """
    Collapse repeated characters (elongation for emphasis or evasion).
    e.g. 'ばかああああ' → 'ばかあ', 'dieeeee' → 'diee' (keep 2 for context)
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def detect_code_switching(text: str) -> Dict[str, bool]:
    """
    Detect presence of multiple scripts (signals code-switching).
    Returns dict of script presence flags.
    """
    has_japanese = bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text))
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    has_arabic_numerals = bool(re.search(r"[0-9]", text))
    has_cyrillic = bool(re.search(r"[\u0400-\u04ff]", text))

    return {
        "japanese": has_japanese,
        "latin": has_latin,
        "numerals": has_arabic_numerals,
        "cyrillic": has_cyrillic,
        "code_switching": has_japanese and has_latin,
    }


def detect_abuse_patterns(text: str) -> List[str]:
    """
    Return list of matched known abuse patterns in text.
    """
    matched = []
    for pattern in KNOWN_ABUSE_PATTERNS:
        if re.search(pattern, text):
            matched.append(pattern)
    return matched


def full_adversarial_normalization(text: str) -> Tuple[str, Dict]:
    """
    Apply all adversarial normalization steps.
    Returns (normalized_text, metadata_dict).

    metadata contains flags useful for downstream severity scoring.
    """
    metadata = {
        "original_length": len(text),
        "had_fullwidth": False,
        "had_homoglyphs": False,
        "had_symbol_insertion": False,
        "had_elongation": False,
        "script_info": {},
        "abuse_patterns_found": [],
    }

    # 1. Full-width normalization
    normalized = normalize_fullwidth(text)
    if normalized != text:
        metadata["had_fullwidth"] = True

    # 2. Homoglyph normalization
    step2 = normalize_homoglyphs(normalized)
    if step2 != normalized:
        metadata["had_homoglyphs"] = True
    normalized = step2

    # 3. Symbol insertion removal
    step3 = remove_symbol_insertion(normalized)
    if step3 != normalized:
        metadata["had_symbol_insertion"] = True
    normalized = step3

    # 4. Elongation normalization
    step4 = normalize_elongation(normalized)
    if step4 != normalized:
        metadata["had_elongation"] = True
    normalized = step4

    # 5. Script analysis
    metadata["script_info"] = detect_code_switching(normalized)

    # 6. Known abuse pattern detection
    metadata["abuse_patterns_found"] = detect_abuse_patterns(normalized)

    return normalized, metadata


def adversarial_risk_score(metadata: Dict) -> float:
    """
    Compute a simple adversarial risk score (0.0–1.0) based on evasion signals.
    Higher score = more likely intentional evasion attempt.
    """
    score = 0.0
    if metadata["had_fullwidth"]:
        score += 0.2
    if metadata["had_homoglyphs"]:
        score += 0.3
    if metadata["had_symbol_insertion"]:
        score += 0.25
    if metadata["had_elongation"]:
        score += 0.1
    if metadata["script_info"].get("cyrillic"):
        score += 0.3
    if metadata["abuse_patterns_found"]:
        score += 0.3 * min(len(metadata["abuse_patterns_found"]), 2)
    return min(score, 1.0)


if __name__ == "__main__":
    test_cases = [
        "普通のコメントです。問題ありません。",
        "ａｂｕｓｅ — full width evasion test",
        "k.i.l.l symbol insertion test",
        "死ねえええええ elongation test",
        "This is こんにちは code-switching test",
        "аbuse with Cyrillic 'а'",
        "バカ野郎！うざい！",
    ]

    print("=== Adversarial Normalization Tests ===\n")
    for text in test_cases:
        normalized, meta = full_adversarial_normalization(text)
        risk = adversarial_risk_score(meta)
        print(f"Original:   {text}")
        print(f"Normalized: {normalized}")
        print(f"Risk Score: {risk:.2f}")
        print(f"Flags:      fullwidth={meta['had_fullwidth']}, homoglyph={meta['had_homoglyphs']}, "
              f"symbol={meta['had_symbol_insertion']}, elongation={meta['had_elongation']}")
        print(f"Patterns:   {meta['abuse_patterns_found'] or 'none'}")
        print()
