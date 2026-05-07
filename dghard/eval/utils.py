"""Scoring utilities shared across benchmarks (extracted verbatim from the
paper-tree implementation so per-benchmark scoring stays bit-for-bit
identical between this release and the paper's eval pipeline)."""
from __future__ import annotations

import re
from typing import Optional


_THINK_TAG = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_ONLY = re.compile(r"<think\b[^>]*>.*$", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (including unclosed ones).

    Reasoning models leak their CoT — strip it so scoring sees only the
    final answer.
    """
    if not text:
        return ""
    text = _THINK_TAG.sub("", text)
    text = _THINK_OPEN_ONLY.sub("", text)
    return text.strip()


def normalize_label(text: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_YES_WORDS = {"yes", "true", "entailment", "entails"}
_NO_WORDS = {"no", "false", "contradiction", "neutral"}
_NO_PHRASES = ("not entailment", "does not entail", "doesn't entail",
               "not entail", "does not follow")


def _split_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text)


def extract_yes_no(text: str) -> Optional[str]:
    """Return 'yes' or 'no'. Handles entailment-style labels and reasoning
    preludes."""
    if not text:
        return None
    text = strip_think_tags(text)
    t = text.lower()
    for phrase in _NO_PHRASES:
        if phrase in t:
            return "no"

    def _first_hit(scan_text: str) -> Optional[str]:
        for token in _split_words(scan_text):
            if token in _YES_WORDS:
                return "yes"
            if token in _NO_WORDS:
                return "no"
        return None

    for marker in ("final answer:", "answer:", "answer is", "answer -"):
        if marker in t:
            tail = t.split(marker, 1)[1][:64]
            hit = _first_hit(tail)
            if hit:
                return hit
    hit = _first_hit(t)
    if hit:
        return hit
    n = normalize_label(text)
    if n in _YES_WORDS:
        return "yes"
    if n in _NO_WORDS:
        return "no"
    return None


def extract_letter(text: str, choices: str = "ABCD") -> Optional[str]:
    """Extract a multiple-choice letter (A-D by default).

    Tries marker-led extraction, then standalone letter tokens, then the
    first character of the response.
    """
    if not text:
        return None
    text = strip_think_tags(text).strip()
    letters = set(choices.upper())
    low = text.lower()
    for marker in ("final answer:", "answer:", "answer is", "option is", "the answer"):
        if marker in low:
            idx = low.index(marker) + len(marker)
            tail = text[idx:idx + 16]
            for ch in tail.upper():
                if ch in letters:
                    return ch
    pattern = r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])"
    if any(c not in "ABCD" for c in choices.upper()):
        # Generalize when choices include letters beyond A-D.
        pattern = (r"(?<![A-Za-z0-9])([" + re.escape(choices.upper()) + r"])"
                   r"(?![A-Za-z0-9])")
    for m in re.finditer(pattern, text.upper()):
        if m.group(1) in letters:
            return m.group(1)
    if text and text[0].upper() in letters:
        return text[0].upper()
    return None


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def to_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def squad_normalize(text: str) -> list[str]:
    """SQuAD canonical token normalization: lowercase, strip articles + punct."""
    if not text:
        return []
    text = strip_think_tags(text)
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def squad_f1_em(pred: str, gold_answers: list[str]) -> tuple[float, float]:
    """SQuAD-style token F1 and EM, taking the max over all gold variants."""
    if not gold_answers:
        return 0.0, 0.0
    pred_tokens = squad_normalize(pred)
    if not pred_tokens:
        return 0.0, 0.0
    from collections import Counter

    best_f1, best_em = 0.0, 0.0
    for g in gold_answers:
        gold_tokens = squad_normalize(g)
        if not gold_tokens:
            continue
        em = float(pred_tokens == gold_tokens)
        ca, cb = Counter(pred_tokens), Counter(gold_tokens)
        common_n = sum(min(n, cb.get(tok, 0)) for tok, n in ca.items())
        if common_n == 0:
            f1 = 0.0
        else:
            precision = common_n / len(pred_tokens)
            recall = common_n / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
        best_em = max(best_em, em)
    return best_f1, best_em
