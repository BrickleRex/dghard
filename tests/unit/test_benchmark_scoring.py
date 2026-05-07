"""Snapshot tests for per-benchmark scoring.

Hand-crafted (pred, row) pairs cover the success path, the wrong-answer
path, and the most common edge cases (think tags, marker phrasing, ...).
"""
from __future__ import annotations

from dghard.eval.benchmarks.gsm8k import Gsm8k, extract_gsm8k_answer
from dghard.eval.benchmarks.mmlu import Mmlu
from dghard.eval.benchmarks.arc_challenge import ArcChallenge
from dghard.eval.benchmarks.hellaswag import HellaSwag
from dghard.eval.benchmarks.truthful_qa import TruthfulQa
from dghard.eval.benchmarks.math_500 import Math500, extract_math_answer
from dghard.eval.benchmarks.mnli import Mnli
from dghard.eval.benchmarks.trivia_qa import TriviaQa


# ---- gsm8k ----------------------------------------------------------------

def test_gsm8k_extract_answer_uses_marker_first():
    assert extract_gsm8k_answer("Working it out... #### 42") == 42.0
    assert extract_gsm8k_answer("The answer is 99") == 99.0
    assert extract_gsm8k_answer("first 1, then 2, finally 3") == 3.0


def test_gsm8k_extract_strips_think_tags():
    txt = "<think>3+4=7 plus 2=9</think>The answer is 9."
    assert extract_gsm8k_answer(txt) == 9.0


def test_gsm8k_score_correct_within_tolerance():
    bench = Gsm8k()
    row = {"answer": "stuff\n#### 42"}
    res = bench.score("The answer is 42.", row)
    assert res.correct == 1.0


def test_gsm8k_score_wrong_returns_zero():
    bench = Gsm8k()
    row = {"answer": "stuff\n#### 42"}
    res = bench.score("The answer is 99.", row)
    assert res.correct == 0.0


# ---- mmlu -----------------------------------------------------------------

def test_mmlu_score_correct_letter():
    bench = Mmlu()
    row = {"question": "Q", "choices": ["a", "b", "c", "d"], "answer": 2,
           "subject": "physics"}
    res = bench.score("Final answer: C", row)
    assert res.correct == 1.0
    assert res.gold == "C"


def test_mmlu_score_handles_first_letter_fallback():
    bench = Mmlu()
    row = {"question": "Q", "choices": ["a", "b", "c", "d"], "answer": 0,
           "subject": "physics"}
    res = bench.score("A. is correct", row)
    assert res.correct == 1.0


# ---- arc -----------------------------------------------------------------

def test_arc_score_normalizes_digit_answer_key():
    bench = ArcChallenge()
    row = {"question": "Q",
           "choices": {"label": ["A", "B", "C", "D"],
                       "text": ["w", "x", "y", "z"]},
           "answerKey": "2"}    # legacy digit
    res = bench.score("Answer: B", row)
    assert res.correct == 1.0
    assert res.gold == "B"


# ---- hellaswag -----------------------------------------------------------

def test_hellaswag_score_correct():
    bench = HellaSwag()
    row = {"label": "1", "ctx": "x", "endings": ["a", "b", "c", "d"]}
    res = bench.score("Answer: B", row)
    assert res.correct == 1.0


# ---- truthful_qa --------------------------------------------------------

def test_truthful_qa_finds_correct_label_index():
    bench = TruthfulQa()
    row = {
        "question": "Q",
        "mc1_targets": {
            "choices": ["wrong1", "right", "wrong2", "wrong3"],
            "labels": [0, 1, 0, 0],
        },
    }
    res = bench.score("Answer: B", row)
    assert res.correct == 1.0
    assert res.gold == "B"


# ---- math_500 -----------------------------------------------------------

def test_math500_extract_boxed_handles_nested_braces():
    txt = r"After steps: \boxed{\frac{1}{2}}"
    assert extract_math_answer(txt) == r"\frac{1}{2}"


def test_math500_extract_boxed_uses_last_occurrence():
    txt = r"\boxed{first} ... \boxed{42}"
    assert extract_math_answer(txt) == "42"


def test_math500_score_canonicalizes_both_sides():
    bench = Math500()
    row = {"answer": "1/2"}
    res = bench.score(r"\boxed{ 1/2 }", row)
    assert res.correct == 1.0


# ---- mnli ---------------------------------------------------------------

def test_mnli_score_letter_on_final_line():
    bench = Mnli()
    row = {"premise": "p", "hypothesis": "h", "label": 1}  # neutral
    res = bench.score("Reasoning...\nAnswer: B", row)
    assert res.correct == 1.0
    assert res.gold == "neutral"


def test_mnli_score_word_match_anywhere():
    bench = Mnli()
    row = {"premise": "p", "hypothesis": "h", "label": 0}  # entailment
    res = bench.score("This is a clear case of entailment.", row)
    assert res.correct == 1.0


# ---- trivia_qa ---------------------------------------------------------

def test_trivia_qa_matches_alias():
    bench = TriviaQa()
    row = {"answer": {"value": "Paris", "normalized_value": "paris",
                       "aliases": ["Paris, France"], "normalized_aliases": []}}
    res = bench.score("Answer: Paris", row)
    assert res.correct == 1.0


def test_trivia_qa_squad_normalization_strips_articles():
    bench = TriviaQa()
    row = {"answer": {"value": "the moon", "normalized_value": "moon",
                       "aliases": [], "normalized_aliases": []}}
    res = bench.score("Answer: The Moon!", row)
    assert res.correct == 1.0
