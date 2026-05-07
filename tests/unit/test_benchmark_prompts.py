"""Snapshot tests for benchmark prompt construction.

Each test feeds a canned dataset row (no `datasets.load_dataset`) and
asserts the prompt string contains the expected scaffolding so any
future drift in the prompt template will fail the test loudly.
"""
from __future__ import annotations

import re

from dghard.eval.benchmarks.gsm8k import build_gsm8k_prompt
from dghard.eval.benchmarks.mmlu import build_mmlu_prompt
from dghard.eval.benchmarks.arc_challenge import build_arc_prompt
from dghard.eval.benchmarks.hellaswag import build_hellaswag_prompt
from dghard.eval.benchmarks.truthful_qa import build_truthfulqa_prompt
from dghard.eval.benchmarks.math_500 import build_math_prompt
from dghard.eval.benchmarks.mnli import build_mnli_prompt
from dghard.eval.benchmarks.trivia_qa import build_trivia_prompt


def test_gsm8k_prompt_has_8_fewshot_examples():
    prompt = build_gsm8k_prompt("Train at 60 mph for 2 hours; how far?")
    assert prompt.count("\nQ: ") == 8       # 8 few-shot Q markers + the query
    assert prompt.count("\n#### ") == 8     # 8 few-shot answer markers
    assert prompt.endswith("\nA:")
    assert "Train at 60 mph" in prompt


def test_mmlu_prompt_includes_subject_and_choices():
    eval_row = {
        "question": "What's the speed of light?",
        "choices": ["3e8 m/s", "3e6 m/s", "3e9 m/s", "1.5e8 m/s"],
        "answer": 0,
        "subject": "high_school_physics",
    }
    fewshot = [
        {"question": "1+1?", "choices": ["1", "2", "3", "4"], "answer": 1,
         "subject": "high_school_physics"},
    ]
    prompt = build_mmlu_prompt(eval_row, fewshot)
    assert "high school physics" in prompt
    assert "Answer with the letter A, B, C, or D" in prompt
    assert "Answer: B" in prompt          # few-shot answer baked in
    assert prompt.endswith("Answer:")     # eval row has no answer


def test_arc_prompt_keeps_dataset_label_letters():
    """ARC may have 4 (A-D) or 5 (A-E) choices; preserve raw labels."""
    row = {
        "question": "What gas do plants take in?",
        "choices": {"label": ["A", "B", "C", "D"],
                    "text": ["O2", "CO2", "N2", "He"]},
        "answerKey": "B",
    }
    prompt = build_arc_prompt(row, fewshot_rows=[])
    assert "A. O2" in prompt
    assert "B. CO2" in prompt
    assert prompt.endswith("Answer:")


def test_hellaswag_prompt_handles_split_context():
    row = {"ctx_a": "She walked", "ctx_b": "into the kitchen.",
           "endings": ["She left.", "She cooked.", "She slept.", "She drove."],
           "activity_label": "Cooking"}
    prompt = build_hellaswag_prompt(row)
    assert "She walked into the kitchen." in prompt
    assert "Activity: Cooking" in prompt
    assert "A. She left." in prompt
    assert "Answer:" in prompt


def test_truthfulqa_prompt_lists_all_choices():
    p = build_truthfulqa_prompt("Is the earth flat?",
                                ["Yes", "No", "Unknown", "Other"])
    assert "A. Yes" in p
    assert "D. Other" in p
    assert p.endswith("Answer:")


def test_math500_prompt_demands_boxed_answer():
    p = build_math_prompt("Compute 2+2.")
    assert "\\boxed{...}" in p
    assert "Compute 2+2." in p


def test_mnli_prompt_lists_three_labels():
    p = build_mnli_prompt("It rained.", "The ground is wet.")
    assert "A. entailment" in p and "B. neutral" in p and "C. contradiction" in p
    assert "Premise: It rained." in p
    assert "Hypothesis: The ground is wet." in p
    assert p.endswith("Answer:")


def test_trivia_prompt_asks_for_short_answer():
    p = build_trivia_prompt("Capital of France?")
    assert "Answer:" in p
    assert "Capital of France?" in p
