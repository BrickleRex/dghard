"""GSM8K — grade-school math word problems (exact numeric match).

Canonical 8-shot CoT prompting with the '####' delimiter convention.
Scoring: take the last signed number in the model's response after
stripping <think> tags, compare against the gold numeric answer.
Ported verbatim from the paper-tree implementation; behavior is identical.
"""
from __future__ import annotations

import random
import re
from typing import Optional

from dghard.eval.benchmarks.base import Benchmark, BenchmarkResult
from dghard.eval.utils import strip_think_tags, to_float


_GSM8K_FEWSHOT = [
    (
        "Q: Natalia sold clips to 48 of her friends in April, and then she sold "
        "half as many clips in May. How many clips did Natalia sell altogether "
        "in April and May?",
        "A: Natalia sold 48/2 = 24 clips in May. Altogether she sold "
        "48+24 = 72 clips. The answer is 72.\n#### 72",
    ),
    (
        "Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
        "minutes of babysitting. How much did she earn?",
        "A: Weng earns 12/60 = $0.2 per minute. Working 50 minutes she earned "
        "0.2 x 50 = $10. The answer is 10.\n#### 10",
    ),
    (
        "Q: Betty is saving money for a new wallet which costs $100. Betty has "
        "only half of the money she needs. Her parents decided to give her $15 "
        "for that purpose, and her grandparents twice as much as her parents. "
        "How much more money does Betty need to buy the wallet?",
        "A: Betty has 100/2 = $50. Her grandparents gave 2*15 = $30. In total "
        "she now has 50+15+30 = $95. She needs 100-95 = $5 more. The answer is 5.\n#### 5",
    ),
    (
        "Q: Julie is reading a 120-page book. Yesterday, she was able to read "
        "12 pages and today, she read twice as many pages as yesterday. If she "
        "wants to read half of the remaining pages tomorrow, how many pages "
        "should she read?",
        "A: Today Julie read 2*12 = 24 pages. So far she read 12+24 = 36 pages. "
        "Pages remaining: 120-36 = 84. Half of that is 84/2 = 42. The answer is 42.\n#### 42",
    ),
    (
        "Q: James writes a 3-page letter to 2 different friends twice a week. "
        "How many pages does he write a year?",
        "A: Each week he writes 3*2*2 = 12 pages. In a year (52 weeks) he writes "
        "12*52 = 624 pages. The answer is 624.\n#### 624",
    ),
    (
        "Q: Mark has a garden with flowers. He planted plants of three different "
        "colors in it. Ten of them are yellow, and there are 80% more of those "
        "in purple. There are only 25% as many green flowers as there are yellow "
        "and purple flowers. How many flowers does Mark have in his garden?",
        "A: Purple = 10 + 80% of 10 = 18. Yellow+purple = 28. Green = 25% of 28 = 7. "
        "Total = 28 + 7 = 35. The answer is 35.\n#### 35",
    ),
    (
        "Q: Albert is wondering how much pizza he can eat in one day. He buys 2 "
        "large pizzas and 2 small pizzas. A large pizza has 16 slices and a small "
        "pizza has 8 slices. If he eats it all, how many pieces does he eat that "
        "day?",
        "A: Large pizzas: 2*16 = 32 slices. Small pizzas: 2*8 = 16 slices. Total "
        "= 32 + 16 = 48. The answer is 48.\n#### 48",
    ),
    (
        "Q: Ken created a care package to send to his brother, who was away at "
        "boarding school. Ken placed a box on a scale, and then he poured into "
        "the box enough jelly beans to bring the weight to 2 pounds. Then, he "
        "added enough brownies to cause the weight to triple. Next, he added "
        "another 2 pounds of jelly beans. And finally, he added enough gummy worms "
        "to double the weight once again. What was the final weight of the box of "
        "goodies, in pounds?",
        "A: After jelly beans: 2. After brownies: 2*3 = 6. After more jelly beans: "
        "6+2 = 8. After gummy worms: 8*2 = 16. The answer is 16.\n#### 16",
    ),
]


def build_gsm8k_prompt(question: str) -> str:
    shots = "\n\n".join(f"{q}\n{a}" for q, a in _GSM8K_FEWSHOT)
    return f"{shots}\n\nQ: {question.strip()}\nA:"


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_answer(text: str) -> Optional[float]:
    if not text:
        return None
    text = strip_think_tags(text)
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return to_float(m.group(1))
    low = text.lower()
    for marker in ("final answer is", "the answer is", "answer is", "answer:"):
        if marker in low:
            idx = low.rindex(marker) + len(marker)
            tail = text[idx:idx + 64]
            m = _NUM_RE.search(tail)
            if m:
                return to_float(m.group(0))
    nums = _NUM_RE.findall(text)
    if nums:
        return to_float(nums[-1])
    return None


class Gsm8k(Benchmark):
    name = "gsm8k"
    metric = "exact_number"

    def load(self, subset_size=None, seed: int = 42, cache_dir=None):
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
        rows = [dict(r) for r in ds]
        if subset_size and subset_size < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, subset_size)
        return rows

    def format_prompt(self, row: dict) -> str:
        return build_gsm8k_prompt(row["question"])

    def score(self, pred: str, row: dict) -> BenchmarkResult:
        gold_text = str(row.get("answer", ""))
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", gold_text)
        gold = to_float(m.group(1)) if m else None
        pred_num = extract_gsm8k_answer(pred)
        correct = 0.0
        if gold is not None and pred_num is not None:
            correct = float(abs(pred_num - gold) < 1e-4)
        return BenchmarkResult(
            correct=correct,
            pred_raw=pred,
            pred_normalized=str(pred_num) if pred_num is not None else "",
            gold=gold,
        )
