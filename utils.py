import re
import string
from collections import Counter
from typing import Callable


def strip_think(text: str) -> str:
    """Strip <think>...</think> block from thinking models (e.g. Qwen3-4B).
    Returns only the text after </think>, or the full text if no block is present."""
    end = text.rfind("</think>")
    return text[end + len("</think>"):].strip() if end != -1 else text


def _normalize_answer(s: str) -> list[str]:
    """Lowercase, strip punctuation, remove articles, split into tokens."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    tokens = s.split()
    articles = {"a", "an", "the"}
    return [t for t in tokens if t not in articles]


def token_f1(pred: str, gold: str) -> float:
    p, g   = _normalize_answer(pred), _normalize_answer(gold)
    common = Counter(p) & Counter(g)   # multiset intersection
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    prec = num_common / len(p) if p else 0.0
    rec  = num_common / len(g) if g else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def extract_number(s: str) -> str | None:
    """Return the last number (int or float, with optional commas) found in s."""
    matches = re.findall(r"-?[\d,]+\.?\d*", s)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


def numeric_reward(pred: str, gold: str) -> float:
    """1.0 if the last number in pred matches the last number in gold, else 0.0."""
    g = extract_number(gold)
    p = extract_number(pred)
    if g is None or p is None:
        return 0.0
    return 1.0 if p == g else 0.0


# ---------------------------------------------------------------------------
# Reward aggregators
# ---------------------------------------------------------------------------

def compute_reward(
    gen_answers:  list[str],
    gold_answers: list[str],
    reward_fn:    Callable[[str, str], float] = token_f1,
) -> float:
    """Mean per-turn reward over all turns."""
    n = min(len(gen_answers), len(gold_answers))
    if n == 0:
        return 0.0
    return sum(reward_fn(g, r) for g, r in zip(gen_answers[:n], gold_answers[:n])) / n
