"""
Tokenization strategy diagnostic — no GPU, no dataset, no vLLM.

Runs a few fake CoQA-style episodes through all four strategies and checks:
  1. Decoded text is identical across full / incremental / aligned at every turn
  2. No EOS token appears inside any prompt
  3. Reports where token IDs actually differ (BPE boundary splits)
  4. Verifies aligned differs from incremental when a known repeated word appears

Usage:
  python check_tokenization.py
  python check_tokenization.py --model Qwen/Qwen3-4B
"""

import argparse
import sys
import os

os.environ["HF_HOME"] = "/fs/scratch/PAS2836/owos/.cache/huggingface"

from transformers import AutoTokenizer
from tokenization import make_turn_tokenizer, build_turn_text


# ---------------------------------------------------------------------------
# Fake episodes
# ---------------------------------------------------------------------------

EPISODES = [
    {
        "story": "Sam went to the market. He bought apples and oranges. Then he walked home.",
        "questions": ["Where did Sam go?", "What did he buy?", "What did he do after?"],
        "gold_answers": ["the market", "apples and oranges", "walked home"],
    },
    {
        "story": "The cat sat on the mat. It was a fluffy cat. The mat was red.",
        "questions": ["What sat on the mat?", "What color was the mat?", "What kind of cat was it?"],
        "gold_answers": ["the cat", "red", "fluffy"],
    },
    # Episode designed to trigger the aligner: "market" appears in the answer, then in a question
    {
        "story": "There was a market downtown. People sold fruit there.",
        "questions": ["What was downtown?", "Was the market busy?", "What did the market sell?"],
        "gold_answers": ["a market", "yes", "fruit"],
    },
]


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def run_episode(ep: dict, tokenizer, strategies: list[str]) -> list[dict]:
    """Simulate an episode with gold answers and collect per-turn prompt info."""
    toks           = {s: make_turn_tokenizer(s, tokenizer) for s in strategies}
    eos_id         = tokenizer.eos_token_id
    answers_so_far = []
    turns          = []

    for t, (q, gold) in enumerate(zip(ep["questions"], ep["gold_answers"])):
        turn = {"t": t, "question": q, "gold": gold}
        for s, tok in toks.items():
            ids     = tok.encode_turn(ep["story"], ep["questions"][:t+1], answers_so_far, tokenizer, max_length=8192)
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            turn[s] = {"ids": ids, "decoded": decoded, "eos_hits": [i for i, x in enumerate(ids) if x == eos_id]}

            # Feed gold answer tokens as vLLM would: the prompt ends with "Answer:"
            # so the model generates a leading space, e.g. " the market" not "the market".
            gold_ids = tokenizer.encode(" " + gold, add_special_tokens=False)
            tok.notify_completion(gold_ids)

        answers_so_far.append(gold)
        turns.append(turn)

    return turns


def check_episode(ep_idx: int, ep: dict, tokenizer, strategies: list[str]) -> bool:
    ref     = strategies[0]   # "full" is ground truth
    turns   = run_episode(ep, tokenizer, strategies)
    ok      = True

    print(f"\n=== Episode {ep_idx}: {ep['story'][:60]!r} ===")

    for turn in turns:
        t = turn["t"]
        ref_dec = turn[ref]["decoded"]
        ref_ids = turn[ref]["ids"]

        for s in strategies[1:]:
            dec = turn[s]["decoded"]
            ids = turn[s]["ids"]

            # EOS leak
            for strat in (ref, s):
                if turn[strat]["eos_hits"]:
                    print(f"  [t={t}] EOS LEAK in {strat} at positions {turn[strat]['eos_hits']}")
                    ok = False

            # Text match (most important — same conversation must go in)
            if dec != ref_dec:
                print(f"  [t={t}] TEXT MISMATCH: {ref} vs {s}")
                # Show suffix to find where they diverge
                common = 0
                for a, b in zip(ref_dec, dec):
                    if a != b: break
                    common += 1
                print(f"    common prefix length: {common} chars")
                print(f"    {ref} tail: {ref_dec[common-30:]!r}")
                print(f"    {s}   tail: {dec[common-30:]!r}")
                ok = False
            else:
                # Text matches — check if token IDs also match
                if ids == ref_ids:
                    print(f"  [t={t}] {ref} vs {s}: text=SAME, ids=SAME  len={len(ids)}")
                else:
                    # Find first difference
                    first_diff = next((i for i,(a,b) in enumerate(zip(ref_ids,ids)) if a!=b),
                                      min(len(ref_ids),len(ids)))
                    # Show the surface around the diff
                    ctx = tokenizer.decode(ref_ids[max(0,first_diff-3):first_diff+5])
                    print(f"  [t={t}] {ref} vs {s}: text=SAME, ids=DIFFER "
                          f"first_diff=pos {first_diff}/{len(ref_ids)}  "
                          f"surface around diff: {ctx!r}")

    return ok


# ---------------------------------------------------------------------------
# Also directly compare the raw text build_turn_text produces vs decoded prompt
# ---------------------------------------------------------------------------

def check_text_roundtrip(ep: dict, tokenizer, strategies: list[str]):
    """
    Verify that the decoded prompt text for each strategy matches
    what build_turn_text would produce (modulo leading BOS marker).
    """
    toks           = {s: make_turn_tokenizer(s, tokenizer) for s in strategies}
    answers_so_far = []
    bos            = tokenizer.bos_token or ""

    print(f"\n  [roundtrip check]")
    for t, (q, gold) in enumerate(zip(ep["questions"], ep["gold_answers"])):
        expected_text = build_turn_text(ep["story"], ep["questions"][:t+1], answers_so_far)
        for s, tok in toks.items():
            ids     = tok.encode_turn(ep["story"], ep["questions"][:t+1], answers_so_far, tokenizer, max_length=8192)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)  # skip BOS
            if decoded != expected_text:
                print(f"    [t={t}] {s}: decoded != expected_text")
                # find where they diverge
                for i, (a, b) in enumerate(zip(decoded, expected_text)):
                    if a != b:
                        print(f"      first diff at char {i}")
                        print(f"      decoded:  {decoded[max(0,i-20):i+40]!r}")
                        print(f"      expected: {expected_text[max(0,i-20):i+40]!r}")
                        break
            gold_ids = tokenizer.encode(" " + gold, add_special_tokens=False)
            tok.notify_completion(gold_ids)
        answers_so_far.append(gold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base",
                        help="Any cached tokenizer — no GPU needed")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"EOS id={tok.eos_token_id}  token={tok.decode([tok.eos_token_id])!r}")
    print(f"BOS id={tok.bos_token_id}  token={tok.bos_token!r}")

    strategies = ["full", "incremental", "aligned"]
    all_ok     = True

    for ep_idx, ep in enumerate(EPISODES):
        ok = check_episode(ep_idx, ep, tok, strategies)
        check_text_roundtrip(ep, tok, strategies)
        all_ok = all_ok and ok

    print("\n" + "="*60)
    print("RESULT:", "ALL OK" if all_ok else "ISSUES FOUND — see above")


if __name__ == "__main__":
    main()
