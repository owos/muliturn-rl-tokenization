"""
CoQA evaluation — multi-turn, token-F1 metric.

Usage:
  python eval_coqa.py --models Qwen/Qwen3-4B
  python eval_coqa.py --models experiments/coqa_grpo_out --strategy incremental --n 200
"""

import re
import json
import argparse
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from tokenization import make_turn_tokenizer
from utils import token_f1, strip_think


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_coqa_val(n: int = 0) -> list[dict]:
    import datasets
    raw = datasets.load_dataset("stanfordnlp/coqa")
    records = []
    for row in raw["validation"]:
        gold_answers = row["answers"]["input_text"]
        if any(a.strip().lower() == "unknown" for a in gold_answers):
            continue
        records.append({
            "story":        row["story"],
            "questions":    row["questions"],
            "gold_answers": gold_answers,
        })
    return records[:n] if n else records


# ---------------------------------------------------------------------------
# Multi-turn evaluation
# ---------------------------------------------------------------------------

def eval_coqa(
    llm: LLM,
    tokenizer,
    records: list[dict],
    tokenization_strategy: str,
    think: bool = True,
    max_ctx_len: int = 8192,
) -> dict:
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=4096 if think else 512,
        repetition_penalty=1.0,
        stop=["</s>", "<|im_end|>"],
    )
    # One tokenizer state per episode (incremental/aligned strategies are stateful)
    turn_toks      = [make_turn_tokenizer(tokenization_strategy, tokenizer) for _ in records]
    answers_so_far = [[] for _ in records]
    ep_results     = [[] for _ in records]
    turn_f1s       = {}   # turn_idx → list[float]

    max_turns = max(len(r["questions"]) for r in records)
    active    = list(range(len(records)))

    for turn_idx in range(max_turns):
        # Build one batched prompt for every active episode that has a question at this turn
        eligible, prompts = [], []
        for i in active:
            if turn_idx >= len(records[i]["questions"]):
                continue
            ids = turn_toks[i].encode_turn(
                records[i]["story"],
                records[i]["questions"][: turn_idx + 1],
                answers_so_far[i],
                tokenizer,
                max_ctx_len,
            )
            prompts.append({"prompt_token_ids": ids})
            eligible.append(i)

        if not eligible:
            break

        # Single batched vLLM call for all active episodes at this turn
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        next_active = []
        for i, out in zip(eligible, outputs):
            gen  = strip_think(out.outputs[0].text).split("\n")[0].strip()
            turn_toks[i].notify_completion(list(out.outputs[0].token_ids))
            gold = records[i]["gold_answers"][turn_idx]
            f1   = token_f1(gen, gold)
            turn_f1s.setdefault(turn_idx, []).append(f1)
            ep_results[i].append({"t": turn_idx, "q": records[i]["questions"][turn_idx],
                                   "gold": gold, "gen": gen, "f1": f1})
            answers_so_far[i].append(gen)
            if turn_idx + 1 < len(records[i]["questions"]):
                next_active.append(i)

        all_f1s = [f for fs in turn_f1s.values() for f in fs]
        print(f"  turn {turn_idx}  batch={len(eligible)}  mean_f1={sum(all_f1s)/len(all_f1s):.3f}", flush=True)
        active = next_active

    # Per-episode summaries printed after all turns
    for i, ep in enumerate(ep_results):
        if not ep:
            continue
        ep_mean = sum(r["f1"] for r in ep) / len(ep)
        print(f"\n--- episode {i}  mean_f1={ep_mean:.3f} ---")
        for r in ep:
            print(f"  turn {r['t']}  f1={r['f1']:.2f}  Q: {r['q']}")
            print(f"             gold={r['gold']!r}  gen={r['gen']!r}")

    per_turn_f1 = {f"turn_{t}_f1": sum(v) / len(v) for t, v in sorted(turn_f1s.items())}
    all_vals    = [f for v in turn_f1s.values() for f in v]
    mean_f1     = sum(all_vals) / len(all_vals)

    return {"mean_f1": mean_f1, "n": len(records), **per_turn_f1, "results": ep_results}


# ---------------------------------------------------------------------------
# Main entry point (shared with modal_eval.py)
# ---------------------------------------------------------------------------

def run_eval(model_path: str, n: int, tokenization_strategy: str, seed: int = 42, think: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"Model        : {model_path}")
    print(f"Dataset      : CoQA (validation)")
    print(f"MT strategy  : {tokenization_strategy}")
    print(f"Seed         : {seed}")
    print(f"Think        : {think}")
    print(f"{'='*60}")

    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=False,
        enable_prefix_caching=False,
        seed=seed,
    )
    if not think:
        llm_kwargs["override_generation_config"] = {"enable_thinking": False}
    llm = LLM(**llm_kwargs)

    records = load_coqa_val(n)
    print(f"\n[multi-turn CoQA]  n={len(records)}  strategy={tokenization_strategy}")
    result = eval_coqa(llm, tokenizer, records, tokenization_strategy, think=think)
    print(f"  mean_f1: {result['mean_f1']:.4f}")

    del llm
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tokenization_strategy", default="incremental",
                        choices=["single", "incremental", "full", "aligned"])
    parser.add_argument("--n",    type=int, default=0, help="Cap eval stories (0=all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  default="eval_coqa_results.json")
    args = parser.parse_args()

    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    all_results = {}
    for model_path in args.models:
        all_results[model_path] = run_eval(model_path, args.n, args.tokenization_strategy)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, res in all_results.items():
        print(f"  model={Path(model).name}  mean_f1={res['mean_f1']:.4f}")

    for model_path, res in all_results.items():
        model_name = Path(model_path).name
        stem   = Path(args.out).stem
        suffix = Path(args.out).suffix
        out_path = Path(args.out).parent / f"{stem}_{model_name}{suffix}"
        out_path.write_text(json.dumps(res, indent=2))
        print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
