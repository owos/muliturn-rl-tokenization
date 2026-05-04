"""
GSM8K evaluation — single-turn and multi-turn.

Single-turn: standard GSM8K prompt, model generates full solution,
             we extract the final number and check exact match.

Multi-turn:  use the pre-generated gsm8k_multiturn_test.jsonl,
             feed questions one at a time using the CoQA-style context,
             report per-turn accuracy and final-turn accuracy.

Usage:
  # compare fine-tuned vs base on both modes
  python eval_gsm8k.py --models gsm8k_grpo_out Qwen/Qwen3-0.6B-Base

  # specific checkpoint
  python eval_gsm8k.py --models gsm8k_grpo_out/checkpoint-500 Qwen/Qwen3-0.6B-Base

  # only one mode
  python eval_gsm8k.py --models gsm8k_grpo_out --mode single
  python eval_gsm8k.py --models gsm8k_grpo_out --mode multi
"""

import re
import json
import argparse
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from tokenization import make_turn_tokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GSM8K_DATA = "/fs/scratch/PAS2836/owos/experiments/RL_MTT/gsm8k_multiturn/gsm8k_multiturn_test.jsonl"

SINGLE_TURN_PROMPT = "Only give short responses:\nContext: {question}\nQuestion: What is the answer?\nAssistant:"


def extract_number(s: str) -> str | None:
    """Return the last number found in s (strips commas and trailing periods)."""
    matches = re.findall(r"-?[\d,]+\.?\d*", s)
    return matches[-1].replace(",", "").rstrip(".") if matches else None


def strip_final_question_from_story(rec: dict) -> dict:
    """Remove the last sentence if it ends with '?' — it's already asked as the final turn."""
    story = rec["story"]
    sentences = re.split(r'(?<=[.!?])\s+', story.strip())
    if sentences and sentences[-1].rstrip().endswith("?"):
        story = " ".join(sentences[:-1]).strip()
    return {**rec, "story": story}


def load_multiturn_test(path: str) -> list[dict]:
    records = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [strip_final_question_from_story(r) for r in records]


def load_gsm8k_test() -> list[dict]:
    import datasets
    ds = datasets.load_dataset("openai/gsm8k", "main", trust_remote_code=True)
    records = []
    for row in ds["test"]:
        m = re.search(r"####\s*(.+)$", row["answer"], re.MULTILINE)
        records.append({
            "question": row["question"],
            "answer":   m.group(1).strip().replace(",", "") if m else "",
        })
    return records


# ---------------------------------------------------------------------------
# Single-turn evaluation
# ---------------------------------------------------------------------------

def eval_single_turn(llm: LLM, records: list[dict], batch_size: int = 32) -> dict:
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
        repetition_penalty=1.0,
        stop=["</s>", "<|im_end|>"],
    )
    correct = 0
    results = []

    for i in range(0, len(records), batch_size):
        batch   = records[i : i + batch_size]
        prompts = [SINGLE_TURN_PROMPT.format(question=r["question"]) for r in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for rec, out in zip(batch, outputs):
            gen  = out.outputs[0].text
            pred = extract_number(gen)
            gold = rec["answer"]
            hit  = (pred == gold)

            # print(gen, gold)
            # breakpoint()

            correct += int(hit)
            results.append({"question": rec["question"], "gold": gold, "pred": pred, "gen": gen, "correct": hit})

        done = min(i + batch_size, len(records))
        print(f"  single-turn: {done}/{len(records)}  acc={correct/done:.3f}", flush=True)

    print()
    return {"accuracy": correct / len(records), "n": len(records), "results": results}


# ---------------------------------------------------------------------------
# Multi-turn evaluation
# ---------------------------------------------------------------------------

def eval_multi_turn(
    llm: LLM,
    tokenizer,
    records: list[dict],
    tokenization_strategy: str,
    max_ctx_len: int = 8192,
) -> dict:
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=512,
        repetition_penalty=1.0,
        stop=["</s>", "<|im_end|>"],
    )
    turn_toks      = [make_turn_tokenizer(tokenization_strategy, tokenizer) for _ in records]
    answers_so_far = [[] for _ in records]
    ep_results     = [[] for _ in records]
    turn_hits      = {}

    max_turns = max(len(r["questions"]) for r in records)
    active    = list(range(len(records)))

    for turn_idx in range(max_turns):
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

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        next_active = []


        for i, out in zip(eligible, outputs):
            gen = out.outputs[0].text.strip().split("\n")[0].strip()
            clean_ids = tokenizer.encode(gen, add_special_tokens=False)
            turn_toks[i].notify_completion(clean_ids)
            gold = records[i]["gold_answers"][turn_idx]
            pred = extract_number(gen)
            gold_num = extract_number(gold)
            hit = (pred is not None and gold_num is not None and pred == gold_num)
            turn_hits.setdefault(turn_idx, []).append(int(hit))
            ep_results[i].append({"t": turn_idx, "q": records[i]["questions"][turn_idx],
                                   "gold": gold, "gen": gen, "correct": hit})
            answers_so_far[i].append(gen)
            if turn_idx + 1 < len(records[i]["questions"]):
                next_active.append(i)

        hits_so_far = [v for vs in turn_hits.values() for v in vs]
        print(f"  turn {turn_idx}  batch={len(eligible)}  acc={sum(hits_so_far)/len(hits_so_far):.3f}", flush=True)
        active = next_active


    final_hits = []
    for i, ep in enumerate(ep_results):
        if not ep:
            continue
        final_correct = ep[-1]["correct"]
        final_hits.append(int(final_correct))
        print(f"\n--- episode {i}  {'CORRECT' if final_correct else 'WRONG'} ---")
        for r in ep:
            status = "OK" if r["correct"] else "XX"
            print(f"  [{status}] turn {r['t']}  Q: {r['q']}")
            print(f"             gold={r['gold']!r}  pred={extract_number(r['gen'])!r}  gen={r['gen']!r}")

    print()
    per_turn_acc = {f"turn_{t}_acc": sum(v) / len(v) for t, v in sorted(turn_hits.items())}
    final_acc    = sum(final_hits) / len(final_hits)

    return {
        "final_accuracy": final_acc,
        "n": len(records),
        **per_turn_acc,
        "results": ep_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(model_path: str, mode: str, n: int, tokenization_strategy: str,  seed: int = 42,) -> dict:
    print(f"\n{'='*60}")
    print(f"Model        : {model_path}")
    print(f"Mode         : {mode}")
    print(f"MT strategy  : {tokenization_strategy}")
    print(f"Seed         : {seed}")
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
        seed=seed,
        enable_prefix_caching=False,
    )
    llm = LLM(**llm_kwargs)

    out = {}

    if mode in ("single", "both"):
        records = load_gsm8k_test()
        if n:
            records = records[:n]
        print(f"\n[single-turn]  n={len(records)}")
        out["single"] = eval_single_turn(llm, records)
        print(f"  accuracy: {out['single']['accuracy']:.4f}")

    if mode in ("multi", "both"):
        records = load_multiturn_test(GSM8K_DATA)
        if n:
            records = records[:n]
        print(f"\n[multi-turn]  n={len(records)}  strategy={tokenization_strategy}")
        out["multi"] = eval_multi_turn(llm, tokenizer, records, tokenization_strategy)
        print(f"  final_accuracy: {out['multi']['final_accuracy']:.4f}")
        for k, v in out["multi"].items():
            if k.startswith("turn_") and k.endswith("_acc"):
                print(f"  {k}: {v:.4f}")

    del llm
    torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+", required=True,
                        help="Model paths to evaluate (HF hub ID or local dir)")
    parser.add_argument("--mode",    default="both", choices=["single", "multi", "both"])
    parser.add_argument("--tokenization_strategy", default="full",
                        choices=["single","incremental", "full", "aligned"],
                        help="Multi-turn tokenization strategy matching training. "
                             "Use 'incremental' for single-turn trained models.")
    parser.add_argument("--n",       type=int, default=0, help="Cap number of test examples (0=all)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out",     default="eval_results.json")
    args = parser.parse_args()

    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    all_results = {}
    for model_path in args.models:
        all_results[model_path] = run_eval(model_path, args.mode, args.n, args.tokenization_strategy)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, res in all_results.items():
        name = Path(model).name
        parts = [f"model={name}"]
        if "single" in res:
            parts.append(f"single={res['single']['accuracy']:.4f}")
        if "multi" in res:
            parts.append(f"multi_final={res['multi']['final_accuracy']:.4f}")
        print("  " + "  |  ".join(parts))

    for model_path, res in all_results.items():
        model_name = Path(model_path).name
        stem = Path(args.out).stem
        suffix = Path(args.out).suffix
        out_path = Path(args.out).parent / f"{stem}_{model_name}{suffix}"
        out_path.write_text(json.dumps(res, indent=2))
        print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
