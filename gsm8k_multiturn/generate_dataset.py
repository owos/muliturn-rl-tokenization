"""
Synthetic multi-turn dataset generation from GSM8K (and optionally MATH).

For each problem, Qwen3-4B decomposes the solution into a sequence of
intermediate sub-questions, producing a CoQA-style multi-turn dataset
suitable for GRPO training.

Output format (JSONL, one record per problem):
  {
    "story":        "<original problem statement>",
    "questions":    ["sub-q1", ..., "final question"],
    "gold_answers": ["val1",   ..., "final answer"]
  }

Usage:
  python generate_dataset.py --split train --out gsm8k_multiturn_train.jsonl
  python generate_dataset.py --split test  --out gsm8k_multiturn_test.jsonl
"""

import re
import json
import argparse
from pathlib import Path

import datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL          = "Qwen/Qwen3-4B"
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.3     # low temp for structured generation
BATCH_SIZE     = 32      # problems per vLLM call

_SEED          = 42      # set via --seed; used in SamplingParams


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a math tutor creating multi-turn tutoring conversations from word problems.

You will be given:
  - A math PROBLEM (the question a student is asked)
  - A SOLUTION that shows the step-by-step reasoning and final answer

Your task is to generate a sequence of 3–6 sub-questions that guide a student \
through the key intermediate values, with the ORIGINAL question as the final turn.

--- STRICT RULES ---

1. Every sub-question must be answerable using ONLY:
   (a) information stated explicitly in the original problem, AND
   (b) answers to earlier sub-questions in the sequence.
   Do NOT write sub-questions whose answer can only be known by reading the solution \
or by knowing the final answer in advance. This prevents answer leakage.

2. Sub-questions must ask about INTERMEDIATE quantities — partial values that a \
student would need to compute on the way to the final answer.

3. Each answer should show the reasoning, then end with a clear final statement \
of the form "The answer is X." \
(e.g. "80% of 10 is 8, so there are 8 more purple flowers than yellow. The answer is 8."). \
This lets a student follow the thinking and unambiguously identify the result.

4. The LAST question must be the original question (word-for-word or very close).

5. The LAST answer must be the final numerical answer from the solution.

6. Return ONLY a valid JSON object. No markdown, no code fences, no commentary.

--- EXAMPLE ---

Problem:
Mark has a garden with flowers. He planted plants of three different colors in it. \
Ten of them are yellow, and there are 80% more of those in purple. \
There are only 25% as many green flowers as there are yellow and purple flowers. \
How many flowers does Mark have in his garden?

Solution:
There are 80/100 * 10 = 8 more purple flowers than yellow flowers.
So in Mark's garden, there are 10 + 8 = 18 purple flowers.
Purple and yellow flowers sum up to 10 + 18 = 28 flowers.
That means in Mark's garden there are 25/100 * 28 = 7 green flowers.
So in total Mark has 28 + 7 = 35 plants in his garden.

Output:
{
  "questions": [
    "How many more purple flowers are there than yellow flowers?",
    "How many purple flowers are in Mark's garden?",
    "How many yellow and purple flowers are there combined?",
    "How many green flowers are in Mark's garden?",
    "How many flowers does Mark have in his garden?"
  ],
  "answers": [
    "8",
    "18",
    "28",
    "7",
    "35"
  ]
}

Note how each sub-question only asks about a value that can be derived from \
what the problem states plus any previously given answers — the student is never \
asked to produce a value that gives away the final answer prematurely.

--- END EXAMPLE ---

JSON schema (your output must match this exactly):
{
  "questions": ["...", "...", "..."],
  "answers":   ["...", "...", "..."]
}
"""

def make_user_prompt(question: str, solution: str) -> str:
    # Strip the <<calc>> annotations vLLM would never see anyway
    clean_solution = re.sub(r"<<[^>]+>>", "", solution).strip()
    return (
        f"Problem:\n{question}\n\n"
        f"Solution:\n{clean_solution}"
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(split: str) -> list[dict]:
    """Returns list of {question, solution, answer} dicts."""
    ds = datasets.load_dataset("openai/gsm8k", "main", trust_remote_code=True)
    records = []
    for row in ds[split]:
        answer_match = re.search(r"####\s*(.+)$", row["answer"], re.MULTILINE)
        final_answer = answer_match.group(1).strip() if answer_match else ""
        records.append({
            "question": row["question"],
            "solution": row["answer"],
            "answer":   final_answer,
        })
    return records


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_chat_prompt(tokenizer, question: str, solution: str) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": make_user_prompt(question, solution)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # Qwen3 non-thinking mode for structured output
    )


def parse_output(text: str) -> dict | None:
    """Extract the JSON object from model output. Returns None on failure."""
    # Try to find a JSON object in the output
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        if (
            isinstance(obj.get("questions"), list)
            and isinstance(obj.get("answers"), list)
            and len(obj["questions"]) == len(obj["answers"])
            and len(obj["questions"]) >= 2
        ):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def generate_multiturn(records: list[dict], tokenizer, llm: LLM) -> list[dict]:
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        stop=["</s>", "<|im_end|>"],
        seed=_SEED,
    )

    results   = []
    failures  = 0

    for i in range(0, len(records), BATCH_SIZE):
        batch   = records[i : i + BATCH_SIZE]
        prompts = [
            build_chat_prompt(tokenizer, r["question"], r["solution"])
            for r in batch
        ]

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for rec, out in zip(batch, outputs):
            text   = out.outputs[0].text
            parsed = parse_output(text)

            if parsed is None:
                failures += 1
                print(f"  [warn] parse failed for: {rec['question'][:60]}...")
                continue

            results.append({
                "story":           rec["question"],
                "questions":       parsed["questions"],
                "gold_answers":    parsed["answers"],
                "original_answer": rec["solution"],
            })

        done = min(i + BATCH_SIZE, len(records))
        print(f"  processed {done}/{len(records)}  (failures so far: {failures})")

    print(f"\nDone. Generated {len(results)}/{len(records)} records ({failures} parse failures).")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    default="train", choices=["train", "test"])
    parser.add_argument("--out",      default="")          # defaults to gsm8k_multiturn_{split}.jsonl
    parser.add_argument("--max",      type=int, default=0, help="cap number of problems (0 = all)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    # vLLM seeds its own RNG via SamplingParams; we pass it through there too
    global _SEED
    _SEED = args.seed

    out_path = args.out or f"gsm8k_multiturn_{args.split}.jsonl"

    print(f"Loading GSM8K ({args.split})...")
    records = load_gsm8k(args.split)
    if args.max:
        records = records[: args.max]
    print(f"  {len(records)} problems loaded.")

    print(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    print(f"Loading vLLM engine: {MODEL}")
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        enforce_eager=False,
        max_model_len=2048,
    )

    print(f"\nGenerating multi-turn decompositions...")
    results = generate_multiturn(records, tokenizer, llm)

    Path(out_path).write_text(
        "\n".join(json.dumps(r) for r in results) + "\n"
    )
    print(f"Saved {len(results)} records → {out_path}")


if __name__ == "__main__":
    main()
