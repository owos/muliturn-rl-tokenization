#!/bin/bash
# Generate the GSM8K multi-turn dataset using Qwen3-4B on an A100.
#
# Usage:
#   bash run.sh              # generates train + test
#   bash run.sh --max 200    # quick smoke-test on 200 problems

set -e
cd "$(dirname "$0")"

EXTRA_ARGS="$@"

echo "=== generating train split ==="
uv run python generate_dataset.py --split train --out gsm8k_multiturn_train.jsonl $EXTRA_ARGS

echo "=== generating test split ==="
uv run python generate_dataset.py --split test  --out gsm8k_multiturn_test.jsonl  $EXTRA_ARGS

echo "=== done ==="
wc -l gsm8k_multiturn_*.jsonl
