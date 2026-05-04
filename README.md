# Multi-Turn Tokenization for RL

GRPO training and evaluation studying how tokenization strategy affects multi-turn RL fine-tuning on GSM8K and CoQA.

## Setup

```bash
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=<your_token>
modal secret create wandb-secret WANDB_API_KEY=<your_key>
```

## Training

Training runs on Modal (H100). Config files are in `configs/`.

```bash
# GSM8K — pick a tokenization strategy
modal run modal_train.py --config configs/gsm8k_incremental.yaml
modal run modal_train.py --config configs/gsm8k_full.yaml
modal run modal_train.py --config configs/gsm8k_aligned.yaml
modal run modal_train.py --config configs/gsm8k_single.yaml

# CoQA
modal run modal_train.py --config configs/coqa_incremental.yaml
```

**Tokenization strategies:**

| Strategy | Description |
|---|---|
| `single` | No history — only the current question each turn |
| `full` | Rebuild and retokenize the full conversation from scratch each turn |
| `incremental` | Cache previous token IDs; only tokenize the new turn and append |
| `aligned` | Incremental + BPE-boundary alignment against prior assistant output |

## Evaluation

Evaluation also runs on Modal and saves results to a persistent volume.

```bash
# GSM8K single-turn
modal run modal_eval.py --dataset gsm8k --model Qwen/Qwen3-4B --mode single --n 500

# GSM8K multi-turn
modal run modal_eval.py --dataset gsm8k --model Qwen/Qwen3-4B --mode multi --n 200 --strategy incremental

# CoQA multi-turn
modal run modal_eval.py --dataset coqa --model Qwen/Qwen3-4B --n 200 --strategy incremental

# Evaluate a trained checkpoint from the volume
modal run modal_eval.py --model experiments/gsm8k_tok_incremental_<timestamp> --strategy incremental
```

Results are downloaded automatically to `results/`.

## Local evaluation (no Modal)

```bash
python eval_gsm8k.py --models Qwen/Qwen3-4B --mode single
python eval_gsm8k.py --models Qwen/Qwen3-4B --mode multi --tokenization_strategy incremental --n 200
```
