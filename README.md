# Multi-Turn Tokenization for RL

GRPO training and evaluation studying how tokenization strategy affects multi-turn RL fine-tuning on GSM8K.

## Setup

```bash
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=<your_token>
modal secret create wandb-secret WANDB_API_KEY=<your_key>
```

## Training

```bash
# On Modal (H100)
modal run modal_train.py --config configs/gsm8k_incremental.yaml

# Locally
bash run_train.sh
```

Config files are in `configs/`. **Tokenization strategies:**

| Strategy | Description |
|---|---|
| `single` | No history — only the current question each turn |
| `full` | Rebuild and retokenize the full conversation from scratch each turn |
| `incremental` | Cache previous token IDs; only tokenize the new turn and append |
| `aligned` | Incremental + BPE-boundary alignment against prior assistant output |

## Evaluation

```bash
# On Modal
modal run modal_eval.py --model Qwen/Qwen3-4B --mode single --n 500
modal run modal_eval.py --model Qwen/Qwen3-4B --mode multi --n 200 --strategy incremental

# Locally
bash run_eval.sh
```

Results are downloaded automatically to `results/`.
