#!/bin/bash
#SBATCH --account PAS2836
#SBATCH --job-name grpo-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80gb
#SBATCH --cluster=ascend
#SBATCH --mail-type=END,FAIL

cat $0
nvidia-smi

export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/PAS2836/${USER}/triton_cache
export HF_HOME=/fs/scratch/PAS2836/${USER}/.cache/huggingface
export HF_DATASETS_CACHE=/fs/scratch/PAS2836/${USER}/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/fs/scratch/PAS2836/${USER}/.cache/huggingface/hub
export HOME=/fs/scratch/PAS2836/${USER}

cd /fs/scratch/PAS2836/owos/experiments/RL_MTT

export HF_TOKEN=${HF_TOKEN:?HF_TOKEN not set}
export PYTHONUNBUFFERED=1
# ---------------------------------------------------------------------------
# Models to evaluate — model path → multi-turn tokenization strategy
# (single-turn eval is always the same; strategy only affects multi-turn eval)
# ---------------------------------------------------------------------------
declare -A models=(
    # [path]=strategy
    ["Qwen/Qwen3-0.6B-Base"]="incremental"
    ["Qwen/Qwen3-4B"]="incremental"
    # ["experiments/gsm8k_tok_single_20260424_180246"]="single"
    # ['experiments/gsm8k_tok_incremental_20260425_200041']="incremental"
    # ['experiments/gsm8k_tok_full_20260425_093252']="full"
    # ['/fs/scratch/PAS2836/owos/experiments/RL_MTT/experiments/gsm8k_grpo_out']="aligned"

    # 20260408 runs (old prompt format — kept for reference)
    # ["/fs/scratch/PAS2836/owos/experiments/RL_MTT/experiments/gsm8k_single_grpo_out_20260403_033931"]="single"
    # ["experiments/gsm8k_tok_full_20260408_215936"]="full"
    # ["experiments/gsm8k_tok_incremental_20260408_013005"]="incremental"
    # ["experiments/gsm8k_tok_aligned_20260408_013005"]="aligned"

    # 20260413 runs
    # ["experiments/gsm8k_tok_single_20260413_010403"]="incremental"
    ["experiments/gsm8k_tok_incremental_20260413_010325"]="incremental"
    ["experiments/gsm8k_tok_full_20260413_010319"]="full"
    ["experiments/gsm8k_tok_aligned_20260413_010301"]="aligned"

    # ["meta-llama/Llama-3.2-3B-Instruct"]="incremental"
    # ['/fs/scratch/PAS2836/owos/experiments/RL_MTT/experiments/gsm8k_tok_single_20260415_161339']="single"
    # ['experiments/gsm8k_tok_single_20260416_052132']="single"
    # ['Qwen/Qwen2.5-1.5B-Instruct']="single"
    # ['Qwen/Qwen2-1.5B-Instruct']="single"
)

# ---------------------------------------------------------------------------
# Eval settings
# ---------------------------------------------------------------------------
MODE="multi"   # single | multi | both
N=200           # number of test examples (0 = all)
# ---------------------------------------------------------------------------

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p results

for model in "${!models[@]}"; do
    strategy="${models[$model]}"
    name=$(basename "$model")
    out="results/eval_gsm8k_${name}_${TIMESTAMP}.json"

    uv run python eval_gsm8k.py \
        --models "$model" \
        --mode   "$MODE" \
        --n      "$N" \
        --out    "$out" \
        --tokenization_strategy "$strategy"

    echo "Results: $out"
done
