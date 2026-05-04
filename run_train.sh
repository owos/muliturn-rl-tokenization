#!/bin/bash
#SBATCH --account PAS2836   
#SBATCH --job-name **single 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=05:30:00
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
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN not set}
export PYTHONUNBUFFERED=1


# cd /fs/scratch/PAS2836/owos/experiments/RL_MTT

# ---------------------------------------------------------------------------
# Select config — pass as argument or set CONFIG here:
#   sbatch run_train.sh configs/gsm8k_full.yaml
# ---------------------------------------------------------------------------
# CONFIG=${1:-configs/gsm8k_full.yaml}

# CONFIG=${1:-configs/gsm8k_incremental.yaml}
# CONFIG=${1:-configs/gsm8k_aligned.yaml}
# CONFIG=${1:-configs/gsm8k_single.yaml}
CONFIG=${1:-configs/coqa_full.yaml}


uv run python grpo_coqa.py --config "$CONFIG"
