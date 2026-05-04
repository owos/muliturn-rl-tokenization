"""
Modal training script for GRPO on GSM8K.

Setup (one-time):
    pip install modal
    modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
    modal secret create wandb-secret WANDB_API_KEY=xxx      # optional

Run:
    modal run modal_train.py
    modal run modal_train.py --config configs/gsm8k_full.yaml
    modal run modal_train.py --config configs/gsm8k_single.yaml --detach
"""

import modal
from pathlib import Path

CODE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Container image — pip deps + code baked in
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("gcc", "g++")
    .pip_install(
        "torch==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "vllm",
        "transformers",
        "bitsandbytes",
        "wandb",
        "datasets",
        "tqdm",
        "peft",
        "pyyaml",
        "accelerate",
        "numpy",
    )
    .env({
        "PYTHONUNBUFFERED":               "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "CC":  "gcc",
        "CXX": "g++",
    })
    # add_local_* must come last — Modal injects these at container startup,
    # keeping the cached image layer stable across code changes.
    .add_local_dir(CODE_DIR / "configs",        remote_path="/app/configs")
    .add_local_dir(CODE_DIR / "gsm8k_multiturn", remote_path="/app/gsm8k_multiturn")
    .add_local_python_source("grpo_coqa", "tokenization", "utils")
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
hf_vol  = modal.Volume.from_name("hf-cache",        create_if_missing=True)
exp_vol = modal.Volume.from_name("grpo-experiments", create_if_missing=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("grpo-training", image=image)


@app.function(
    gpu="H200",
    timeout=60 * 60 * 24,  # 24 hours
    volumes={
        "/hf_cache":    hf_vol,
        "/experiments": exp_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def train(config: str = "configs/gsm8k_single.yaml"):
    import os, sys, dataclasses, yaml

    os.environ["HF_HOME"]            = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"]  = "/hf_cache/datasets"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache/hub"
    # WANDB_API_KEY is injected via modal.Secret.from_name("wandb-secret")

    sys.path.insert(0, "/app")
    os.chdir("/app")

    from grpo_coqa import GRPOConfig, train as run_training

    cfg_dict = {f.name: f.default for f in dataclasses.fields(GRPOConfig)}
    with open(config) as fh:
        cfg_dict.update(yaml.safe_load(fh))

    # Redirect output to the persistent experiments volume
    rel = cfg_dict.get("output_dir", "").lstrip("./").removeprefix("experiments/")
    cfg_dict["output_dir"] = f"/experiments/{rel or cfg_dict['dataset']}"

    cfg_dict["gsm8k_data_dir"] = "/app/gsm8k_multiturn"

    cfg = GRPOConfig(**cfg_dict)
    run_training(cfg)       # mutates cfg.output_dir to include the timestamp
    exp_vol.commit()
    return cfg.output_dir   # e.g. /experiments/gsm8k_tok_single_20260417_143022


@app.local_entrypoint()
def main(config: str = "configs/gsm8k_single.yaml"):
    import subprocess

    output_dir = train.remote(config=config)

    local_name = Path(output_dir).name
    print(f"\nDownloading {output_dir} → ./{local_name} ...")
    subprocess.run(
        ["modal", "volume", "get", "grpo-experiments", output_dir, local_name],
        check=True,
    )
    print(f"Saved to ./{local_name}")
