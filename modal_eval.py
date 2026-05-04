"""
Modal evaluation script for GSM8K and CoQA.

Setup (one-time):
    pip install modal
    modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_xxx

Run:
    modal run modal_eval.py
    modal run modal_eval.py --dataset gsm8k --model Qwen/Qwen3-4B --mode single --n 200 --strategy single
    modal run modal_eval.py --dataset coqa  --model Qwen/Qwen3-4B --n 200 --strategy incremental
    modal run modal_eval.py --dataset gsm8k --model experiments/gsm8k_tok_single_20260424_180246 --strategy single
    modal run modal_eval.py --detach   # fire-and-forget

    #meta-llama/Llama-3.2-3B-Instruct
"""

import modal
from pathlib import Path

CODE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Container image
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
        "datasets",
        "tqdm",
        "numpy",
    )
    .env({
        "PYTHONUNBUFFERED":               "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_USE_DEEP_GEMM":             "0",
        "CC":  "gcc",
        "CXX": "g++",
    })
    .add_local_dir(CODE_DIR / "gsm8k_multiturn", remote_path="/app/gsm8k_multiturn")
    .add_local_python_source("eval_gsm8k", "eval_coqa", "tokenization", "utils")
)

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
hf_vol  = modal.Volume.from_name("hf-cache",        create_if_missing=True)
exp_vol = modal.Volume.from_name("grpo-experiments", create_if_missing=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("gsm8k-eval", image=image)


@app.function(
    gpu="H200",
    timeout=60 * 60 * 4,  # 4 hours
    volumes={
        "/hf_cache":    hf_vol,
        "/experiments": exp_vol,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def evaluate(
    dataset:  str = "gsm8k",        # "gsm8k" | "coqa"
    model:    str = "Qwen/Qwen3-4B",
    mode:     str = "multi",        # gsm8k only: "single" | "multi" | "both"
    n:        int = 200,
    strategy: str = "incremental",
    seed:     int = 42,
    think:    bool = True,          # False → disable thinking for Qwen3-style models
) -> str:
    import os, sys, json
    from datetime import datetime
    from pathlib import Path as P

    os.environ["HF_HOME"]            = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"]  = "/hf_cache/datasets"
    os.environ["TRANSFORMERS_CACHE"] = "/hf_cache/hub"

    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Map experiments/ paths to the mounted volume; HF hub IDs pass through unchanged
    if model.startswith("experiments/"):
        model_path = f"/{model}"
    elif model.startswith("/"):
        model_path = model
    else:
        model_path = model

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name      = P(model_path).name

    # /experiments is the volume mount point; results live under results/ within the volume
    out_file = f"/experiments/results/eval_{dataset}_{name}_{strategy}_{mode}_{timestamp}.json"
    P(out_file).parent.mkdir(parents=True, exist_ok=True)

    result = None
    try:
        if dataset == "coqa":
            import eval_coqa
            result = eval_coqa.run_eval(model_path, n, strategy, seed=seed, think=think)
        else:
            import eval_gsm8k
            eval_gsm8k.GSM8K_DATA = "/app/gsm8k_multiturn/gsm8k_multiturn_test.jsonl"
            result = eval_gsm8k.run_eval(model_path, mode, n, strategy, seed=seed)
    finally:
        if result is not None:
            P(out_file).write_text(json.dumps(result, indent=2))
            exp_vol.commit()
            print(f"\nResults saved → {out_file}")

    return out_file


@app.local_entrypoint()
def main(
    dataset:  str = "gsm8k",
    model:    str = "Qwen/Qwen3-0.6B-Base",
    mode:     str = "multi",
    n:        int = 200,
    strategy: str = "incremental",
    seed:     int = 42,
    think:    bool = False,
):
    import subprocess

    out_path   = evaluate.remote(dataset=dataset, model=model, mode=mode, n=n, strategy=strategy, seed=seed, think=think)
    local_name = Path(out_path).name

    # modal volume get expects a path relative to the volume root,
    # not the container mount path (/experiments/results/... → /results/...)
    vol_path = out_path.removeprefix("/experiments")

    print(f"\nDownloading {vol_path} → ./{local_name} ...")
    subprocess.run(
        ["modal", "volume", "get", "grpo-experiments", vol_path, "results/"],
        check=True,
    )
    print(f"Saved to ./{local_name}")
