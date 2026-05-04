"""
GRPO training on CoQA — full manual control over forward pass and tokenization.

No GRPOTrainer.  We implement the algorithm directly:

  Per batch of stories:
    1. Rollout: generate G completions per episode, turn-by-turn
    2. Score:   compute reward (mean token-F1) for every completion
    3. Advantage: group-relative normalization within each story's G rollouts
    4. Recompute log-probs with a fresh forward pass (this is what you control)
    5. GRPO loss = clipped surrogate + optional KL vs reference model
    6. Optimizer step

GRPO loss reference: DeepSeekMath (arxiv 2402.03300), eq. 4
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import os
import wandb
import bitsandbytes as bnb
import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import token_f1, numeric_reward, compute_reward
from tokenization import TurnTokenizer, make_turn_tokenizer

# Disable vLLM multiprocessing so model_executor is accessible in-process
# (required for weight sync after each optimizer step)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    model_name:         str   = "allenai/OLMo-2-0425-1B-SFT" #"Qwen/Qwen3-0.6B-Base"
    output_dir:         str   = ""   # auto-set to ./{dataset}_grpo_out_{timestamp} if empty
    seed:               int   = 42

    # rollout
    num_generations:    int   = 4      # G: completions per story
    max_turns:          int   = 5      # max conversation turns per story
    max_new_tokens:     int   = 300    # per answer turn
    temperature:        float = 0.8
    max_ctx_len:        int   = 2048   # max tokens fed to the model per turn

    # loss
    clip_eps:           float = 0.2    # PPO-style clipping epsilon
    beta:               float = 0.01   # KL penalty coefficient (0 = pure GRPO)
    use_ref_model:      bool  = True

    # vllm
    vllm_gpu_memory_utilization: float = 0.4   # fraction of GPU VRAM for vLLM KV cache

    # training
    lr:                 float = 1e-5
    batch_size:         int   = 8      # stories per gradient step
    num_epochs:         int   = 1
    max_steps:          int   = 1000   # stop after this many gradient steps
    grad_clip:          float = 1.0
    bf16:               bool  = True

    # dataset
    dataset:            str   = "coqa"     # "coqa" or "gsm8k"
    gsm8k_data_dir:     str   = "/fs/scratch/PAS2836/owos/experiments/RL_MTT/gsm8k_multiturn"
    reward_type:        str   = "f1"       # "f1" or "numeric"

    # checkpointing
    save_every:         int   = 100         # save checkpoint every N steps
    resume_from:        str   = ""          # path to checkpoint dir to resume from

    # LoRA
    use_lora:               bool  = False
    lora_r:                 int   = 16
    lora_alpha:             int   = 32
    lora_dropout:           float = 0.05
    lora_target_modules:    str   = "all-linear"  # "all-linear" or comma-separated names

    # tokenization
    tokenization_strategy: str = "incremental"  # "single" | "full" | "incremental" | "aligned"

    # logging
    use_wandb:          bool  = True
    wandb_project:      str   = "grpo-coqa"
    wandb_run_name:     str   = ""          # empty = wandb auto-names
    log_every:          int   = 10          # train log interval (steps)
    eval_every:         int   = 100         # eval interval (steps)
    eval_samples:       int   = 50          # stories to score at eval time
    max_answer_words:   int   = 5           # exclude samples with any gold answer longer than this (0 = no filter)


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

def load_coqa():
    raw = datasets.load_dataset("stanfordnlp/coqa", trust_remote_code=True)

    def fmt(sample):
        return {
            "story":        sample["story"],
            "questions":    sample["questions"],
            "gold_answers": sample["answers"]["input_text"],
        }

    train = raw["train"].map(fmt, remove_columns=raw["train"].column_names)
    val   = raw["validation"].map(fmt, remove_columns=raw["validation"].column_names)
    return train, val


def load_gsm8k_singleturn():
    """Raw GSM8K wrapped as single-turn episodes for a baseline GRPO run."""
    import re
    raw = datasets.load_dataset("openai/gsm8k", "main", trust_remote_code=True)

    def fmt(row):
        m = re.search(r"####\s*(.+)$", row["answer"], re.MULTILINE)
        return {
            "story":        row["question"],
            "questions":    ["What is the answer?"],
            "gold_answers": [m.group(1).strip().replace(",", "") if m else ""],
        }

    train = [fmt(r) for r in raw["train"]]
    val   = [fmt(r) for r in raw["test"]]
    return train, val


def strip_final_question_from_story(rec: dict) -> dict:
    """Remove the last sentence if it ends with '?' — it's already asked as the final turn."""
    import re
    story = rec["story"]
    sentences = re.split(r'(?<=[.!?])\s+', story.strip())
    if sentences and sentences[-1].rstrip().endswith("?"):
        story = " ".join(sentences[:-1]).strip()
    return {**rec, "story": story}


def load_gsm8k_multiturn(data_dir: str):
    """Load pre-generated GSM8K multi-turn JSONL files."""
    data_dir = Path(data_dir)
    def read_jsonl(path):
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return [strip_final_question_from_story(r) for r in recs]
    train = read_jsonl(data_dir / "gsm8k_multiturn_train.jsonl")
    val   = read_jsonl(data_dir / "gsm8k_multiturn_test.jsonl")
    return train, val


def filter_short_answers(ds, max_words: int):
    """Drop samples with unanswerable questions or (if max_words > 0) overly long answers."""
    def keep(s):
        for a in s["gold_answers"]:
            if a.strip().lower() == "unknown":
                return False
            if max_words > 0 and len(a.split()) > max_words:
                return False
        return True
    return [s for s in ds if keep(s)]


def collate_fn(batch):
    """Keep as list-of-dicts; no tensor padding needed at this stage."""
    return batch


# ---------------------------------------------------------------------------
# 2. Tokenization  (your control point #1)
# ---------------------------------------------------------------------------
# Prompt format lives in tokenization.build_turn_text.
# Strategy selection is via GRPOConfig.tokenization_strategy:
#   "single"      — current question only, no history
#   "full"        — rebuild + retokenize full context each turn  (default)
#   "incremental" — cache previous IDs, only tokenize new user turn
#   "aligned"     — incremental + BPE-boundary alignment against assistant splits


# ---------------------------------------------------------------------------
# 3. Rollout  — sequential best-of-N via vLLM
# ---------------------------------------------------------------------------

def sync_weights_to_vllm(model, llm: LLM):
    """
    Push current policy weights into the vLLM engine.
    Must be called after every optimizer step so rollouts use up-to-date weights.

    Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 (set at module import above),
    which makes vLLM run in-process so we can access the model directly.
    Path: llm_engine → model_executor (UniProcExecutor)
        → driver_worker (WorkerWrapperBase) → worker → model_runner → model

    For LoRA models: computes merged weights on the fly via get_delta_weight()
    without ever writing back to base_layer.weight.  The merge/unmerge approach
    accumulates bf16 rounding error into the base weights over thousands of steps,
    causing the rollout model and training model to silently diverge.
    """
    from peft import PeftModel

    vllm_model = (
        llm.llm_engine.model_executor
           .driver_worker.worker
           .model_runner.model
    )

    if isinstance(model, PeftModel):
        # Import the LoRA Linear class to identify LoRA-wrapped layers.
        try:
            from peft.tuners.lora.layer import Linear as LoraLinear
        except ImportError:                          # older peft versions
            from peft.tuners.lora import Linear as LoraLinear

        # Pre-compute merged weights for every LoRA layer.
        # get_delta_weight() returns lora_B @ lora_A * scaling with no in-place ops.
        lora_overrides: dict[str, torch.Tensor] = {}
        for module_path, module in model.named_modules():
            if isinstance(module, LoraLinear):
                for adapter in module.active_adapters:
                    hf_name = module_path.removeprefix("base_model.model.") + ".weight"
                    delta = module.get_delta_weight(adapter)   # read-only, no drift
                    lora_overrides[hf_name] = module.base_layer.weight.data + delta

        # Walk the base model's parameters; substitute merged tensors where needed.
        def weights_iter():
            for name, param in model.get_base_model().named_parameters():
                yield name, lora_overrides.get(name, param.data)

        vllm_model.load_weights(weights_iter())
    else:
        # load_weights accepts HuggingFace-format (name, tensor) pairs and handles
        # vLLM's internal layer fusions (e.g. q/k/v → qkv_proj) automatically.
        vllm_model.load_weights(
            (name, param.data) for name, param in model.named_parameters()
        )


@torch.no_grad()
def rollout_batch(
    batch:     list[dict],
    llm:       LLM,
    tokenizer,
    cfg:       GRPOConfig,
    device,
    reward_fn: Callable,
) -> list[dict]:
    """
    Batched multi-turn rollout: one vLLM call per turn across all stories,
    instead of one call per story×turn.  Reduces from batch_size×max_turns
    to max_turns vLLM calls per step.

    At each turn t:
      1. Collect prompts for every story still in play.
      2. Issue a single batched llm.generate() for all of them.
      3. Pick the best answer per story and advance its context.
    """
    B              = len(batch)
    G              = cfg.num_generations
    max_prompt_len = cfg.max_ctx_len - cfg.max_new_tokens

    best_answers = [[] for _ in range(B)]
    turn_toks    = [make_turn_tokenizer(cfg.tokenization_strategy, tokenizer) for _ in range(B)]
    turn_data    = [[] for _ in range(B)]   # turn_data[i] = list-of-turns, each a list of G candidate dicts
    active       = list(range(B))

    sampling_params = SamplingParams(
        n=G,
        temperature=cfg.temperature,
        max_tokens=cfg.max_new_tokens,
        logprobs=1,
        repetition_penalty=1.0,
    )

    for turn_idx in range(cfg.max_turns):
        eligible   = []
        prompt_map = {}   # story index → prompt_ids_list

        for i in active:
            questions = batch[i]["questions"]
            if turn_idx >= len(questions):
                continue
            ids = turn_toks[i].encode_turn(
                batch[i]["story"], questions[: turn_idx + 1], best_answers[i],
                tokenizer, max_length=10_000,
            )
            if len(ids) > max_prompt_len:
                continue   # context exceeded — drop story from remaining turns
            prompt_map[i] = ids
            eligible.append(i)

        if not eligible:
            break

        # One batched vLLM call for all eligible stories at this turn
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_map[i]} for i in eligible],
            sampling_params, use_tqdm=False,
        )

        eos_id     = tokenizer.eos_token_id
        next_active = []
        for i, vllm_out in zip(eligible, outputs):
            prompt_ids = torch.tensor(prompt_map[i], dtype=torch.long)
            gold       = batch[i]["gold_answers"][turn_idx]

            candidates = []
            for output in vllm_out.outputs:
                # Strip trailing EOS — in multi-turn the conversation continues,
                # so we must not train the model to emit EOS mid-dialogue or
                # let it bleed into the next turn's prompt via the cache.
                tok_ids = list(output.token_ids)
                lp_pairs = list(zip(output.token_ids, output.logprobs))
                if tok_ids and tok_ids[-1] == eos_id:
                    tok_ids  = tok_ids[:-1]
                    lp_pairs = lp_pairs[:-1]

                comp_ids = torch.tensor(tok_ids, dtype=torch.long)
                old_lp   = torch.tensor([
                    lp_dict[tok_id].logprob
                    for tok_id, lp_dict in lp_pairs
                ])
                answer = output.text.strip().split("\n")[0].strip()
                candidates.append({
                    "prompt_ids":   prompt_ids,
                    "comp_ids":     comp_ids,
                    "old_logprobs": old_lp,
                    "answer":       answer,
                    "f1":           token_f1(answer, gold),
                })

            turn_data[i].append(candidates)
            best = max(candidates, key=lambda c: c["f1"])
            best_answers[i].append(best["answer"])
            turn_toks[i].notify_completion(best["comp_ids"].tolist())
            next_active.append(i)

        active = next_active

    # Pack into episode dicts, score, and compute advantages
    all_episodes = []
    for i in range(B):
        n_turns = len(turn_data[i])
        if n_turns == 0:
            continue

        questions    = batch[i]["questions"]
        gold_answers = batch[i]["gold_answers"]

        group_episodes = []
        for g in range(G):
            group_episodes.append({
                "turn_prompt_ids":     [turn_data[i][t][g]["prompt_ids"] for t in range(n_turns)],
                "turn_completion_ids": [turn_data[i][t][g]["comp_ids"]   for t in range(n_turns)],
                "old_logprobs":        torch.cat([turn_data[i][t][g]["old_logprobs"] for t in range(n_turns)]),
                "answers":             [turn_data[i][t][g]["answer"] for t in range(n_turns)],
            })

        score_and_assign_advantages(group_episodes, gold_answers[:n_turns], reward_fn)

        for ep in group_episodes:
            ep["questions"]    = questions
            ep["gold_answers"] = gold_answers

        all_episodes.extend(group_episodes)

    return all_episodes


def score_and_assign_advantages(
    group_episodes: list[dict],
    gold_answers:   list[str],
    reward_fn:      Callable[[str, str], float],
) -> None:
    """
    Compute per-turn rewards and group-relative advantages for a group of G episodes.
    Mutates each episode in-place, adding: turn_rewards, reward, advantages.

    For each turn t, normalize across the G rollouts of that turn.
    Tokens in turn t get advantage_t — precise credit assignment vs episode average.
    """
    n_turns = len(group_episodes[0]["answers"])

    for ep in group_episodes:
        ep["turn_rewards"] = [
            reward_fn(gen, gold)
            for gen, gold in zip(ep["answers"], gold_answers)
        ]
        ep["reward"] = sum(ep["turn_rewards"]) / len(ep["turn_rewards"])

    # Group-relative advantage per turn: shape (G,) for each t
    turn_advs = []
    for t in range(n_turns):
        r   = torch.tensor([ep["turn_rewards"][t] for ep in group_episodes])
        adv = (r - r.mean()) / (r.std() + 1e-8)
        turn_advs.append(adv)

    for g_idx, ep in enumerate(group_episodes):
        ep["advantages"] = torch.cat([
            turn_advs[t][g_idx].expand(len(ep["turn_completion_ids"][t]))
            for t in range(n_turns)
        ])  # (sum_T,)


def get_episodes(batch, llm, tokenizer, cfg: GRPOConfig, device, reward_fn: Callable) -> list[dict]:
    """
    Rollout G episodes per story in the batch, score them, and compute advantages.
    Returns a flat list of episode dicts (length = batch_size * G).
    """
    return rollout_batch(batch, llm, tokenizer, cfg, device, reward_fn)


# ---------------------------------------------------------------------------
# 4. Forward pass for log-prob recomputation  (your control point #2)
# ---------------------------------------------------------------------------

def compute_logprobs(
    model,
    turn_prompt_ids:     list[torch.Tensor],   # one (L,) tensor per turn
    turn_completion_ids: list[torch.Tensor],   # one (T,) tensor per turn
    device,
) -> torch.Tensor:
    """
    YOUR FORWARD PASS — computed per turn to keep sequence length bounded.

    Each turn's forward pass is:  [prompt_turn_t | completion_turn_t]
    Max length = max_ctx_len + max_new_tokens, regardless of episode length.

    This is the function you modify to:
      - use custom attention masks
      - inject adapters / LoRA hooks
      - change dtype / precision
      - add any custom computation on top of the logits
    """
    all_lp = []
    for prompt_ids, comp_ids in zip(turn_prompt_ids, turn_completion_ids):
        full_ids = torch.cat([prompt_ids, comp_ids]).unsqueeze(0).to(device)  # (1, L+T)

        logits = model(input_ids=full_ids).logits  # (1, L+T, V)

        shift_logits = logits[0, len(prompt_ids) - 1 : -1, :]           # (T, V)
        lp           = F.log_softmax(shift_logits, dim=-1)
        tok_lp       = lp[torch.arange(len(comp_ids)), comp_ids.to(device)]  # (T,)
        all_lp.append(tok_lp)

    return torch.cat(all_lp)  # (sum_T,)


# ---------------------------------------------------------------------------
# 5. GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss(
    logprobs:     torch.Tensor,   # (sum_T,)  new log-probs
    old_logprobs: torch.Tensor,   # (sum_T,)  from rollout
    advantages:   torch.Tensor,   # (sum_T,)  per-token advantages
    ref_logprobs: Optional[torch.Tensor],  # (sum_T,) or None
    cfg:          GRPOConfig,
) -> torch.Tensor:
    """
    GRPO objective (DeepSeekMath eq. 4):

      L = -E[ min(r*A, clip(r, 1-ε, 1+ε)*A) ] + β * KL(π || π_ref)

    where r = exp(log π - log π_old)
    """
    ratio       = torch.exp(logprobs - old_logprobs)
    clipped     = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
    surrogate   = torch.min(ratio * advantages, clipped * advantages)
    policy_loss = -surrogate.mean()

    kl_loss = torch.tensor(0.0, device=logprobs.device)
    if cfg.use_ref_model and ref_logprobs is not None and cfg.beta > 0.0:
        # KL(π || π_ref) unbiased non-negative estimator (always ≥ 0):
        #   E[exp(log π_ref - log π) - 1 - (log π_ref - log π)]
        log_ratio = ref_logprobs - logprobs
        kl_loss = (torch.exp(log_ratio) - 1 - log_ratio).mean()

    return policy_loss + cfg.beta * kl_loss


def update_policy(
    model,
    ref_model,
    episodes:  list[dict],
    optimizer,
    cfg:       GRPOConfig,
    device,
) -> float:
    """
    One gradient step over all episodes.  Returns mean loss.

    Ref log-probs are computed in a single GPU block (ref model moved to GPU
    once, then back to CPU) to save VRAM.  Policy gradients are accumulated
    one episode at a time so only one episode's activations live in memory.
    """
    n = len(episodes)

    if ref_model is not None:
        ref_model.to(device)
        with torch.no_grad():
            for ep in episodes:
                ep["ref_logprobs"] = compute_logprobs(
                    ref_model, ep["turn_prompt_ids"], ep["turn_completion_ids"], device
                )
        ref_model.cpu()
        torch.cuda.empty_cache()

    total_loss = 0.0
    for ep in episodes:
        logprobs = compute_logprobs(
            model, ep["turn_prompt_ids"], ep["turn_completion_ids"], device
        )
        loss = grpo_loss(
            logprobs,
            ep["old_logprobs"].to(device),
            ep["advantages"].to(device),
            ep.get("ref_logprobs"),
            cfg,
        )
        (loss / n).backward()   # accumulate gradients, free activations immediately
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss / n


# ---------------------------------------------------------------------------
# 6. Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, tokenizer, optimizer, global_step: int, cfg: GRPOConfig):
    from peft import PeftModel
    ckpt_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # PeftModel.save_pretrained saves only the LoRA adapter weights + config.
    # Full model save_pretrained saves all weights as usual.
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save({"global_step": global_step}, os.path.join(ckpt_dir, "trainer_state.pt"))
    print(f"checkpoint saved → {ckpt_dir}")


def load_checkpoint(model, optimizer, cfg: GRPOConfig) -> int:
    """Load from cfg.resume_from. Returns the global_step to resume from."""
    from peft import PeftModel
    ckpt_dir = cfg.resume_from
    if isinstance(model, PeftModel):
        model.load_adapter(ckpt_dir, adapter_name="default")
    else:
        model.load_state_dict(
            AutoModelForCausalLM.from_pretrained(ckpt_dir).state_dict()
        )
    optimizer.load_state_dict(torch.load(os.path.join(ckpt_dir, "optimizer.pt"), weights_only=True))
    state = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"), weights_only=True)
    global_step = state["global_step"]
    print(f"resumed from {ckpt_dir}  (step {global_step})")
    return global_step


# ---------------------------------------------------------------------------
# 7. Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, eval_ds, cfg: GRPOConfig, device, reward_fn: Callable, turn_tok: TurnTokenizer) -> float:
    """
    Greedy-decode one episode per story (no sampling) and return mean reward.
    Runs on the first cfg.eval_samples stories of the eval set.
    """
    model.eval()
    rewards = []
    eval_list = eval_ds if isinstance(eval_ds, list) else list(eval_ds)
    for sample in eval_list[: cfg.eval_samples]:
        story        = sample["story"]
        questions    = sample["questions"]
        gold_answers = sample["gold_answers"]

        answers_so_far = []
        turn_tok.reset()
        for turn_idx in range(len(questions)):
            ids = turn_tok.encode_turn(
                story, questions[: turn_idx + 1], answers_so_far, tokenizer, cfg.max_ctx_len - cfg.max_new_tokens
            )
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            attn_mask = torch.ones_like(input_ids)
            out = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device),
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,            # greedy for eval
                pad_token_id=tokenizer.eos_token_id,
            )
            comp = out[0, len(ids):]
            comp_ids = comp.cpu().tolist()
            if comp_ids and comp_ids[-1] == tokenizer.eos_token_id:
                comp_ids = comp_ids[:-1]
            answer = tokenizer.decode(comp_ids, skip_special_tokens=True).strip().split("\n")[0].strip()
            answers_so_far.append(answer)
            turn_tok.notify_completion(comp_ids)

        rewards.append(compute_reward(answers_so_far, gold_answers, reward_fn))

    return sum(rewards) / len(rewards)


# ---------------------------------------------------------------------------
# 7. Logging
# ---------------------------------------------------------------------------

def log_step(episodes: list[dict], mean_loss: float, global_step: int, epoch: int, use_wandb: bool = True):
    mean_reward = sum(ep["reward"] for ep in episodes) / len(episodes)

    max_turns = max(len(ep["turn_rewards"]) for ep in episodes)
    turn_mean = []
    for t in range(max_turns):
        eps_with_turn = [ep for ep in episodes if t < len(ep["turn_rewards"])]
        turn_mean.append(sum(ep["turn_rewards"][t] for ep in eps_with_turn) / len(eps_with_turn))

    if use_wandb:
        wandb.log({
            "train/loss":        mean_loss,
            "train/mean_reward": mean_reward,
            "train/max_reward":  max(ep["reward"] for ep in episodes),
            "train/min_reward":  min(ep["reward"] for ep in episodes),
            "epoch":             epoch,
            **{f"train/turn_{t}_reward": v for t, v in enumerate(turn_mean)},
        }, step=global_step)

    print(f"step {global_step}  loss={mean_loss:.4f}  train_reward={mean_reward:.4f}")
    print("\n--- sample generations ---")
    for ep in episodes[:5]:
        for q, gen, gold, r in zip(ep["questions"], ep["answers"], ep["gold_answers"], ep["turn_rewards"]):
            print(f"  Q: {q}")
            print(f"  gold: {gold}")
            print(f"  gen:  {gen}  (f1={r:.2f})")
            print()
    print("---\n")


# ---------------------------------------------------------------------------
# 8. Training loop
# ---------------------------------------------------------------------------

def train(cfg: GRPOConfig):
    import random
    import numpy as np
    from datetime import datetime
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not cfg.output_dir:
        cfg.output_dir = f"./{cfg.dataset}_grpo_out_{timestamp}"
    else:
        cfg.output_dir = f"{cfg.output_dir}_{timestamp}"
    print(f"output_dir: {cfg.output_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_config.yaml"), "w") as fh:
        import dataclasses, yaml
        yaml.dump(dataclasses.asdict(cfg), fh, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if cfg.bf16 and torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(device)
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        target_modules = (
            cfg.lora_target_modules
            if cfg.lora_target_modules == "all-linear"
            else [m.strip() for m in cfg.lora_target_modules.split(",")]
        )
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # model = torch.compile(model, dynamic=True)   # reduces activation memory + speeds up training

    ref_model = None
    if cfg.use_ref_model:
        # Keep ref model on CPU — only moved to GPU during update_policy.
        ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    # Adam8bit: optimizer states in 8-bit instead of 32-bit — ~4x smaller.
    # With LoRA only the adapter parameters require gradients.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.lr)

    # vLLM engine for generation — lives alongside the training model on the same GPU.
    llm = LLM(
        model=cfg.model_name,
        dtype="bfloat16" if cfg.bf16 else "float32",
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,   # skip CUDA graph capture to save memory at startup
    )

    reward_fn = numeric_reward if cfg.reward_type == "numeric" else token_f1
    turn_tok  = make_turn_tokenizer(cfg.tokenization_strategy, tokenizer)

    if cfg.dataset == "gsm8k":
        train_ds, eval_ds = load_gsm8k_multiturn(cfg.gsm8k_data_dir)
    elif cfg.dataset == "gsm8k_single":
        train_ds, eval_ds = load_gsm8k_singleturn()
    else:
        train_ds, eval_ds = load_coqa()

    before   = len(train_ds)
    train_ds = filter_short_answers(list(train_ds), cfg.max_answer_words)
    eval_ds  = filter_short_answers(list(eval_ds),  cfg.max_answer_words)
    print(f"[dataset] kept {len(train_ds)}/{before} train samples (max_answer_words={cfg.max_answer_words})")

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name or None, config=cfg.__dict__)

    global_step = 0
    if cfg.resume_from:
        global_step = load_checkpoint(model, optimizer, cfg)
        sync_weights_to_vllm(model, llm)
    for epoch in range(cfg.num_epochs):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):

            model.eval()
            episodes = get_episodes(batch, llm, tokenizer, cfg, device, reward_fn)

            model.train()
            mean_loss = update_policy(model, ref_model, episodes, optimizer, cfg, device)
            sync_weights_to_vllm(model, llm)

            global_step += 1

            if global_step % cfg.log_every == 0:
                log_step(episodes, mean_loss, global_step, epoch, cfg.use_wandb)

            if global_step % cfg.save_every == 0:
                save_checkpoint(model, tokenizer, optimizer, global_step, cfg)

            if global_step % cfg.eval_every == 0:
                eval_reward = evaluate(model, tokenizer, eval_ds, cfg, device, reward_fn, turn_tok)
                if cfg.use_wandb:
                    wandb.log({"eval/mean_reward": eval_reward}, step=global_step)
                print(f"step {global_step}  eval_reward={eval_reward:.4f}")
                model.train()

            if global_step >= cfg.max_steps:
                break

        if global_step >= cfg.max_steps:
            break

    if cfg.use_wandb:
        wandb.finish()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    import argparse, dataclasses, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")
    # Optional per-field overrides — any field left unset falls back to the config file or the dataclass default.
    for f in dataclasses.fields(GRPOConfig):
        parser.add_argument(f"--{f.name}", default=None,
                            type=lambda x, t=type(f.default): (
                                x.lower() not in ("false", "0", "no") if t is bool else t(x)
                            ))
    args = parser.parse_args()

    # 1. Start from dataclass defaults.
    cfg_dict = {f.name: f.default for f in dataclasses.fields(GRPOConfig)}

    # 2. Override with YAML config file.
    if args.config:
        with open(args.config) as fh:
            cfg_dict.update(yaml.safe_load(fh))

    # 3. Override with any explicit CLI flags.
    for f in dataclasses.fields(GRPOConfig):
        val = getattr(args, f.name)
        if val is not None:
            cfg_dict[f.name] = val

    train(GRPOConfig(**cfg_dict))
