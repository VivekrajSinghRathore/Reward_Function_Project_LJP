"""
train_grpo.py
=============

Three-GPU GRPO training for the FIRAC Qwen3-1.7B pipeline.

Launched with:

    torchrun --standalone --nproc_per_node=3 train_grpo.py [options]

Changes relative to the previous version, in order of expected impact:

1.  vLLM colocate generation instead of HuggingFace `generate`.
2.  bf16 LoRA instead of 4-bit QLoRA (no NF4 dequantisation on every pass).
3.  Prompts pre-rendered with `enable_thinking=False`, so GRPO rollouts use
    the same template SFT was trained on.
4.  Sampling uses vLLM-safe neutral values and an exact-call-boundary
    sanitizer: `top_p=1.0`, `top_k=-1`, `min_p=0.0`, and
    `repetition_penalty=1.0`. This prevents a later TRL default from
    passing `None` into vLLM.
5.  `num_generations` raised from 3 to 8.
6.  `max_prompt_length` raised so the system prompt is no longer
    left-truncated away.
7.  Truncated completions masked out of the loss.
8.  Reference embeddings precomputed once instead of during training.
9.  Shared code imported from `firac_common`, not exec'd out of notebook
    cells by index.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

import firac_common as fc


# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--sft-adapter", default="./firac_qwen3_1_7b_sft_v2_adapter")
parser.add_argument("--output-dir", default="./firac_opt_v2/grpo_checkpoints")
parser.add_argument("--final-adapter", default="./firac_opt_v2/grpo_adapter")
parser.add_argument("--cache-dir", default="./firac_opt_v2/prepared")
parser.add_argument("--logging-dir", default="./firac_opt_v2/tensorboard_logs")
parser.add_argument("--resume", default="")

parser.add_argument("--num-generations", type=int, default=8)
parser.add_argument("--per-device-batch", type=int, default=2)
parser.add_argument("--grad-accum", type=int, default=16)
parser.add_argument("--learning-rate", type=float, default=5e-6)
parser.add_argument("--beta", type=float, default=0.01)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-prompt-length", type=int, default=1024)
parser.add_argument("--max-completion-length", type=int, default=512)
parser.add_argument("--epochs", type=float, default=1.0)
parser.add_argument("--save-steps", type=int, default=25)

parser.add_argument("--no-vllm", action="store_true",
                    help="Fall back to HuggingFace generate.")
parser.add_argument("--vllm-memory", type=float, default=0.25)
parser.add_argument("--gradient-checkpointing", action="store_true",
                    help="Only needed if you hit OOM; costs ~30%% throughput.")
parser.add_argument("--max-steps", type=int, default=-1,
                    help="Set to e.g. 20 to benchmark seconds/step, then exit.")
parser.add_argument("--train-subset", type=int, default=0,
                    help="Use only the first N prompts (0 = all).")

args = parser.parse_args()


# =============================================================================
# Distributed setup
# =============================================================================

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required.")

if torch.cuda.device_count() < WORLD_SIZE:
    raise RuntimeError(
        f"WORLD_SIZE={WORLD_SIZE} but only "
        f"{torch.cuda.device_count()} GPUs are visible."
    )

torch.cuda.set_device(LOCAL_RANK)

if WORLD_SIZE > 1 and not dist.is_initialized():
    dist.init_process_group(backend="nccl")

IS_MAIN = RANK == 0


def main_print(*values, **kwargs):
    if IS_MAIN:
        print(*values, **kwargs, flush=True)


def barrier():
    if WORLD_SIZE > 1 and dist.is_initialized():
        dist.barrier()


from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed  # noqa: E402
from peft import PeftModel  # noqa: E402

# The model and adapter are bf16 LoRA, not AWQ. Disable PEFT's irrelevant AWQ
# dispatcher so a stale AutoAWQ package cannot break adapter loading.
import peft.tuners.lora.awq as peft_lora_awq  # noqa: E402
peft_lora_awq.is_auto_awq_available = lambda: False
print(
    f"[rank {RANK}] PEFT_AWQ_DISPATCH_DISABLED",
    flush=True,
)

from trl import GRPOConfig, GRPOTrainer  # noqa: E402


def install_safe_vllm_sampling_patch() -> None:
    """Sanitise the exact kwargs passed by TRL to vLLM SamplingParams.

    Some TRL/vLLM combinations construct generation kwargs containing
    ``top_k=None`` or ``min_p=None`` even when GRPOConfig was assigned numeric
    values. vLLM validates these fields immediately and raises TypeError.
    Replacing the module-level constructor used by TRL guarantees that None
    cannot reach vLLM.
    """
    if args.no_vllm:
        return

    try:
        import trl.generation.vllm_generation as trl_vllm_generation
    except Exception as error:
        raise RuntimeError(
            "vLLM was requested but TRL's vLLM generation module could not "
            f"be imported: {error}"
        ) from error

    original = trl_vllm_generation.SamplingParams
    if getattr(original, "_firac_safe_sampling_wrapper", False):
        return

    def safe_sampling_params(*positional, **generation_kwargs):
        if generation_kwargs.get("top_k") is None:
            generation_kwargs["top_k"] = -1
        if generation_kwargs.get("min_p") is None:
            generation_kwargs["min_p"] = 0.0
        if generation_kwargs.get("top_p") is None:
            generation_kwargs["top_p"] = 1.0
        if generation_kwargs.get("repetition_penalty") is None:
            generation_kwargs["repetition_penalty"] = 1.0
        return original(*positional, **generation_kwargs)

    safe_sampling_params._firac_safe_sampling_wrapper = True
    trl_vllm_generation.SamplingParams = safe_sampling_params
    print(
        f"[rank {RANK}] FIRAC_VLLM_SAMPLING_PATCH_ACTIVE "
        "top_k=-1 min_p=0.0 top_p=1.0 repetition_penalty=1.0",
        flush=True,
    )


install_safe_vllm_sampling_patch()

SEED = 42
set_seed(SEED)


# =============================================================================
# Tokenizer
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"


# =============================================================================
# Dataset -- prepared once on rank 0, then read by every rank
# =============================================================================

cache_dir = Path(args.cache_dir)
train_cache = cache_dir / "grpo_train"

if IS_MAIN and not train_cache.exists():
    from datasets import load_dataset

    main_print("Rank 0: building the balanced 10K / 5K split...")

    dataset = load_dataset(fc.DATASET_NAME, fc.DATASET_CONFIG)
    raw_train, raw_eval = fc.build_balanced_splits(dataset)

    training_schedule = fc.build_grpo_training_schedule(raw_train)
    mapper = fc.make_grpo_mapper(tokenizer)

    keep = [
        "prompt",
        "reference_issues",
        "reference_rules",
        "reference_authorities",
        "outcome_label",
        "case_id",
    ]

    prepared_train = training_schedule.map(
        mapper,
        remove_columns=[
            c for c in training_schedule.column_names if c not in keep
        ],
        desc="Rendering GRPO prompts (OTHER repeated once)",
    )
    prepared_eval = raw_eval.map(
        mapper,
        remove_columns=[c for c in raw_eval.column_names if c not in keep],
        desc="Rendering held-out prompts",
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_train.save_to_disk(str(train_cache))
    prepared_eval.save_to_disk(str(cache_dir / "heldout_eval"))

    main_print(f"Rank 0: cached {len(prepared_train):,} train / "
               f"{len(prepared_eval):,} eval rows.")

barrier()

from datasets import load_from_disk  # noqa: E402

grpo_train = load_from_disk(str(train_cache))

if args.train_subset > 0:
    grpo_train = grpo_train.select(range(min(args.train_subset, len(grpo_train))))

required_columns = {
    "prompt",
    "reference_issues",
    "reference_rules",
    "reference_authorities",
    "outcome_label",
}

missing = required_columns - set(grpo_train.column_names)
if missing:
    raise ValueError(f"Missing GRPO columns: {sorted(missing)}")

if not isinstance(grpo_train[0]["prompt"], str):
    raise TypeError(
        "The prompt column must be a pre-rendered string. A list-of-messages "
        "prompt would make TRL re-apply the chat template with "
        "enable_thinking=True, which is the mismatch this script fixes."
    )


# =============================================================================
# Prompt length audit -- silent left-truncation is a correctness bug
# =============================================================================

if IS_MAIN:
    sample = grpo_train.select(range(min(512, len(grpo_train))))
    lengths = [
        len(tokenizer(text, add_special_tokens=False)["input_ids"])
        for text in sample["prompt"]
    ]
    lengths.sort()
    over = sum(1 for n in lengths if n > args.max_prompt_length)

    main_print("-" * 70)
    main_print("PROMPT LENGTH AUDIT (512-row sample)")
    main_print(f"  median      : {lengths[len(lengths) // 2]}")
    main_print(f"  p95         : {lengths[int(len(lengths) * 0.95)]}")
    main_print(f"  max         : {lengths[-1]}")
    main_print(f"  limit       : {args.max_prompt_length}")
    main_print(f"  truncated   : {over} / {len(lengths)} "
               f"({100.0 * over / len(lengths):.1f}%)")
    if over:
        main_print("  WARNING: TRL truncates prompts from the LEFT, so these "
                   "rows lose the start of the system prompt.")
    main_print("-" * 70)


# =============================================================================
# Model -- bf16 LoRA, no 4-bit quantisation
# =============================================================================

compute_dtype = (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

load_kwargs = {
    "device_map": {"": LOCAL_RANK},
    "low_cpu_mem_usage": True,
}


def load_base_model(name, **extra):
    """transformers renamed `torch_dtype` to `dtype`. Both arrive through
    **kwargs, so the signature cannot be inspected -- try and fall back."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            name, dtype=compute_dtype, **extra
        )
    except (TypeError, ValueError) as error:
        if "dtype" not in str(error):
            raise
        return AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=compute_dtype, **extra
        )


base_model = load_base_model(fc.MODEL_NAME, **load_kwargs)
base_model.config.use_cache = False

policy_model = PeftModel.from_pretrained(
    base_model, args.sft_adapter, is_trainable=True
)

if args.gradient_checkpointing:
    policy_model.enable_input_require_grads()
    policy_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in policy_model.parameters())


# =============================================================================
# Reward -- reference embeddings precomputed once
# =============================================================================

scorer = fc.SemanticScorer(device=f"cuda:{LOCAL_RANK}", batch_size=256)
scorer.load()

reference_strings = fc.collect_reference_strings(grpo_train)
main_print(f"Unique reference strings to embed: {len(reference_strings):,}")

precompute_start = time.time()
scorer.precompute_references(reference_strings, verbose=IS_MAIN)
main_print(f"Reference embedding took {time.time() - precompute_start:.1f}s")

reward_state = {}
firac_reward = fc.build_firac_reward(scorer, reward_state)


# =============================================================================
# GRPO configuration
# =============================================================================

GENERATION_BATCH = args.per_device_batch * WORLD_SIZE * args.grad_accum

if GENERATION_BATCH % args.num_generations != 0:
    raise ValueError(
        f"per_device_batch({args.per_device_batch}) * world_size({WORLD_SIZE}) "
        f"* grad_accum({args.grad_accum}) = {GENERATION_BATCH} must be "
        f"divisible by num_generations({args.num_generations})."
    )

PROMPTS_PER_STEP = GENERATION_BATCH // args.num_generations
APPROX_STEPS = math.ceil(len(grpo_train) * args.epochs / PROMPTS_PER_STEP)

config_parameters = inspect.signature(GRPOConfig.__init__).parameters


def supported(name: str) -> bool:
    return name in config_parameters


config_kwargs = {
    "output_dir": args.output_dir,
    "num_train_epochs": args.epochs,
    "per_device_train_batch_size": args.per_device_batch,
    "gradient_accumulation_steps": args.grad_accum,
    "num_generations": args.num_generations,

    "learning_rate": args.learning_rate,
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,

    "bf16": torch.cuda.is_bf16_supported(),
    "fp16": not torch.cuda.is_bf16_supported(),

    "gradient_checkpointing": args.gradient_checkpointing,

    "logging_strategy": "steps",
    "logging_steps": 5,
    "logging_first_step": True,

    "save_strategy": "steps",
    "save_steps": args.save_steps,
    "save_total_limit": 3,

    "eval_strategy": "no",
    "report_to": (
        ["tensorboard", "wandb"]
        if os.environ.get("ENABLE_WANDB", "0") == "1"
        else ["tensorboard"]
    ),
    "logging_dir": args.logging_dir,
    "run_name": "qwen3-1.7b-firac-grpo-v2",
    "seed": SEED,

    # Sampling must match the distribution the log-probs are taken from.
    # A repetition penalty or nucleus cut here biases the importance ratio.
    "temperature": args.temperature,

    "ddp_find_unused_parameters": False,
    "dataloader_num_workers": 2,
}

if args.max_steps > 0:
    config_kwargs["max_steps"] = args.max_steps

if args.gradient_checkpointing and supported("gradient_checkpointing_kwargs"):
    config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

if supported("top_p"):
    config_kwargs["top_p"] = 1.0
if supported("top_k"):
    # vLLM SamplingParams requires an integer. -1 disables top-k filtering.
    config_kwargs["top_k"] = -1
if supported("repetition_penalty"):
    config_kwargs["repetition_penalty"] = 1.0
if supported("min_p"):
    # vLLM requires a float. 0.0 disables min-p filtering.
    config_kwargs["min_p"] = 0.0

# TRL merges generation_kwargs after the regular sampling fields.  Put the
# vLLM-safe values here as well so no default/model config can override them.
if supported("generation_kwargs"):
    config_kwargs["generation_kwargs"] = {
        "top_k": -1,
        "min_p": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }

if supported("max_prompt_length"):
    config_kwargs["max_prompt_length"] = args.max_prompt_length

if supported("max_completion_length"):
    config_kwargs["max_completion_length"] = args.max_completion_length
elif supported("max_new_tokens"):
    config_kwargs["max_new_tokens"] = args.max_completion_length
else:
    raise ValueError("GRPOConfig exposes no completion-length argument.")

# KL coefficient. 0.0 means the reference model is never loaded.
if supported("beta"):
    config_kwargs["beta"] = args.beta

# Do not train on completions that hit the length cap: their reward reflects
# truncation rather than reasoning quality.
if supported("mask_truncated_completions"):
    config_kwargs["mask_truncated_completions"] = True

# Dr. GRPO: dividing by the group standard deviation over-weights prompts the
# policy already finds easy. The argument is a bool in some releases and a
# string enum in others, so match whatever this version expects.
if supported("scale_rewards"):
    default = config_parameters["scale_rewards"].default
    if isinstance(default, bool):
        config_kwargs["scale_rewards"] = False
    elif isinstance(default, str):
        config_kwargs["scale_rewards"] = "none"

if supported("log_completions"):
    config_kwargs["log_completions"] = True
if supported("num_completions_to_print"):
    config_kwargs["num_completions_to_print"] = 2

USE_VLLM = (not args.no_vllm) and supported("use_vllm")

if USE_VLLM:
    config_kwargs["use_vllm"] = True
    if supported("vllm_mode"):
        config_kwargs["vllm_mode"] = "colocate"
    if supported("vllm_gpu_memory_utilization"):
        config_kwargs["vllm_gpu_memory_utilization"] = args.vllm_memory
    if supported("vllm_tensor_parallel_size"):
        config_kwargs["vllm_tensor_parallel_size"] = 1
    for name in ("vllm_max_model_len", "vllm_max_model_length"):
        if supported(name):
            config_kwargs[name] = (
                args.max_prompt_length + args.max_completion_length + 32
            )
            break

grpo_args = GRPOConfig(**config_kwargs)

# Defensive assignment for TRL releases whose dataclass/post-init modifies
# generation fields after construction.
if hasattr(grpo_args, "top_k"):
    grpo_args.top_k = -1
if hasattr(grpo_args, "min_p"):
    grpo_args.min_p = 0.0
if hasattr(grpo_args, "generation_kwargs"):
    grpo_args.generation_kwargs = dict(grpo_args.generation_kwargs or {})
    grpo_args.generation_kwargs.update({
        "top_k": -1,
        "min_p": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    })


# =============================================================================
# Monitoring callback
# =============================================================================

from transformers import TrainerCallback  # noqa: E402


class FIRACMonitoringCallback(TrainerCallback):
    reward_keys = [
        "R_structure", "R_coverage", "R_format_soft", "R_format",
        "R_outcome", "R_issues", "R_rules", "R_authority",
        "R_reasoning", "R_total",
    ]

    def __init__(self, logging_dir, state_dict):
        self.logging_dir = logging_dir
        self.state_dict = state_dict
        self.writer = None
        self.previous_time = None
        self.previous_step = 0
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.logging_dir)
        self.previous_time = time.time()
        self.start_time = time.time()
        self.previous_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or self.writer is None:
            return

        step = state.global_step
        details = self.state_dict.get("details", [])

        if details:
            for key in self.reward_keys:
                values = [float(d[key]) for d in details if key in d]
                if values:
                    self.writer.add_scalar(
                        f"firac/{key}", sum(values) / len(values), step
                    )

            outcomes = [
                str(d.get("generated_outcome") or "unknown") for d in details
            ]
            for label in ("allowed", "dismissed", "other", "unknown"):
                rate = sum(value == label for value in outcomes) / len(outcomes)
                self.writer.add_scalar(
                    f"outcome_distribution/{label}", rate, step
                )

            valid_rate = sum(
                bool(d.get("format_valid", False)) for d in details
            ) / len(details)
            self.writer.add_scalar("firac/format_valid_rate", valid_rate, step)

            dominant_rate = max(
                sum(value == label for value in outcomes) / len(outcomes)
                for label in ("allowed", "dismissed", "other", "unknown")
            )
            if dominant_rate >= 0.85:
                print(
                    f"WARNING step {step}: one generated outcome occupies "
                    f"{dominant_rate:.1%} of the latest reward batch.",
                    flush=True,
                )

            totals = [float(d["R_total"]) for d in details]
            if len(totals) > 1:
                mean = sum(totals) / len(totals)
                variance = sum((t - mean) ** 2 for t in totals) / len(totals)
                # If this sits near zero the groups are degenerate and the
                # advantages -- and therefore the gradients -- vanish.
                self.writer.add_scalar("firac/reward_batch_std",
                                       variance ** 0.5, step)
                self.writer.add_scalar("firac/zero_reward_fraction",
                                       sum(t == 0.0 for t in totals) / len(totals),
                                       step)

        for key, value in self.state_dict.get("timings", {}).items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"performance/reward_{key}", value, step)

        now = time.time()
        completed = step - self.previous_step

        if completed > 0 and self.previous_time is not None:
            seconds_per_step = (now - self.previous_time) / completed
            self.writer.add_scalar("performance/seconds_per_step",
                                   seconds_per_step, step)

            if state.max_steps > 0:
                remaining = max(0, state.max_steps - step)
                hours_left = remaining * seconds_per_step / 3600.0
                self.writer.add_scalar(
                    "performance/estimated_hours_remaining", hours_left, step
                )
                elapsed = (now - self.start_time) / 3600.0
                print(
                    f"[step {step}/{state.max_steps}] "
                    f"{seconds_per_step:.1f}s/step | "
                    f"elapsed {elapsed:.2f}h | ETA {hours_left:.2f}h",
                    flush=True,
                )

        self.writer.add_scalar("gpu/allocated_gib",
                               torch.cuda.memory_allocated() / 1024 ** 3, step)
        self.writer.add_scalar("gpu/reserved_gib",
                               torch.cuda.memory_reserved() / 1024 ** 3, step)

        self.writer.flush()
        self.previous_time = now
        self.previous_step = step

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


# =============================================================================
# Trainer
# =============================================================================

trainer_kwargs = {
    "model": policy_model,
    "args": grpo_args,
    "reward_funcs": [firac_reward],
    "train_dataset": grpo_train,
    "callbacks": [FIRACMonitoringCallback(args.logging_dir, reward_state)],
}

trainer_parameters = inspect.signature(GRPOTrainer.__init__).parameters

if "processing_class" in trainer_parameters:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_parameters:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = GRPOTrainer(**trainer_kwargs)

# Final runtime guard: this is the exact object called by
# GRPOTrainer._generate_single_turn.  Force numeric values both on its normal
# attributes and in generation_kwargs, which is merged last by TRL.
if USE_VLLM and hasattr(trainer, "vllm_generation"):
    vllm_gen = trainer.vllm_generation
    vllm_gen.top_k = -1
    vllm_gen.min_p = 0.0
    vllm_gen.generation_kwargs = dict(
        getattr(vllm_gen, "generation_kwargs", None) or {}
    )
    vllm_gen.generation_kwargs.update({
        "top_k": -1,
        "min_p": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    })

    runtime_top_k = vllm_gen.generation_kwargs.get("top_k", vllm_gen.top_k)
    runtime_min_p = vllm_gen.generation_kwargs.get("min_p", vllm_gen.min_p)
    print(
        f"[rank {RANK}] FORCED_VLLM_SAMPLING "
        f"top_k={runtime_top_k!r} min_p={runtime_min_p!r}",
        flush=True,
    )
    if runtime_top_k is None or not isinstance(runtime_top_k, int):
        raise TypeError(
            f"Runtime vLLM top_k must be int, received {runtime_top_k!r}."
        )
    if runtime_min_p is None or not isinstance(runtime_min_p, (int, float)):
        raise TypeError(
            f"Runtime vLLM min_p must be numeric, received {runtime_min_p!r}."
        )

main_print("=" * 78)
main_print("FIRAC GRPO -- OPTIMISED")
main_print("=" * 78)
main_print(f"  world size            : {WORLD_SIZE}")
main_print(f"  training prompts      : {len(grpo_train):,}")
main_print(f"  generations / prompt  : {args.num_generations}")
main_print(f"  per-device batch      : {args.per_device_batch} completions")
main_print(f"  gradient accumulation : {args.grad_accum}")
main_print(f"  completions / step    : {GENERATION_BATCH}")
main_print(f"  prompts / step        : {PROMPTS_PER_STEP}")
main_print(f"  optimiser steps       : {APPROX_STEPS:,}")
main_print(f"  total completions     : "
           f"{len(grpo_train) * args.num_generations * args.epochs:,.0f}")
main_print(f"  max prompt / compl.   : {args.max_prompt_length} / "
           f"{args.max_completion_length}")
main_print(f"  precision             : {compute_dtype} (no 4-bit)")
main_print(f"  vLLM generation       : {USE_VLLM}")
main_print(f"  gradient checkpoint   : {args.gradient_checkpointing}")
main_print(f"  learning rate         : {args.learning_rate}")
main_print(f"  beta (KL)             : {args.beta}")
main_print(f"  temperature           : {args.temperature}")
main_print(f"  checkpoint interval   : {args.save_steps}")
main_print(f"  reward weights        : {fc.REWARD_WEIGHTS}")
main_print(f"  trainable params      : {trainable:,} / {total:,} "
           f"({100.0 * trainable / total:.2f}%)")
main_print("=" * 78)


resume_value = None
if args.resume:
    resume_value = True if args.resume.lower() == "true" else args.resume

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

start_time = time.time()
train_output = trainer.train(resume_from_checkpoint=resume_value)
elapsed_hours = (time.time() - start_time) / 3600.0

peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
print(f"Rank {RANK} peak allocated VRAM: {peak_memory:.2f} GiB", flush=True)

if IS_MAIN:
    final_adapter = Path(args.final_adapter)
    final_adapter.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))

    summary = {
        "world_size": WORLD_SIZE,
        "training_prompts": len(grpo_train),
        "num_generations": args.num_generations,
        "per_device_batch": args.per_device_batch,
        "gradient_accumulation": args.grad_accum,
        "completions_per_step": GENERATION_BATCH,
        "prompts_per_step": PROMPTS_PER_STEP,
        "approximate_steps": APPROX_STEPS,
        "actual_steps": int(trainer.state.global_step),
        "seconds_per_step": (
            elapsed_hours * 3600.0 / max(1, trainer.state.global_step)
        ),
        "elapsed_hours": elapsed_hours,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "temperature": args.temperature,
        "use_vllm": USE_VLLM,
        "quantisation": "none (bf16)",
        "gradient_checkpointing": args.gradient_checkpointing,
        "rank0_peak_memory_gib": peak_memory,
        "reward_timings": reward_state.get("timings", {}),
        "reference_cache_size": scorer.reference_cache_size,
        "metrics": train_output.metrics,
    }

    summary_path = final_adapter.parent / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=float)

    main_print("Training complete.")
    main_print(f"  adapter        : {final_adapter.resolve()}")
    main_print(f"  steps          : {trainer.state.global_step}")
    main_print(f"  seconds / step : {summary['seconds_per_step']:.1f}")
    main_print(f"  elapsed hours  : {elapsed_hours:.2f}")
    main_print(f"  summary        : {summary_path.resolve()}")

barrier()

if WORLD_SIZE > 1 and dist.is_initialized():
    dist.destroy_process_group()
