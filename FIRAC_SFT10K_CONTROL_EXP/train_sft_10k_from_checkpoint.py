"""
train_sft_10k_from_checkpoint.py
================================

Continue the existing FIRAC V2 SFT LoRA adapter with supervised training over
the prepared 10K control dataset.

The dataset is prepared by the accompanying notebook from the same deterministic
10K/5K split defined in the archived firac_common.py.

Important:
- Starts from the SAME SFT adapter used to initialise GRPO.
- Continues the EXISTING LoRA weights (does not create a fresh adapter).
- Uses completion-only supervised loss.
- Never uses the held-out 5K for gradient updates.
- Supports 3-GPU torchrun/DDP.
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

parser = argparse.ArgumentParser()
parser.add_argument("--project-root", required=True)
parser.add_argument("--start-adapter", required=True)
parser.add_argument("--train-dataset", required=True)
parser.add_argument("--output-adapter", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--summary-path", required=True)

parser.add_argument("--epochs", type=float, default=1.0)
parser.add_argument("--per-device-batch", type=int, default=1)
parser.add_argument("--grad-accum", type=int, default=8)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--max-length", type=int, default=1536)
parser.add_argument("--save-steps", type=int, default=100)
parser.add_argument("--logging-steps", type=int, default=10)
parser.add_argument("--gradient-checkpointing", action="store_true")
parser.add_argument("--max-steps", type=int, default=-1)
args = parser.parse_args()

PROJECT_ROOT = Path(args.project_root).resolve()
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

import firac_common as fc  # noqa: E402

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN = RANK == 0

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required.")

if torch.cuda.device_count() < WORLD_SIZE:
    raise RuntimeError(
        f"WORLD_SIZE={WORLD_SIZE}, but only {torch.cuda.device_count()} GPUs are visible."
    )

torch.cuda.set_device(LOCAL_RANK)


def p0(*values, **kwargs):
    if IS_MAIN:
        print(*values, **kwargs, flush=True)


from datasets import load_from_disk  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: E402
from peft import PeftModel  # noqa: E402

# The archived pipeline uses bf16 LoRA, not AWQ. Disable PEFT's irrelevant
# AWQ dispatcher so a stale AutoAWQ install cannot break ordinary LoRA loading.
import peft.tuners.lora.awq as peft_lora_awq  # noqa: E402
peft_lora_awq.is_auto_awq_available = lambda: False

from trl import SFTConfig, SFTTrainer  # noqa: E402
import trl.trainer.sft_trainer as sft_impl  # noqa: E402

if hasattr(sft_impl, "_CHUNKED_LM_HEAD_CHUNK_SIZE"):
    sft_impl._CHUNKED_LM_HEAD_CHUNK_SIZE = 64

set_seed(fc.SFT_SEED)

p0("=" * 86)
p0("FIRAC V2 — SFT-10K CONTROL FROM EXISTING SFT CHECKPOINT")
p0("=" * 86)
p0("project root   :", PROJECT_ROOT)
p0("start adapter  :", args.start_adapter)
p0("train dataset  :", args.train_dataset)
p0("output adapter :", args.output_adapter)
p0("world size     :", WORLD_SIZE)

# ---------------------------------------------------------------------------
# Prepared prompt-completion dataset
# ---------------------------------------------------------------------------

full_train = load_from_disk(args.train_dataset)
required = {"case_id", "simplified_outcome_label", "prompt", "completion"}
missing = required - set(full_train.column_names)
if missing:
    raise ValueError(f"Prepared SFT dataset is missing columns: {sorted(missing)}")

unique_case_ids = len(set(map(str, full_train["case_id"])))
if unique_case_ids != 10_000:
    raise ValueError(
        f"Expected 10,000 unique training cases, found {unique_case_ids:,}."
    )

# SFTTrainer needs only prompt and completion.
train_dataset = full_train.remove_columns(
    [c for c in full_train.column_names if c not in {"prompt", "completion"}]
)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(args.start_adapter, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------------------------------------------------------------------------
# Base model + EXISTING trainable LoRA adapter
# ---------------------------------------------------------------------------

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_base_model(name: str):
    kwargs = {"low_cpu_mem_usage": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=compute_dtype, **kwargs
        )
    except (TypeError, ValueError) as error:
        if "dtype" not in str(error):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=compute_dtype, **kwargs
        )
    return model


base_model = load_base_model(fc.MODEL_NAME)
base_model.config.use_cache = False
base_model.to(torch.device("cuda", LOCAL_RANK))

model = PeftModel.from_pretrained(
    base_model,
    args.start_adapter,
    is_trainable=True,
)

if args.gradient_checkpointing:
    model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

if trainable <= 0:
    raise RuntimeError(
        "No trainable parameters. The existing SFT LoRA adapter was not loaded trainably."
    )

# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

config_parameters = inspect.signature(SFTConfig.__init__).parameters

config_kwargs = {
    "output_dir": args.output_dir,
    "num_train_epochs": args.epochs,
    "per_device_train_batch_size": args.per_device_batch,
    "gradient_accumulation_steps": args.grad_accum,

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
    "logging_steps": args.logging_steps,
    "logging_first_step": True,
    "save_strategy": "steps",
    "save_steps": args.save_steps,
    "save_total_limit": 2,

    "report_to": "none",
    "seed": fc.SFT_SEED,
    "packing": False,
    "dataloader_num_workers": 2,
}

# The 5K is a final held-out evaluation set, not an in-training validation set.
if "eval_strategy" in config_parameters:
    config_kwargs["eval_strategy"] = "no"
elif "evaluation_strategy" in config_parameters:
    config_kwargs["evaluation_strategy"] = "no"

if "completion_only_loss" not in config_parameters:
    raise RuntimeError(
        "Installed TRL does not expose completion_only_loss. "
        "Refusing to run a different SFT objective."
    )
config_kwargs["completion_only_loss"] = True

if args.max_steps > 0:
    config_kwargs["max_steps"] = args.max_steps

if args.gradient_checkpointing and "gradient_checkpointing_kwargs" in config_parameters:
    config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

if "loss_type" in config_parameters:
    config_kwargs["loss_type"] = "chunked_nll"

if "activation_offloading" in config_parameters:
    config_kwargs["activation_offloading"] = True

if "torch_empty_cache_steps" in config_parameters:
    config_kwargs["torch_empty_cache_steps"] = 1

if "ddp_find_unused_parameters" in config_parameters and WORLD_SIZE > 1:
    config_kwargs["ddp_find_unused_parameters"] = False

if "max_length" in config_parameters:
    config_kwargs["max_length"] = args.max_length
elif "max_seq_length" in config_parameters:
    config_kwargs["max_seq_length"] = args.max_length
else:
    raise RuntimeError("SFTConfig exposes no max_length/max_seq_length parameter.")

sft_args = SFTConfig(**config_kwargs)

trainer_kwargs = {
    "model": model,
    "args": sft_args,
    "train_dataset": train_dataset,
}

trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters
if "processing_class" in trainer_parameters:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_parameters:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = SFTTrainer(**trainer_kwargs)

# Fail closed: verify prompt tokens are masked and completion tokens train.
prepared_example = trainer.train_dataset[0]
collated = trainer.data_collator([prepared_example])
labels = collated["labels"][0]

masked_tokens = int((labels == -100).sum().item())
trained_tokens = int((labels != -100).sum().item())

p0(
    "completion-only verification:",
    f"masked_prompt_tokens={masked_tokens}, trained_completion_tokens={trained_tokens}",
)

if masked_tokens <= 0 or trained_tokens <= 0:
    raise RuntimeError("Completion-only masking verification failed.")

global_effective_batch = WORLD_SIZE * args.per_device_batch * args.grad_accum
approx_steps = math.ceil(
    len(train_dataset) * args.epochs / max(1, global_effective_batch)
)

p0("=" * 86)
p0("TRAINING CONFIGURATION")
p0("=" * 86)
p0("training exposures    :", len(train_dataset))
p0("unique training cases :", unique_case_ids)
p0("epochs                :", args.epochs)
p0("per-device batch      :", args.per_device_batch)
p0("gradient accumulation :", args.grad_accum)
p0("global effective batch:", global_effective_batch)
p0("approx optimiser steps:", approx_steps)
p0("learning rate         :", args.learning_rate)
p0("max sequence length   :", args.max_length)
p0("precision             :", compute_dtype)
p0(
    "trainable params      :",
    f"{trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)",
)
p0("=" * 86)

# ---------------------------------------------------------------------------
# Train and save
# ---------------------------------------------------------------------------

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
start = time.time()

train_output = trainer.train()

elapsed_hours = (time.time() - start) / 3600.0
peak_gib = torch.cuda.max_memory_allocated() / 1024**3

print(f"[rank {RANK}] peak allocated VRAM: {peak_gib:.2f} GiB", flush=True)

if trainer.is_world_process_zero():
    output_adapter = Path(args.output_adapter)
    output_adapter.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(output_adapter))
    tokenizer.save_pretrained(str(output_adapter))

    required_files = ["adapter_config.json", "tokenizer_config.json"]
    missing_files = [
        name for name in required_files if not (output_adapter / name).exists()
    ]
    if missing_files:
        raise RuntimeError(f"Saved adapter is incomplete: {missing_files}")

    summary = {
        "experiment": "SFT-10K control continued from archived V2 SFT adapter",
        "base_model": fc.MODEL_NAME,
        "start_adapter": str(Path(args.start_adapter).resolve()),
        "output_adapter": str(output_adapter.resolve()),
        "training_exposures": len(train_dataset),
        "unique_training_cases": unique_case_ids,
        "world_size": WORLD_SIZE,
        "epochs": args.epochs,
        "per_device_batch": args.per_device_batch,
        "gradient_accumulation": args.grad_accum,
        "global_effective_batch": global_effective_batch,
        "approximate_steps": approx_steps,
        "actual_steps": int(trainer.state.global_step),
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "precision": str(compute_dtype),
        "quantisation": "none",
        "loss_scope": "completion_only",
        "elapsed_hours": elapsed_hours,
        "rank0_peak_memory_gib": peak_gib,
        "metrics": train_output.metrics,
    }

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )

    p0("=" * 86)
    p0("SFT-10K CONTROL COMPLETE")
    p0("adapter       :", output_adapter.resolve())
    p0("actual steps  :", trainer.state.global_step)
    p0("elapsed hours :", f"{elapsed_hours:.3f}")
    p0("summary       :", summary_path.resolve())
    p0("=" * 86)
