"""
train_sft.py
============

Produces the SFT adapter that `train_grpo.py` starts from.

    python3 train_sft.py --output-adapter ./firac_qwen3_1_7b_sft_v2_adapter

Single GPU by design. The SFT set is 2,400 training examples, so this is short
(roughly 1-2 hours on one RTX 3090) and the extra failure modes of a
distributed launch are not worth the saving.

Changes from the original SFT cell:

*   bf16 LoRA instead of 4-bit QLoRA. Qwen3-1.7B is 3.4 GB in bf16; NF4 only
    added a dequantisation to every forward pass.
*   OOM-safe defaults for a 24 GiB GPU: `per_device_train_batch_size=1`
    and `gradient_accumulation_steps=8` (effective batch remains 8).
*   Gradient checkpointing is enabled by the notebook launch cell.
*   Targets and prompts built from `firac_common`, so SFT and GRPO cannot
    drift apart.
*   The adapter and the tokenizer are saved to the same directory, because
    `train_grpo.py` loads the tokenizer from the adapter path.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import torch

import firac_common as fc

parser = argparse.ArgumentParser()
parser.add_argument("--output-adapter", default="./firac_qwen3_1_7b_sft_v2_adapter")
parser.add_argument("--output-dir", default="./firac_opt_v2/sft_checkpoints")
parser.add_argument("--cache-dir", default="./firac_opt_v2/prepared")

parser.add_argument("--epochs", type=float, default=2.0)
parser.add_argument("--per-device-batch", type=int, default=1)
parser.add_argument("--grad-accum", type=int, default=8)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--max-length", type=int, default=1536)
parser.add_argument("--lora-r", type=int, default=16)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--gradient-checkpointing", action="store_true")
parser.add_argument("--max-steps", type=int, default=-1)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required.")

from datasets import load_from_disk  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed  # noqa: E402
from peft import LoraConfig  # noqa: E402

# This pipeline uses a normal bf16 Qwen model, not an AWQ-quantized model.
# A stale/deprecated AutoAWQ installation can make PEFT's generic dispatcher
# import incompatible AWQ code while it scans ordinary torch Linear layers.
# Disable only that irrelevant dispatcher; the normal LoRA dispatcher remains.
import peft.tuners.lora.awq as peft_lora_awq  # noqa: E402
peft_lora_awq.is_auto_awq_available = lambda: False
print("PEFT_AWQ_DISPATCH_DISABLED", flush=True)

from trl import SFTConfig, SFTTrainer  # noqa: E402
import trl.trainer.sft_trainer as sft_impl  # noqa: E402

# Qwen3 has a very large vocabulary. TRL's chunked-NLL path creates a
# temporary float32 [chunk_size, vocab_size] tensor. Some TRL builds use a
# chunk large enough to require hundreds of MiB, which can OOM even with a
# micro-batch of one. Use a smaller, mathematically equivalent chunk.
if hasattr(sft_impl, "_CHUNKED_LM_HEAD_CHUNK_SIZE"):
    old_chunk_size = sft_impl._CHUNKED_LM_HEAD_CHUNK_SIZE
    sft_impl._CHUNKED_LM_HEAD_CHUNK_SIZE = 64
    print(
        "TRL chunked-NLL LM-head chunk:",
        old_chunk_size,
        "->",
        sft_impl._CHUNKED_LM_HEAD_CHUNK_SIZE,
    )
else:
    print("WARNING: TRL chunk-size constant was not found.")

set_seed(fc.SFT_SEED)

print("=" * 78)
print("FIRAC SFT")
print("=" * 78)


# =============================================================================
# Tokenizer -- base model, since no adapter exists yet
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(fc.MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Right padding for training; GRPO and evaluation switch to left for generation.
tokenizer.padding_side = "right"


# =============================================================================
# Data
# =============================================================================

cache_dir = Path(args.cache_dir)
raw_train_path = cache_dir / "raw_grpo_train"
raw_eval_path = cache_dir / "raw_heldout_eval"

if raw_train_path.exists():
    raw_train = load_from_disk(str(raw_train_path))
    raw_eval = load_from_disk(str(raw_eval_path))
    print(f"Loaded cached raw splits from {cache_dir}")
else:
    from datasets import load_dataset

    print("Building the balanced 10K / 5K split...")
    dataset = load_dataset(fc.DATASET_NAME, fc.DATASET_CONFIG)
    raw_train, raw_eval = fc.build_balanced_splits(dataset)

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_train.save_to_disk(str(raw_train_path))
    raw_eval.save_to_disk(str(raw_eval_path))
    print(f"Cached raw splits to {cache_dir}")

# The SFT subset is drawn from the 10K training pool only, so the held-out
# 5K stays unseen through both stages.
sft_train, sft_validation = fc.build_sft_pool(raw_train, heldout_eval=raw_eval)

mapper = fc.make_sft_mapper(tokenizer)

sft_train = sft_train.map(
    mapper,
    remove_columns=[
        c for c in sft_train.column_names
        if c not in {"case_id", "simplified_outcome_label"}
    ],
    desc="Rendering SFT prompt-completion pairs",
)
sft_validation = sft_validation.map(
    mapper,
    remove_columns=[
        c for c in sft_validation.column_names
        if c not in {"case_id", "simplified_outcome_label"}
    ],
    desc="Rendering SFT validation pairs",
)

print(f"SFT train rows      : {len(sft_train):,}")
print(f"SFT validation rows : {len(sft_validation):,}")
print("SFT train labels:")
from collections import Counter
print(Counter(str(x).lower() for x in sft_train["simplified_outcome_label"]))
print("SFT validation labels:")
print(Counter(str(x).lower() for x in sft_validation["simplified_outcome_label"]))

# Keep only the fields supported by the prompt-completion trainer.
sft_train = sft_train.remove_columns(
    [c for c in sft_train.column_names if c not in {"prompt", "completion"}]
)
sft_validation = sft_validation.remove_columns(
    [c for c in sft_validation.column_names if c not in {"prompt", "completion"}]
)


# =============================================================================
# Length audit -- prompt and completion remain separate
# =============================================================================

sample = sft_train.select(range(min(600, len(sft_train))))
lengths = []
prompt_lengths = []
completion_lengths = []
truncated = 0

for prompt, completion in zip(sample["prompt"], sample["completion"]):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    total = len(prompt_ids) + len(completion_ids)
    prompt_lengths.append(len(prompt_ids))
    completion_lengths.append(len(completion_ids))
    lengths.append(total)
    truncated += int(total > args.max_length)

lengths_sorted = sorted(lengths)
print("-" * 70)
print("SFT PROMPT-COMPLETION LENGTH AUDIT")
print(f"  median total tokens    : {lengths_sorted[len(lengths_sorted) // 2]}")
print(f"  p95 total tokens       : {lengths_sorted[int(len(lengths_sorted) * 0.95)]}")
print(f"  max total tokens       : {max(lengths_sorted)}")
print(f"  max prompt tokens      : {max(prompt_lengths)}")
print(f"  max completion tokens  : {max(completion_lengths)}")
print(f"  max_length             : {args.max_length}")
print(f"  sequences over limit   : {truncated} / {len(lengths)}")
print("-" * 70)

if truncated:
    raise RuntimeError(
        "Some SFT examples exceed max_length. Increase --max-length or reduce "
        "the evidence budget before training."
    )

print("\nExample prompt:\n")
print(sft_train[0]["prompt"][:1200])
print("\nExample completion:\n")
print(sft_train[0]["completion"])


# =============================================================================
# Model -- bf16 LoRA, no 4-bit
# =============================================================================

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_base_model(name, **extra):
    """transformers renamed `torch_dtype` to `dtype`; both go through
    **kwargs, so try and fall back rather than inspecting the signature."""
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=compute_dtype, **extra)
    except (TypeError, ValueError) as error:
        if "dtype" not in str(error):
            raise
        return AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=compute_dtype, **extra
        )


model = load_base_model(
    fc.MODEL_NAME, device_map={"": 0}, low_cpu_mem_usage=True
)
model.config.use_cache = False

if args.gradient_checkpointing:
    model.enable_input_require_grads()

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

print(f"Loaded {fc.MODEL_NAME} in {compute_dtype}, "
      f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB allocated")


# =============================================================================
# Trainer
# =============================================================================

config_parameters = inspect.signature(SFTConfig.__init__).parameters

config_kwargs = {
    "output_dir": args.output_dir,
    "num_train_epochs": args.epochs,
    "per_device_train_batch_size": args.per_device_batch,
    "gradient_accumulation_steps": args.grad_accum,
    "per_device_eval_batch_size": 1,

    "learning_rate": args.learning_rate,
    "optim": "adamw_torch",
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,

    "bf16": torch.cuda.is_bf16_supported(),
    "fp16": not torch.cuda.is_bf16_supported(),
    "gradient_checkpointing": args.gradient_checkpointing,

    "eval_strategy": "no",
    "logging_strategy": "steps",
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 2,

    "report_to": "none",
    "prediction_loss_only": True,
    "seed": fc.SFT_SEED,
    "packing": False,
}

if "completion_only_loss" not in config_parameters:
    raise RuntimeError(
        "Installed TRL does not support completion_only_loss. "
        "Do not fall back to full-sequence SFT."
    )
config_kwargs["completion_only_loss"] = True

if args.max_steps > 0:
    config_kwargs["max_steps"] = args.max_steps

if args.gradient_checkpointing and "gradient_checkpointing_kwargs" in config_parameters:
    config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

# Explicit memory controls for recent TRL releases.
if "loss_type" in config_parameters:
    config_kwargs["loss_type"] = "chunked_nll"

if "activation_offloading" in config_parameters:
    config_kwargs["activation_offloading"] = True

if "torch_empty_cache_steps" in config_parameters:
    config_kwargs["torch_empty_cache_steps"] = 1

if "max_length" in config_parameters:
    config_kwargs["max_length"] = args.max_length
elif "max_seq_length" in config_parameters:
    config_kwargs["max_seq_length"] = args.max_length
else:
    raise ValueError("SFTConfig exposes no maximum-length argument.")

sft_args = SFTConfig(**config_kwargs)

trainer_kwargs = {
    "model": model,
    "args": sft_args,
    "train_dataset": sft_train,
    "eval_dataset": sft_validation,
    "peft_config": lora_config,
}

trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters

if "processing_class" in trainer_parameters:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_parameters:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = SFTTrainer(**trainer_kwargs)

# Fail closed: verify that prompt tokens are masked and completion tokens train.
prepared_example = trainer.train_dataset[0]
collated = trainer.data_collator([prepared_example])
labels = collated["labels"][0]
masked_tokens = int((labels == -100).sum().item())
trained_tokens = int((labels != -100).sum().item())

print(
    f"Completion-only loss verification: masked_prompt_tokens={masked_tokens}, "
    f"trained_completion_tokens={trained_tokens}"
)
if masked_tokens <= 0 or trained_tokens <= 0:
    raise RuntimeError(
        "Completion-only masking verification failed. Refusing to train."
    )

trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())

print("=" * 78)
print(f"  train rows       : {len(sft_train):,}")
print(f"  effective batch  : {args.per_device_batch * args.grad_accum}")
print(f"  max length       : {args.max_length}")
print(f"  precision        : {compute_dtype} (no 4-bit)")
print(f"  learning rate    : {args.learning_rate}")
print(f"  LoRA r / alpha   : {args.lora_r} / {args.lora_alpha}")
print(f"  trainable params : {trainable:,} / {total:,} "
      f"({100.0 * trainable / total:.2f}%)")
print("=" * 78)

torch.cuda.reset_peak_memory_stats()
start = time.time()

train_output = trainer.train()

elapsed_hours = (time.time() - start) / 3600.0
peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3


# =============================================================================
# Save -- adapter AND tokenizer, since GRPO loads the tokenizer from here
# =============================================================================

output_adapter = Path(args.output_adapter)
output_adapter.mkdir(parents=True, exist_ok=True)

trainer.save_model(str(output_adapter))
tokenizer.save_pretrained(str(output_adapter))

required_files = ["adapter_config.json", "tokenizer_config.json"]
missing = [f for f in required_files if not (output_adapter / f).exists()]
if missing:
    raise RuntimeError(f"Adapter directory is incomplete, missing: {missing}")

summary = {
    "adapter": str(output_adapter.resolve()),
    "train_rows": len(sft_train),
    "validation_rows": len(sft_validation),
    "epochs": args.epochs,
    "effective_batch": args.per_device_batch * args.grad_accum,
    "max_length": args.max_length,
    "learning_rate": args.learning_rate,
    "lora_r": args.lora_r,
    "lora_alpha": args.lora_alpha,
    "quantisation": "none (bf16)",
    "loss_scope": "completion_only",
    "conclusion_contract": "one_word",
    "steps": int(trainer.state.global_step),
    "elapsed_hours": elapsed_hours,
    "peak_memory_gib": peak_memory,
    "metrics": train_output.metrics,
}

with open(output_adapter.parent / "sft_summary.json", "w", encoding="utf-8") as h:
    json.dump(summary, h, indent=2, default=float)

print("=" * 78)
print("SFT complete.")
print(f"  adapter       : {output_adapter.resolve()}")
print(f"  files         : {sorted(p.name for p in output_adapter.iterdir())}")
print(f"  steps         : {trainer.state.global_step}")
print(f"  elapsed hours : {elapsed_hours:.2f}")
print(f"  peak VRAM     : {peak_memory:.2f} GiB")
print("=" * 78)
print("\nNext: run the GRPO sections of the notebook, which will now find")
print(f"this adapter at {output_adapter}.")
