"""
train_sft_binary10k_fresh_3gpu.py
================================

Fresh binary-only SFT from raw Qwen/Qwen3-1.7B.

Scientific contract:
- No previous SFT/GRPO adapter is loaded.
- Same 10,000 unique ALLOWED/DISMISSED cases used by the subsequent GRPO stage.
- Same physical 10,008-row balanced exposure schedule used by GRPO (8 balanced
  padding exposures make the DDP/GRPO geometry exact; unique IDs remain 10,000).
- Same binary system/user prompt as GRPO.
- Completion-only supervised loss; prompt tokens are masked.
- Exactly one epoch; no early stopping and no user max_steps.
- LoRA r=16, alpha=32, dropout=0.05, bias=none on attention + MLP projections.
- 3-GPU DDP, bf16, SDPA, fused AdamW, no quantisation.
- Final adapter is saved only if the exact expected optimizer-step count completes.
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

import firac_binary_soft_rca as fc

parser = argparse.ArgumentParser()
parser.add_argument('--train-dataset', required=True)
parser.add_argument('--output-adapter', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--summary-path', required=True)
parser.add_argument('--epochs', type=float, default=1.0)
parser.add_argument('--per-device-batch', type=int, default=4)
parser.add_argument('--grad-accum', type=int, default=2)
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--max-length', type=int, default=1664)
parser.add_argument('--lora-r', type=int, default=16)
parser.add_argument('--lora-alpha', type=int, default=32)
parser.add_argument('--save-steps', type=int, default=100)
parser.add_argument('--logging-steps', type=int, default=5)
parser.add_argument('--dataloader-workers', type=int, default=4)
parser.add_argument('--gradient-checkpointing', action='store_true')
args = parser.parse_args()

LOCAL_RANK = int(os.environ.get('LOCAL_RANK', '0'))
RANK = int(os.environ.get('RANK', '0'))
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
IS_MAIN = RANK == 0

if args.epochs != 1.0:
    raise ValueError('Fresh dissertation SFT is fixed to exactly one epoch.')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required.')
if torch.cuda.device_count() < WORLD_SIZE:
    raise RuntimeError(f'WORLD_SIZE={WORLD_SIZE} but only {torch.cuda.device_count()} GPUs are visible.')
torch.cuda.set_device(LOCAL_RANK)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

def p0(*values, **kwargs):
    if IS_MAIN:
        print(*values, **kwargs, flush=True)

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
import peft
import peft.tuners.lora.awq as peft_lora_awq
peft_lora_awq.is_auto_awq_available = lambda: False
from trl import SFTConfig, SFTTrainer
import trl.trainer.sft_trainer as sft_impl

# Qwen3 has a large vocabulary; this mathematically equivalent chunked-NLL
# setting avoids large transient float32 LM-head tensors.
if hasattr(sft_impl, '_CHUNKED_LM_HEAD_CHUNK_SIZE'):
    sft_impl._CHUNKED_LM_HEAD_CHUNK_SIZE = 64

set_seed(fc.SFT_SEED)

# ---------------------------------------------------------------------------
# Exact cached prompt/completion dataset prepared by the notebook.
# ---------------------------------------------------------------------------
train_path = Path(args.train_dataset)
if not train_path.exists():
    raise FileNotFoundError(train_path)
sft_train = load_from_disk(str(train_path))
required = {'prompt', 'completion'}
missing = required - set(sft_train.column_names)
if missing:
    raise ValueError(f'Missing SFT columns: {sorted(missing)}')
if len(sft_train) != 10_008:
    raise RuntimeError(f'Expected 10,008 physical SFT exposures, found {len(sft_train):,}.')

# Keep only fields consumed by the prompt-completion SFT trainer.
sft_train = sft_train.remove_columns(
    [c for c in sft_train.column_names if c not in {'prompt', 'completion'}]
)

# ---------------------------------------------------------------------------
# Tokenizer + full length audit. No silent truncation is allowed.
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(fc.MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

over_limit = 0
max_total = max_prompt = max_completion = 0
for prompt, completion in zip(sft_train['prompt'], sft_train['completion']):
    p = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
    c = len(tokenizer(completion, add_special_tokens=False)['input_ids'])
    total = p + c
    max_total = max(max_total, total)
    max_prompt = max(max_prompt, p)
    max_completion = max(max_completion, c)
    over_limit += int(total > args.max_length)
if over_limit:
    raise RuntimeError(
        f'{over_limit} SFT examples exceed max_length={args.max_length}; refusing silent truncation. '
        f'max_total={max_total}, max_prompt={max_prompt}, max_completion={max_completion}'
    )
p0(f'SFT_LENGTH_AUDIT_PASS max_total={max_total} max_prompt={max_prompt} max_completion={max_completion}')

# ---------------------------------------------------------------------------
# Fresh base model + new LoRA adapter.
# ---------------------------------------------------------------------------
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def load_base_model(name, **extra):
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=compute_dtype, **extra)
    except (TypeError, ValueError) as error:
        if 'dtype' not in str(error):
            raise
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=compute_dtype, **extra)

model = load_base_model(
    fc.MODEL_NAME,
    device_map={'': LOCAL_RANK},
    low_cpu_mem_usage=True,
    attn_implementation='sdpa',
)
model.config.use_cache = False
if args.gradient_checkpointing:
    model.enable_input_require_grads()

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=[
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
    ],
)

# ---------------------------------------------------------------------------
# SFT configuration.
# ---------------------------------------------------------------------------
config_parameters = inspect.signature(SFTConfig.__init__).parameters
config_kwargs = {
    'output_dir': args.output_dir,
    'num_train_epochs': args.epochs,
    'per_device_train_batch_size': args.per_device_batch,
    'gradient_accumulation_steps': args.grad_accum,
    'learning_rate': args.learning_rate,
    'optim': 'adamw_torch_fused' if torch.cuda.is_available() else 'adamw_torch',
    'weight_decay': 0.01,
    'warmup_ratio': 0.03,
    'lr_scheduler_type': 'cosine',
    'max_grad_norm': 1.0,
    'bf16': torch.cuda.is_bf16_supported(),
    'fp16': not torch.cuda.is_bf16_supported(),
    'gradient_checkpointing': args.gradient_checkpointing,
    'logging_strategy': 'steps',
    'logging_steps': args.logging_steps,
    'logging_first_step': True,
    'save_strategy': 'steps',
    'save_steps': args.save_steps,
    'save_total_limit': 3,
    'eval_strategy': 'no',
    'report_to': ['tensorboard'],
    'logging_dir': str(Path(args.output_dir).parent / 'tensorboard_logs'),
    'run_name': 'qwen3-1.7b-fresh-binary-sft10k',
    'seed': fc.SFT_SEED,
    'data_seed': fc.SFT_SEED,
    'packing': False,
    'ddp_find_unused_parameters': False,
    'dataloader_num_workers': args.dataloader_workers,
}
if 'completion_only_loss' not in config_parameters:
    raise RuntimeError('Installed TRL lacks completion_only_loss; refusing full-sequence SFT.')
config_kwargs['completion_only_loss'] = True
if 'max_length' in config_parameters:
    config_kwargs['max_length'] = args.max_length
elif 'max_seq_length' in config_parameters:
    config_kwargs['max_seq_length'] = args.max_length
else:
    raise RuntimeError('SFTConfig exposes no maximum-length argument.')
if args.gradient_checkpointing and 'gradient_checkpointing_kwargs' in config_parameters:
    config_kwargs['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
if 'loss_type' in config_parameters:
    config_kwargs['loss_type'] = 'nll'
if 'dataloader_pin_memory' in config_parameters:
    config_kwargs['dataloader_pin_memory'] = True
if 'dataloader_persistent_workers' in config_parameters:
    config_kwargs['dataloader_persistent_workers'] = args.dataloader_workers > 0
if 'dataloader_prefetch_factor' in config_parameters and args.dataloader_workers > 0:
    config_kwargs['dataloader_prefetch_factor'] = 4

sft_args = SFTConfig(**config_kwargs)
trainer_kwargs = {
    'model': model,
    'args': sft_args,
    'train_dataset': sft_train,
    'peft_config': lora_config,
}
trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters
if 'processing_class' in trainer_parameters:
    trainer_kwargs['processing_class'] = tokenizer
elif 'tokenizer' in trainer_parameters:
    trainer_kwargs['tokenizer'] = tokenizer
trainer = SFTTrainer(**trainer_kwargs)

# Completion-only loss must be proven before the first optimizer step.
prepared_example = trainer.train_dataset[0]
collated = trainer.data_collator([prepared_example])
labels = collated['labels'][0]
masked_tokens = int((labels == -100).sum().item())
trained_tokens = int((labels != -100).sum().item())
if masked_tokens <= 0 or trained_tokens <= 0:
    raise RuntimeError('Completion-only masking verification failed.')
p0(f'COMPLETION_ONLY_LOSS_PREFLIGHT_PASS masked={masked_tokens} trained={trained_tokens}')

GLOBAL_MICROBATCH = args.per_device_batch * WORLD_SIZE
EFFECTIVE_BATCH = GLOBAL_MICROBATCH * args.grad_accum
if len(sft_train) % EFFECTIVE_BATCH != 0:
    raise RuntimeError(
        f'SFT schedule {len(sft_train)} must be divisible by effective batch {EFFECTIVE_BATCH} '
        'for an exact full-epoch step count.'
    )
EXPECTED_STEPS = len(sft_train) // EFFECTIVE_BATCH

trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
p0('='*78)
p0('FRESH BINARY SFT-10K')
p0(f'  world size            : {WORLD_SIZE}')
p0(f'  physical exposures    : {len(sft_train):,}')
p0('  unique source cases   : 10,000 (5,000 ALLOWED + 5,000 DISMISSED)')
p0(f'  per-device batch      : {args.per_device_batch}')
p0(f'  gradient accumulation : {args.grad_accum}')
p0(f'  effective batch       : {EFFECTIVE_BATCH}')
p0(f'  expected steps        : {EXPECTED_STEPS}')
p0(f'  max length            : {args.max_length}')
p0(f'  learning rate         : {args.learning_rate}')
p0(f'  precision             : {compute_dtype}')
p0(f'  LoRA r / alpha        : {args.lora_r} / {args.lora_alpha}')
p0(f'  trainable params      : {trainable:,}/{total:,} ({100*trainable/total:.2f}%)')
p0('='*78)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
start = time.time()
train_output = trainer.train()
elapsed_hours = (time.time() - start) / 3600.0
peak_memory = torch.cuda.max_memory_allocated() / 1024**3
actual_steps = int(trainer.state.global_step)

# Fail closed: an incomplete run must never be saved as the final SFT baseline.
if actual_steps != EXPECTED_STEPS:
    failure = {
        'expected_steps': EXPECTED_STEPS,
        'actual_steps': actual_steps,
        'train_rows': len(sft_train),
        'world_size': WORLD_SIZE,
    }
    if IS_MAIN:
        Path(args.summary_path).with_name('FAILED_INCOMPLETE_SFT.json').write_text(
            json.dumps(failure, indent=2), encoding='utf-8'
        )
    raise RuntimeError(f'Incomplete SFT run: expected {EXPECTED_STEPS}, got {actual_steps}.')

# Synchronise once before filesystem I/O; only rank 0 writes final artifacts.
trainer.accelerator.wait_for_everyone()
if IS_MAIN:
    output_adapter = Path(args.output_adapter)
    output_adapter.mkdir(parents=True, exist_ok=True)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    save_kwargs = {'safe_serialization': True}
    if 'selected_adapters' in inspect.signature(unwrapped.save_pretrained).parameters:
        save_kwargs['selected_adapters'] = ['default']
    unwrapped.save_pretrained(str(output_adapter), **save_kwargs)
    tokenizer.save_pretrained(str(output_adapter))
    required_files = ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer_config.json']
    missing = [name for name in required_files if not (output_adapter/name).exists()]
    if missing:
        raise RuntimeError(f'Fresh SFT adapter save incomplete: {missing}')
    summary = {
        'stage': 'fresh_binary_sft10k',
        'base_model': fc.MODEL_NAME,
        'previous_adapter_loaded': False,
        'binary_only': True,
        'unique_training_cases': 10_000,
        'unique_train_counts': {'allowed': 5000, 'dismissed': 5000},
        'physical_training_exposures': len(sft_train),
        'padding_exposures': 8,
        'epochs': args.epochs,
        'world_size': WORLD_SIZE,
        'per_device_batch': args.per_device_batch,
        'gradient_accumulation': args.grad_accum,
        'effective_batch': EFFECTIVE_BATCH,
        'expected_steps': EXPECTED_STEPS,
        'actual_steps': actual_steps,
        'completion_only_loss': True,
        'qwen_thinking': False,
        'max_length': args.max_length,
        'learning_rate': args.learning_rate,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': 0.05,
        'lora_bias': 'none',
        'target_modules': ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        'gradient_checkpointing': bool(args.gradient_checkpointing),
        'elapsed_hours': elapsed_hours,
        'rank0_peak_memory_gib': peak_memory,
        'metrics': train_output.metrics,
    }
    Path(args.summary_path).write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    p0('FRESH_BINARY_SFT_COMPLETE')
    p0(f'  adapter       : {output_adapter.resolve()}')
    p0(f'  steps         : {actual_steps}/{EXPECTED_STEPS}')
    p0(f'  elapsed hours : {elapsed_hours:.2f}')
    p0(f'  peak VRAM     : {peak_memory:.2f} GiB')
