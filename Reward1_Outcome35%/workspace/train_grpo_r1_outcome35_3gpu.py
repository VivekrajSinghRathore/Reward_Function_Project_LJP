"""
train_grpo.py
=============

Fresh-experiment binary 10K-unique three-GPU GRPO after the newly trained binary SFT.

This version fixes the sampler bug found in V1.5: it NEVER overrides
GRPOTrainer._get_train_sampler, so TRL 0.23.1 retains its native RepeatSampler.

Launched with:

    torchrun --standalone --nproc_per_node=3 train_grpo.py [options]

Purpose and fixed design:

1. Start from the validated SFT-10K LoRA checkpoint.
2. Train on exactly 10,000 unique binary cases: 5,000 ALLOWED + 5,000 DISMISSED.
   The physical schedule has 10,008 rows only to fill complete 12-prompt native
   GRPO groups on 3 GPUs; eight deterministic padding exposures are balanced 4/4.
3. Generate eight completions per prompt using TRL's native RepeatSampler.
4. Use the soft-gated reward designed from the earlier bias analysis.
5. Use the frozen initial SFT adapter as the KL reference.
6. Mask truncated completions and preserve the full FIRAC system prompt.
7. Record outcome distributions, reward components, completion samples,
   gradients, KL, entropy, LoRA weight movement and explicit bias tensors.
8. Never stop early: every configured training example is processed.
9. Save checkpoints safely under three-GPU DDP.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from packaging.version import Version

import torch
import torch.distributed as dist

import firac_binary_reward35_rca as fc


# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--sft-adapter", default="./fresh_binary_sft_adapter")
parser.add_argument("--output-dir", default="./firac_grpo_binary10k_rca/checkpoints")
parser.add_argument("--final-adapter", default="./firac_grpo_binary10k_rca/adapter")
parser.add_argument("--cache-dir", default="./firac_grpo_binary10k_rca/prepared")
parser.add_argument("--logging-dir", default="./firac_grpo_binary10k_rca/tensorboard_logs")
parser.add_argument("--resume", default="")

parser.add_argument("--num-generations", type=int, default=8)
parser.add_argument("--per-device-batch", type=int, default=4)
parser.add_argument("--grad-accum", type=int, default=8)
parser.add_argument("--generation-batch-size", type=int, default=192,
                    help="Global rollout batch: 24 prompts x 8 generations, reused across two optimizer steps.")
parser.add_argument("--learning-rate", type=float, default=8e-7)
parser.add_argument("--beta", type=float, default=0.05)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-prompt-length", type=int, default=1024)
parser.add_argument("--max-completion-length", type=int, default=640)
parser.add_argument("--epochs", type=float, default=1.0)
parser.add_argument("--save-steps", type=int, default=50)
parser.add_argument("--save-total-limit", type=int, default=40)
parser.add_argument("--warmup-ratio", type=float, default=0.15)
parser.add_argument("--max-grad-norm", type=float, default=0.30)
parser.add_argument("--parameter-log-steps", type=int, default=25)
parser.add_argument("--histogram-log-steps", type=int, default=100)
parser.add_argument("--audit-jsonl", default="")
parser.add_argument("--parameter-stats-jsonl", default="")

parser.add_argument("--no-vllm", action="store_true",
                    help="Fall back to HuggingFace generate.")
parser.add_argument("--vllm-memory", type=float, default=0.28)
parser.add_argument("--semantic-batch", type=int, default=128)
parser.add_argument("--dataloader-workers", type=int, default=4)
parser.add_argument("--gradient-checkpointing", action="store_true",
                    help="Fallback only if this profile hits OOM; checkpointing lowers throughput.")

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

# Ampere throughput settings. The policy remains bf16; TF32 is used only for
# eligible float32 matrix operations. SDPA is selected when loading Qwen.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(True)

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
import peft  # noqa: E402
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
import trl  # noqa: E402
import transformers  # noqa: E402
import accelerate  # noqa: E402
import datasets as datasets_pkg  # noqa: E402

EXPECTED_VERSIONS = {
    "trl": "0.23.1",
    "transformers": "4.56.2",
    "peft": "0.19.1",
    "accelerate": "1.10.1",
    "datasets": "4.0.0",
}
ACTUAL_VERSIONS = {
    "trl": trl.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    "datasets": datasets_pkg.__version__,
}
if ACTUAL_VERSIONS != EXPECTED_VERSIONS:
    raise RuntimeError(
        "Pinned environment mismatch. Expected "
        f"{EXPECTED_VERSIONS}, got {ACTUAL_VERSIONS}. Run the notebook setup cell."
    )

# API-level guard for the exact sampler fix.
sampler_source = inspect.getsource(GRPOTrainer._get_train_sampler)
if "RepeatSampler" not in sampler_source or "mini_repeat_count=self.num_generations" not in sampler_source:
    raise RuntimeError(
        "Installed TRL GRPOTrainer does not expose the validated native RepeatSampler implementation."
    )


def install_safe_vllm_sampling_patch() -> None:
    """Patch ``SamplingParams`` across TRL 0.23.x and newer layouts.

    TRL 0.23.x exposes ``SamplingParams`` from
    ``trl.trainer.grpo_trainer``. Newer releases moved it under
    ``trl.generation.vllm_generation``. Supporting both avoids the
    ``No module named trl.generation`` failure while retaining the exact
    vLLM-safe neutral sampling values required by this experiment.
    """
    if args.no_vllm:
        return

    backend = None
    backend_name = None

    try:
        import trl.generation.vllm_generation as backend
        backend_name = "trl.generation.vllm_generation"
    except (ImportError, ModuleNotFoundError):
        backend = None

    if backend is None:
        try:
            import trl.trainer.grpo_trainer as backend
            backend_name = "trl.trainer.grpo_trainer"
        except Exception as error:
            raise RuntimeError(
                "Could not import a compatible TRL vLLM backend: "
                f"{error}"
            ) from error

    if not hasattr(backend, "SamplingParams"):
        raise RuntimeError(
            f"{backend_name} does not expose SamplingParams. "
            "Check the pinned TRL/vLLM installation."
        )

    original = backend.SamplingParams
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
    backend.SamplingParams = safe_sampling_params
    print(
        f"[rank {RANK}] FIRAC_VLLM_SAMPLING_PATCH_ACTIVE "
        f"backend={backend_name} top_k=-1 min_p=0.0 top_p=1.0 "
        "repetition_penalty=1.0",
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
# Dataset -- prepared and validated by the notebook before torchrun
# =============================================================================

cache_dir = Path(args.cache_dir)
train_cache = cache_dir / "grpo_train_binary_schedule"

if not train_cache.exists():
    raise FileNotFoundError(
        f"Prepared binary GRPO schedule not found: {train_cache}. "
        "Run the notebook data-preparation cell first."
    )

from datasets import load_from_disk  # noqa: E402
from collections import Counter  # noqa: E402

grpo_train = load_from_disk(str(train_cache))
cache_counts = Counter(fc.normalise_label(x) for x in grpo_train["outcome_label"])
cache_unique_ids = len(set(map(str, grpo_train["case_id"])))
padding_count = sum(bool(x) for x in grpo_train["is_padding"])
expected_counts = Counter({"allowed": 5004, "dismissed": 5004})

if len(grpo_train) != fc.PADDED_SCHEDULE_ROWS:
    raise ValueError(
        f"Binary GRPO schedule must contain {fc.PADDED_SCHEDULE_ROWS:,} rows; "
        f"found {len(grpo_train):,}."
    )
if cache_unique_ids != fc.UNIQUE_TRAIN_CASES:
    raise ValueError(
        f"Binary GRPO schedule must contain exactly {fc.UNIQUE_TRAIN_CASES:,} unique IDs; "
        f"found {cache_unique_ids:,}."
    )
if cache_counts != expected_counts:
    raise ValueError(
        f"Binary GRPO schedule distribution is {dict(cache_counts)}; "
        f"expected {dict(expected_counts)}."
    )
if padding_count != 8:
    raise ValueError(f"Expected exactly 8 balanced padding exposures; found {padding_count}.")
if any(fc.normalise_label(x) not in fc.BINARY_LABELS for x in grpo_train["outcome_label"]):
    raise ValueError("OTHER/non-binary reference detected in training cache.")

required_columns = {
    "prompt", "reference_issues", "reference_rules",
    "reference_authorities", "outcome_label", "case_id", "is_padding",
}
missing = required_columns - set(grpo_train.column_names)
if missing:
    raise ValueError(f"Missing GRPO columns: {sorted(missing)}")
if not isinstance(grpo_train[0]["prompt"], str):
    raise TypeError("The prompt column must be a pre-rendered string.")


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
    "attn_implementation": "sdpa",
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


# =============================================================================
# PEFT 0.19.1 + Transformers 4.56.2 compatibility for ordinary DDP
# =============================================================================
#
# PEFT 0.19.1's adapter-loading helper imports Tensor-Parallel classes
# (including EmbeddingParallel) before it checks whether the model actually
# uses tensor parallelism. Transformers 4.56.2 does not expose that class.
#
# This run uses three independent DDP replicas (one complete model per GPU),
# not Transformers tensor parallelism. Therefore no adapter-weight sharding is
# required. Patch only that TP helper, and refuse to bypass it if TP metadata is
# ever detected.

import peft.utils.save_and_load as peft_save_and_load


def _active_lora_tensor_parallel_layers(model) -> list[tuple[str, object, object]]:
    """Return only LoRA base layers with *active* Transformers TP metadata.

    A model class may define a static `_tp_plan` describing how tensor
    parallelism could be applied. That does not mean TP is active. PEFT's own
    adapter-loading implementation treats a LoRA layer as tensor-parallel only
    when its base layer has both `_hf_tp_plan` and `_hf_device_mesh`.
    """
    active = []

    for name, module in model.named_modules():
        get_base_layer = getattr(module, "get_base_layer", None)
        if not callable(get_base_layer):
            continue

        try:
            base_layer = get_base_layer()
        except Exception:
            continue

        tp_plan = getattr(base_layer, "_hf_tp_plan", None)
        device_mesh = getattr(base_layer, "_hf_device_mesh", None)

        if tp_plan is not None and device_mesh is not None:
            active.append((name, tp_plan, device_mesh))

    return active


try:
    from transformers.integrations.tensor_parallel import EmbeddingParallel  # noqa: F401
    _embedding_parallel_available = True
except ImportError:
    _embedding_parallel_available = False


if not _embedding_parallel_available:
    def _skip_tp_sharding_for_non_tp_ddp(model, state_dict, adapter_name):
        active_tp_layers = _active_lora_tensor_parallel_layers(model)

        if active_tp_layers:
            preview = [
                {
                    "layer": name,
                    "tp_plan": str(tp_plan),
                    "device_mesh": str(device_mesh),
                }
                for name, tp_plan, device_mesh in active_tp_layers[:5]
            ]
            raise RuntimeError(
                "Active tensor parallelism was detected on LoRA base layers, "
                "but this environment does not expose Transformers "
                f"EmbeddingParallel. First active layers: {preview}"
            )

        # PEFT's original function mutates state_dict in place and returns None.
        # In this run each DDP rank owns a complete model and no base LoRA layer
        # has an active device mesh, so adapter sharding is neither required nor
        # correct.
        print(
            f"[rank {RANK}] PEFT_TP_SHARDING_SKIPPED "
            f"adapter={adapter_name} reason=no_active_device_mesh",
            flush=True,
        )
        return None

    peft_save_and_load._maybe_shard_state_dict_for_tp = (
        _skip_tp_sharding_for_non_tp_ddp
    )

    print(
        f"[rank {RANK}] PEFT_TP_SHARDING_PATCH_ACTIVE "
        "mode=DDP_NO_TENSOR_PARALLEL",
        flush=True,
    )


# The checkpoint may contain null configuration fields introduced by PEFT
# 0.20.0. PEFT 0.19.1 can safely ignore them only when they are inactive.
adapter_config_path = Path(args.sft_adapter) / "adapter_config.json"
adapter_config_json = json.loads(
    adapter_config_path.read_text(encoding="utf-8")
)

forward_only_fields = {
    key: adapter_config_json.get(key)
    for key in ("monteclora_config", "velora_config")
    if key in adapter_config_json
}

active_unsupported_fields = {
    key: value
    for key, value in forward_only_fields.items()
    if value not in (None, False, {}, [])
}

if active_unsupported_fields:
    raise RuntimeError(
        "The SFT checkpoint actively uses PEFT features unsupported by the "
        f"pinned environment: {active_unsupported_fields}"
    )

if forward_only_fields:
    print(
        f"[rank {RANK}] Inactive newer PEFT config fields accepted: "
        f"{forward_only_fields}",
        flush=True,
    )

    warnings.filterwarnings(
        "ignore",
        message=(
            r"Unexpected keyword arguments .*"
            r"(monteclora_config|velora_config).*"
            r"for class LoraConfig.*"
        ),
        category=UserWarning,
        module=r"peft\.config",
    )


policy_model = PeftModel.from_pretrained(
    base_model, args.sft_adapter, is_trainable=True
)


# =============================================================================
# Correct reference policy for TRL 0.23.1 + a pre-trained PEFT adapter
# =============================================================================
#
# TRL 0.23.1 computes PEFT reference log-probabilities inside:
#
#     with model.disable_adapter():
#
# For a policy that starts from an already-trained SFT adapter, disabling the
# adapter gives the raw base model, not the SFT starting policy. That makes the
# reported KL large before GRPO has meaningfully changed the policy and applies
# the KL penalty toward the wrong model.
#
# Create a frozen copy of the initial SFT adapter named "ref", then redirect the
# exact context manager used by TRL to activate that copy. The trainable
# "default" adapter remains the policy and is saved normally in checkpoints.

if Version(peft.__version__) < Version("0.19.0"):
    raise RuntimeError(
        "This SFT checkpoint was saved with a newer PEFT schema. "
        f"Installed PEFT={peft.__version__}; install peft==0.19.1, "
        "restart the kernel, and rerun from the beginning."
    )

if "default" not in policy_model.peft_config:
    raise RuntimeError(
        "Expected the loaded SFT adapter to be named 'default', but found: "
        f"{list(policy_model.peft_config)}"
    )

if "ref" in policy_model.peft_config:
    raise RuntimeError(
        "The input SFT adapter unexpectedly already contains an adapter named "
        "'ref'. Use the clean SFT-10K adapter directory."
    )

# Load the same trained SFT adapter a second time from disk. This is safer than
# creating a new adapter and manually copying only named parameters because
# modern PEFT checkpoints may contain auxiliary adapter state and configuration
# fields that older PEFT releases do not understand.
policy_model.load_adapter(
    args.sft_adapter,
    adapter_name="ref",
    is_trainable=False,
    torch_device=f"cuda:{LOCAL_RANK}",
    autocast_adapter_dtype=True,
)


def _set_default_trainable(model) -> None:
    """Keep the GRPO policy trainable and the SFT reference frozen."""
    if hasattr(model, "set_requires_grad"):
        model.set_requires_grad("default", True)
        model.set_requires_grad("ref", False)
    else:
        for name, parameter in model.named_parameters():
            if ".default." in name:
                parameter.requires_grad_(True)
            elif ".ref." in name:
                parameter.requires_grad_(False)


def _activate_adapter(model, adapter_name: str) -> None:
    """Activate one adapter with the correct trainability mode."""
    try:
        model.set_adapter(
            adapter_name,
            inference_mode=(adapter_name == "ref"),
        )
    except TypeError:
        # Compatibility fallback. PEFT 0.19.1 accepts inference_mode.
        model.set_adapter(adapter_name)

    _set_default_trainable(model)


_activate_adapter(policy_model, "default")


@contextmanager
def _use_frozen_sft_reference(self):
    """TRL 0.23.1 compatibility context: use frozen SFT instead of raw Qwen."""
    previous = getattr(self, "active_adapter", "default")
    if isinstance(previous, (list, tuple)):
        previous = previous[0] if previous else "default"
    previous = str(previous or "default")

    _activate_adapter(self, "ref")
    try:
        yield
    finally:
        _activate_adapter(self, previous)


# TRL 0.23.1 calls disable_adapter() to calculate reference log-probabilities.
# Redirect that exact call to the frozen `ref` copy of SFT-10K.
policy_model.disable_adapter = types.MethodType(
    _use_frozen_sft_reference,
    policy_model,
)


def _verify_adapter_tensors(model) -> tuple[int, float]:
    """Require exact equality between all default/ref adapter tensors."""
    checked = 0
    maximum_difference = 0.0
    missing = []

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    for name, tensor in {**parameters, **buffers}.items():
        if ".default." not in name:
            continue

        ref_name = name.replace(".default.", ".ref.")

        if ref_name in parameters:
            ref_tensor = parameters[ref_name]
        elif ref_name in buffers:
            ref_tensor = buffers[ref_name]
        else:
            missing.append(ref_name)
            continue

        checked += 1
        difference = float(
            torch.max(
                torch.abs(
                    tensor.detach().float()
                    - ref_tensor.detach().float()
                )
            ).item()
        )
        maximum_difference = max(maximum_difference, difference)

    if missing:
        raise RuntimeError(
            "Frozen reference is missing adapter tensors. First examples: "
            + ", ".join(missing[:5])
        )

    if checked == 0:
        raise RuntimeError("No default/ref adapter tensors were found.")

    if maximum_difference != 0.0:
        raise RuntimeError(
            "The frozen SFT reference weights do not exactly match the policy. "
            f"Maximum adapter-tensor difference: {maximum_difference}"
        )

    return checked, maximum_difference


def _logit_alignment_diagnostic(
    model,
    tokenizer,
    prompt_text: str,
) -> dict[str, float]:
    """Measure output differences without using an unrealistic bf16 1e-5 gate.

    Repeated bf16 CUDA forwards can have non-zero maximum logit differences
    even when the underlying adapter tensors are identical. Exact tensor
    equality is the hard correctness check; logits are retained as diagnostics.
    """
    was_training = model.training
    model.eval()

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    encoded = {
        key: value.to(f"cuda:{LOCAL_RANK}")
        for key, value in encoded.items()
    }

    with torch.no_grad():
        _activate_adapter(model, "default")
        default_logits_1 = model(**encoded).logits[:, -1, :].float()

        _activate_adapter(model, "default")
        default_logits_2 = model(**encoded).logits[:, -1, :].float()

        _activate_adapter(model, "ref")
        reference_logits = model(**encoded).logits[:, -1, :].float()

    _activate_adapter(model, "default")

    if was_training:
        model.train()

    repeat_difference = torch.abs(default_logits_1 - default_logits_2)
    reference_difference = torch.abs(default_logits_1 - reference_logits)

    return {
        "repeat_max": float(repeat_difference.max().item()),
        "repeat_mean": float(repeat_difference.mean().item()),
        "reference_max": float(reference_difference.max().item()),
        "reference_mean": float(reference_difference.mean().item()),
        "default_top1": int(default_logits_1.argmax(dim=-1).item()),
        "reference_top1": int(reference_logits.argmax(dim=-1).item()),
    }


copied_reference_tensors, reference_tensor_max_difference = (
    _verify_adapter_tensors(policy_model)
)

reference_probe_prompt = (
    "Summarise the legal issue and give a concise FIRAC conclusion."
)
reference_diagnostic = _logit_alignment_diagnostic(
    policy_model,
    tokenizer,
    reference_probe_prompt,
)
reference_max_logit_difference = reference_diagnostic["reference_max"]
reference_repeat_max_logit_difference = reference_diagnostic["repeat_max"]

print(
    f"[rank {RANK}] SFT_REFERENCE_ADAPTER_ACTIVE "
    f"exact_tensors={copied_reference_tensors} "
    f"tensor_max_difference={reference_tensor_max_difference:.8f} "
    f"repeat_logit_max={reference_repeat_max_logit_difference:.8f} "
    f"reference_logit_max={reference_max_logit_difference:.8f} "
    f"top1_match={reference_diagnostic['default_top1'] == reference_diagnostic['reference_top1']}",
    flush=True,
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

scorer = fc.SemanticScorer(
    device=f"cuda:{LOCAL_RANK}",
    batch_size=args.semantic_batch,
)
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

OPTIMIZER_COMPLETION_BATCH = args.per_device_batch * WORLD_SIZE * args.grad_accum

if OPTIMIZER_COMPLETION_BATCH % args.num_generations != 0:
    raise ValueError(
        f"per_device_batch({args.per_device_batch}) * world_size({WORLD_SIZE}) "
        f"* grad_accum({args.grad_accum}) = {OPTIMIZER_COMPLETION_BATCH} must be "
        f"divisible by num_generations({args.num_generations})."
    )

PROMPTS_PER_OPTIMIZER_STEP = OPTIMIZER_COMPLETION_BATCH // args.num_generations
ROLLOUT_GENERATION_BATCH = int(args.generation_batch_size)
GLOBAL_MICROBATCH = args.per_device_batch * WORLD_SIZE
if ROLLOUT_GENERATION_BATCH % GLOBAL_MICROBATCH != 0:
    raise ValueError(
        f"generation_batch_size={ROLLOUT_GENERATION_BATCH} must be divisible by "
        f"global microbatch={GLOBAL_MICROBATCH}."
    )
if ROLLOUT_GENERATION_BATCH % args.num_generations != 0:
    raise ValueError(
        f"generation_batch_size={ROLLOUT_GENERATION_BATCH} must be divisible by "
        f"num_generations={args.num_generations}."
    )
ROLLOUT_PROMPTS = ROLLOUT_GENERATION_BATCH // args.num_generations
EXPECTED_STEPS_PER_GENERATION = ROLLOUT_GENERATION_BATCH // GLOBAL_MICROBATCH
if EXPECTED_STEPS_PER_GENERATION % args.grad_accum != 0:
    raise ValueError(
        "The rollout batch must span a whole number of optimizer steps; "
        f"steps_per_generation={EXPECTED_STEPS_PER_GENERATION}, grad_accum={args.grad_accum}."
    )
OPTIMIZER_STEPS_PER_ROLLOUT = EXPECTED_STEPS_PER_GENERATION // args.grad_accum
if args.epochs != 1.0:
    raise ValueError("This dissertation run is fixed to exactly one epoch (epochs=1.0).")
if len(grpo_train) % ROLLOUT_PROMPTS != 0:
    raise RuntimeError(
        f"Schedule length {len(grpo_train)} is not divisible by rollout prompts {ROLLOUT_PROMPTS}; "
        "TRL RepeatSampler would drop an incomplete rollout group."
    )
if len(grpo_train) % PROMPTS_PER_OPTIMIZER_STEP != 0:
    raise RuntimeError(
        f"Schedule length {len(grpo_train)} is not divisible by optimizer prompts "
        f"{PROMPTS_PER_OPTIMIZER_STEP}."
    )
APPROX_STEPS = len(grpo_train) // PROMPTS_PER_OPTIMIZER_STEP
EXPECTED_ROLLOUT_CALLS = len(grpo_train) // ROLLOUT_PROMPTS

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
    "optim": "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    "weight_decay": 0.01,
    "warmup_ratio": args.warmup_ratio,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": args.max_grad_norm,

    "bf16": torch.cuda.is_bf16_supported(),
    "fp16": not torch.cuda.is_bf16_supported(),

    "gradient_checkpointing": args.gradient_checkpointing,

    "logging_strategy": "steps",
    "logging_steps": 1,
    "logging_first_step": True,

    "save_strategy": "steps",
    "save_steps": args.save_steps,
    "save_total_limit": args.save_total_limit,

    "eval_strategy": "no",
    "report_to": (
        ["tensorboard", "wandb"]
        if os.environ.get("ENABLE_WANDB", "0") == "1"
        else ["tensorboard"]
    ),
    "logging_dir": args.logging_dir,
    "run_name": "qwen3-1.7b-firac-binary10k-softgate-rca",
    "seed": SEED,
    "data_seed": SEED,

    # IMPORTANT: keep our deterministic 6/6 source blocks, but retain TRL's
    # native RepeatSampler. Never replace _get_train_sampler.
    "shuffle_dataset": False,

    # Sampling must match the distribution the log-probs are taken from.
    # A repetition penalty or nucleus cut here biases the importance ratio.
    "temperature": args.temperature,

    "ddp_find_unused_parameters": False,
    "dataloader_num_workers": args.dataloader_workers,
}


if supported("generation_batch_size"):
    config_kwargs["generation_batch_size"] = ROLLOUT_GENERATION_BATCH
elif ROLLOUT_GENERATION_BATCH != OPTIMIZER_COMPLETION_BATCH:
    raise RuntimeError(
        "This high-utilisation profile requires GRPOConfig.generation_batch_size, "
        "but the installed TRL does not expose it."
    )
if supported("pad_to_multiple_of"):
    config_kwargs["pad_to_multiple_of"] = 8

if args.gradient_checkpointing and supported("gradient_checkpointing_kwargs"):
    config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

if supported("disable_dropout"):
    config_kwargs["disable_dropout"] = True
if supported("dataloader_pin_memory"):
    config_kwargs["dataloader_pin_memory"] = True
if supported("dataloader_persistent_workers"):
    config_kwargs["dataloader_persistent_workers"] = args.dataloader_workers > 0
if supported("dataloader_prefetch_factor") and args.dataloader_workers > 0:
    config_kwargs["dataloader_prefetch_factor"] = 4

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

if int(getattr(grpo_args, "generation_batch_size", ROLLOUT_GENERATION_BATCH)) != ROLLOUT_GENERATION_BATCH:
    raise RuntimeError("GRPOConfig changed the requested rollout generation batch size.")
if int(getattr(grpo_args, "steps_per_generation", EXPECTED_STEPS_PER_GENERATION)) != EXPECTED_STEPS_PER_GENERATION:
    raise RuntimeError(
        f"Expected steps_per_generation={EXPECTED_STEPS_PER_GENERATION}, got "
        f"{getattr(grpo_args, 'steps_per_generation', None)}."
    )

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
# RCA monitoring: no warnings and no early stopping
# =============================================================================

from transformers import TrainerCallback  # noqa: E402


class FIRACRCAMonitoringCallback(TrainerCallback):
    """Record the evidence needed to explain any output bias after training.

    This callback never changes ``control.should_training_stop``. It only logs:
    reward components, generated/reference outcomes, confusion counts, completion
    samples, GPU use, and LoRA parameter movement from the SFT-10K start.
    """

    reward_keys = [
        "R_structure", "R_coverage", "R_format_soft", "R_format",
        "R_outcome", "R_issues", "R_rules", "R_authority",
        "R_reasoning", "R_total",
    ]

    def __init__(
        self,
        logging_dir,
        state_dict,
        audit_jsonl="",
        parameter_stats_jsonl="",
        parameter_log_steps=5,
        histogram_log_steps=25,
    ):
        self.logging_dir = Path(logging_dir)
        self.state_dict = state_dict
        audit_base = Path(audit_jsonl) if audit_jsonl else self.logging_dir / "completion_audit.jsonl"
        self.audit_jsonl = audit_base.with_name(
            f"{audit_base.stem}.rank{RANK}{audit_base.suffix or '.jsonl'}"
        )
        self.parameter_stats_jsonl = (
            Path(parameter_stats_jsonl)
            if parameter_stats_jsonl
            else self.logging_dir / "parameter_stats.jsonl"
        )
        self.parameter_log_steps = max(1, int(parameter_log_steps))
        self.histogram_log_steps = max(1, int(histogram_log_steps))
        self.writer = None
        self.audit_handle = None
        self.parameter_handle = None
        self.previous_time = None
        self.previous_step = 0
        self.start_time = None
        self.start_params = {}
        self.last_audit_step = None
        self.last_audit_call_count = -1
        self.last_reward_log_call_count = -1
        self.last_parameter_step = None

    @staticmethod
    def _safe_name(name):
        return name.replace(".", "/").replace(":", "_")

    def _capture_start_parameters(self, model):
        self.start_params = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        bias_names = [name for name in self.start_params if ".bias" in name.lower()]
        self.writer.add_scalar("parameters/trainable_tensor_count", len(self.start_params), 0)
        self.writer.add_scalar("parameters/trainable_bias_tensor_count", len(bias_names), 0)
        self.writer.add_text(
            "parameters/bias_note",
            (
                "No explicit trainable bias vectors. Behavioural outcome bias must "
                "therefore arise from LoRA weight changes and token probabilities."
                if not bias_names else "\n".join(bias_names)
            ),
            0,
        )

    def _log_parameter_state(self, model, step, include_histograms):
        if not self.start_params or step == self.last_parameter_step:
            return

        tensor_rows = []
        start_sq = current_sq = delta_sq = 0.0
        maximum_absolute_delta = 0.0
        trainable_numel = 0

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.start_params:
                continue
            current = parameter.detach().float().cpu()
            start = self.start_params[name]
            if current.shape != start.shape:
                continue
            delta = current - start
            safe = self._safe_name(name)
            start_l2 = torch.linalg.vector_norm(start).item()
            current_l2 = torch.linalg.vector_norm(current).item()
            delta_l2 = torch.linalg.vector_norm(delta).item()
            relative = delta_l2 / max(start_l2, 1e-12)

            tensor_type = (
                "lora_A" if "lora_A" in name else
                "lora_B" if "lora_B" in name else
                "bias" if ".bias" in name.lower() else
                "other"
            )
            tensor_rows.append({
                "name": name,
                "tensor_type": tensor_type,
                "numel": int(delta.numel()),
                "start_l2": start_l2,
                "current_l2": current_l2,
                "delta_l2": delta_l2,
                "relative_delta": relative,
                "delta_mean": float(delta.mean().item()),
                "delta_std": float(delta.std(unbiased=False).item()),
                "delta_max_abs": float(delta.abs().max().item()),
            })

            self.writer.add_scalar(f"parameters/current_l2/{safe}", current_l2, step)
            self.writer.add_scalar(f"parameters/delta_l2/{safe}", delta_l2, step)
            self.writer.add_scalar(f"parameters/relative_delta/{safe}", relative, step)

            if include_histograms:
                self.writer.add_histogram(f"weights/current/{safe}", current, step)
                self.writer.add_histogram(f"weights/delta_from_sft10k/{safe}", delta, step)
                if tensor_type == "bias":
                    self.writer.add_histogram(f"bias/current/{safe}", current, step)
                    self.writer.add_histogram(f"bias/delta_from_sft10k/{safe}", delta, step)

            start_sq += torch.sum(start * start).item()
            current_sq += torch.sum(current * current).item()
            delta_sq += torch.sum(delta * delta).item()
            maximum_absolute_delta = max(maximum_absolute_delta, delta.abs().max().item())
            trainable_numel += delta.numel()

        global_row = {
            "step": int(step),
            "global_start_l2": start_sq ** 0.5,
            "global_current_l2": current_sq ** 0.5,
            "global_delta_l2": delta_sq ** 0.5,
            "global_relative_delta": (delta_sq ** 0.5) / max(start_sq ** 0.5, 1e-12),
            "global_max_abs_delta": maximum_absolute_delta,
            "trainable_numel": int(trainable_numel),
            "tensors": tensor_rows,
        }
        self.parameter_handle.write(json.dumps(global_row, default=float) + "\n")

        self.writer.add_scalar("parameters/global_start_l2", global_row["global_start_l2"], step)
        self.writer.add_scalar("parameters/global_current_l2", global_row["global_current_l2"], step)
        self.writer.add_scalar("parameters/global_delta_l2", global_row["global_delta_l2"], step)
        self.writer.add_scalar("parameters/global_relative_delta", global_row["global_relative_delta"], step)
        self.writer.add_scalar("parameters/global_max_abs_delta", maximum_absolute_delta, step)
        self.writer.add_scalar("parameters/trainable_numel", trainable_numel, step)
        self.last_parameter_step = step

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.audit_handle = self.audit_jsonl.open("w", encoding="utf-8", buffering=1024 * 1024)
        self.previous_time = time.time()
        self.start_time = time.time()
        self.previous_step = int(state.global_step)

        if state.is_world_process_zero:
            from torch.utils.tensorboard import SummaryWriter
            self.logging_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.logging_dir))
            self.parameter_stats_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self.parameter_handle = self.parameter_stats_jsonl.open(
                "w", encoding="utf-8", buffering=1024 * 1024
            )
            if model is not None:
                self._capture_start_parameters(model)
                self._log_parameter_state(model, int(state.global_step), include_histograms=True)

    def _write_audit(self, step, details, call_count):
        # A rollout batch is reused across multiple optimizer steps. Write each
        # newly generated rollout once; duplicate JSON I/O would otherwise leave
        # the GPUs idle and inflate the audit by the reuse factor.
        if self.audit_handle is None or int(call_count) == self.last_audit_call_count:
            return
        for detail in details:
            row = {
                "step": int(step),
                "reward_call_index": int(call_count),
                "rank": int(RANK),
                **detail,
            }
            self.audit_handle.write(json.dumps(row, default=float) + "\n")
        self.last_audit_step = step
        self.last_audit_call_count = int(call_count)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        logs = logs or {}
        step = int(state.global_step)
        details = self.state_dict.get("details", [])
        call_count = int(self.state_dict.get("call_count", 0))
        new_reward_batch = bool(details) and call_count != self.last_reward_log_call_count

        # Every rank writes each newly generated rollout exactly once.
        if details:
            self._write_audit(step, details, call_count)

        if not state.is_world_process_zero or self.writer is None:
            return

        outcome_rates = {label: 0.0 for label in ("allowed", "dismissed", "unknown")}
        if new_reward_batch:
            for key in self.reward_keys:
                values = [float(d[key]) for d in details if key in d]
                if values:
                    self.writer.add_scalar(f"firac/{key}", sum(values) / len(values), step)

            outcomes = [str(d.get("generated_outcome") or "unknown") for d in details]
            references = [str(d.get("reference_outcome") or "unknown") for d in details]
            for label in outcome_rates:
                outcome_rates[label] = sum(value == label for value in outcomes) / len(outcomes)
                self.writer.add_scalar(f"outcome_distribution/{label}", outcome_rates[label], step)

            signed_gap = outcome_rates["dismissed"] - outcome_rates["allowed"]
            self.writer.add_scalar("outcome_distribution/dismissed_minus_allowed", signed_gap, step)
            self.writer.add_scalar("outcome_distribution/absolute_allowed_dismissed_gap", abs(signed_gap), step)

            valid_rate = sum(bool(d.get("format_valid", False)) for d in details) / len(details)
            self.writer.add_scalar("firac/format_valid_rate", valid_rate, step)

            totals = [float(d["R_total"]) for d in details]
            mean_total = sum(totals) / len(totals)
            variance = sum((t - mean_total) ** 2 for t in totals) / len(totals)
            self.writer.add_scalar("firac/reward_batch_std", variance ** 0.5, step)
            self.writer.add_scalar(
                "firac/zero_reward_fraction",
                sum(t == 0.0 for t in totals) / len(totals),
                step,
            )

            for generated_label in outcome_rates:
                group = [
                    d for d in details
                    if str(d.get("generated_outcome") or "unknown") == generated_label
                ]
                if not group:
                    continue
                for key in (
                    "R_total", "R_outcome", "R_issues", "R_rules",
                    "R_authority", "R_reasoning", "R_format_soft",
                ):
                    values = [float(d[key]) for d in group if key in d]
                    if values:
                        self.writer.add_scalar(
                            f"reward_by_generated_outcome/{generated_label}/{key}",
                            sum(values) / len(values), step,
                        )

            for reference_label in ("allowed", "dismissed"):
                group = [
                    d for d in details
                    if str(d.get("reference_outcome") or "unknown") == reference_label
                ]
                if group:
                    recall = sum(
                        str(d.get("generated_outcome") or "unknown") == reference_label
                        for d in group
                    ) / len(group)
                    self.writer.add_scalar(
                        f"batch_recall_by_reference/{reference_label}", recall, step
                    )

            for reference_label in ("allowed", "dismissed", "unknown"):
                for generated_label in ("allowed", "dismissed", "unknown"):
                    rate = sum(
                        r == reference_label and g == generated_label
                        for r, g in zip(references, outcomes)
                    ) / len(outcomes)
                    self.writer.add_scalar(
                        f"batch_confusion/{reference_label}_to_{generated_label}", rate, step
                    )

            self.last_reward_log_call_count = call_count

        for key, value in self.state_dict.get("timings", {}).items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"performance/reward_{key}", value, step)

        if model is not None and step % self.parameter_log_steps == 0:
            self._log_parameter_state(
                model,
                step,
                include_histograms=(step % self.histogram_log_steps == 0),
            )

        now = time.time()
        completed = step - self.previous_step
        if completed > 0 and self.previous_time is not None:
            seconds_per_step = (now - self.previous_time) / completed
            self.writer.add_scalar("performance/seconds_per_step", seconds_per_step, step)
            remaining = max(0, int(state.max_steps) - step)
            hours_left = remaining * seconds_per_step / 3600.0
            self.writer.add_scalar("performance/estimated_hours_remaining", hours_left, step)
            elapsed = (now - self.start_time) / 3600.0
            print(
                f"[step {step}/{state.max_steps}] {seconds_per_step:.1f}s/step | "
                f"elapsed {elapsed:.2f}h | ETA {hours_left:.2f}h",
                flush=True,
            )

        self.writer.add_scalar(
            "gpu/allocated_gib", torch.cuda.memory_allocated() / 1024 ** 3, step
        )
        self.writer.add_scalar(
            "gpu/reserved_gib", torch.cuda.memory_reserved() / 1024 ** 3, step
        )
        self.writer.flush()
        self.previous_time = now
        self.previous_step = step

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and self.writer is not None:
            if model is not None:
                self._log_parameter_state(
                    model, int(state.global_step), include_histograms=True
                )
            self.writer.flush()
            self.writer.close()
        if self.parameter_handle is not None:
            self.parameter_handle.flush()
            self.parameter_handle.close()
        if self.audit_handle is not None:
            self.audit_handle.flush()
            self.audit_handle.close()


# =============================================================================
# Trainer
# =============================================================================

# CRITICAL FIX FROM V1.5:
# Do NOT subclass GRPOTrainer._get_train_sampler. TRL 0.23.1 must retain its
# native RepeatSampler so each prompt is repeated exactly num_generations times
# before group-relative advantages are computed.

trainer_kwargs = {
    "model": policy_model,
    "args": grpo_args,
    "reward_funcs": [firac_reward],
    "train_dataset": grpo_train,
    "callbacks": [FIRACRCAMonitoringCallback(
        args.logging_dir,
        reward_state,
        audit_jsonl=args.audit_jsonl,
        parameter_stats_jsonl=args.parameter_stats_jsonl,
        parameter_log_steps=args.parameter_log_steps,
        histogram_log_steps=args.histogram_log_steps,
    )],
}

trainer_parameters = inspect.signature(GRPOTrainer.__init__).parameters

if "processing_class" in trainer_parameters:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in trainer_parameters:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = GRPOTrainer(**trainer_kwargs)

# -------------------------------------------------------------------------
# HARD SAMPLER PREFLIGHT -- catches the exact V1.5 bug before training.
# -------------------------------------------------------------------------
import itertools  # noqa: E402

sampler_probe = trainer._get_train_sampler(grpo_train)
sampler_name = type(sampler_probe).__name__
if sampler_name != "RepeatSampler":
    raise RuntimeError(
        f"Expected TRL native RepeatSampler, found {sampler_name}. "
        "Do not override GRPOTrainer._get_train_sampler."
    )

generation_batch_size = int(grpo_args.generation_batch_size)
source_prompts_per_generation_batch = generation_batch_size // args.num_generations
if source_prompts_per_generation_batch != ROLLOUT_PROMPTS:
    raise RuntimeError(
        f"This high-utilisation profile expects {ROLLOUT_PROMPTS} unique prompts per rollout "
        f"batch; got {source_prompts_per_generation_batch}."
    )
probe_indices = list(itertools.islice(iter(sampler_probe), generation_batch_size))
if len(probe_indices) != generation_batch_size:
    raise RuntimeError(
        f"Sampler returned only {len(probe_indices)} indices for a "
        f"{generation_batch_size}-completion generation batch."
    )
probe_case_ids = [str(grpo_train[int(i)]["case_id"]) for i in probe_indices]
probe_labels = [fc.normalise_label(grpo_train[int(i)]["outcome_label"]) for i in probe_indices]
probe_counts = Counter(probe_case_ids)
if len(probe_counts) != source_prompts_per_generation_batch:
    raise RuntimeError(
        "GRPO sampler grouping invalid: expected "
        f"{source_prompts_per_generation_batch} unique prompts, found "
        f"{len(probe_counts)}. This is the failure that produced the old 14-step run."
    )
if set(probe_counts.values()) != {args.num_generations}:
    raise RuntimeError(
        "GRPO sampler grouping invalid: every prompt must occur exactly "
        f"{args.num_generations} times; observed counts={sorted(set(probe_counts.values()))}."
    )
probe_label_counts = Counter(probe_labels)
if probe_label_counts != Counter({"allowed": generation_batch_size // 2, "dismissed": generation_batch_size // 2}):
    raise RuntimeError(
        f"First rollout batch is not exactly balanced: {dict(probe_label_counts)}"
    )
print(
    f"[rank {RANK}] GRPO_REPEAT_SAMPLER_PREFLIGHT_PASS "
    f"sampler={sampler_name} generation_batch={generation_batch_size} "
    f"unique_prompts={len(probe_counts)} repetitions={args.num_generations} "
    f"labels={dict(probe_label_counts)}",
    flush=True,
)

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
main_print("FIRAC BINARY GRPO -- 10K UNIQUE -- SOFT-GATE RCA AFTER SFT-10K")
main_print("=" * 78)
main_print(f"  world size            : {WORLD_SIZE}")
main_print(f"  unique training cases : {cache_unique_ids:,} (5,000 ALLOWED + 5,000 DISMISSED)")
main_print(f"  schedule exposures    : {len(grpo_train):,} (8 balanced padding exposures)")
main_print(f"  generations / prompt  : {args.num_generations}")
main_print(f"  per-device batch      : {args.per_device_batch} completions")
main_print(f"  gradient accumulation : {args.grad_accum}")
main_print(f"  optimizer compl. batch: {OPTIMIZER_COMPLETION_BATCH}")
main_print(f"  prompts / opt. step   : {PROMPTS_PER_OPTIMIZER_STEP}")
main_print(f"  rollout gen. batch    : {ROLLOUT_GENERATION_BATCH}")
main_print(f"  prompts / rollout     : {ROLLOUT_PROMPTS}")
main_print(f"  steps / generation    : {EXPECTED_STEPS_PER_GENERATION}")
main_print(f"  opt. steps / rollout  : {OPTIMIZER_STEPS_PER_ROLLOUT}")
main_print(f"  rollout calls         : {EXPECTED_ROLLOUT_CALLS}")
main_print(f"  optimiser steps       : {APPROX_STEPS:,}")
main_print(f"  total completions     : "
           f"{len(grpo_train) * args.num_generations * args.epochs:,.0f}")
main_print("  OTHER references      : 0 (hard invariant)")
main_print("  sampler               : TRL native RepeatSampler (preflight passed)")
main_print(f"  max prompt / compl.   : {args.max_prompt_length} / "
           f"{args.max_completion_length}")
main_print(f"  precision             : {compute_dtype} (no 4-bit)")
main_print(f"  vLLM generation       : {USE_VLLM}")
main_print(f"  gradient checkpoint   : {args.gradient_checkpointing}")
main_print(f"  learning rate         : {args.learning_rate}")
main_print(f"  beta (KL)             : {args.beta}")
main_print("  KL reference          : frozen copy of initial SFT-10K adapter")
main_print(f"  PEFT version          : {peft.__version__}")
main_print(f"  ref tensor max diff   : {reference_tensor_max_difference:.8f}")
main_print(f"  ref logit diagnostic  : {reference_max_logit_difference:.8f}")
main_print(f"  temperature           : {args.temperature}")
main_print(f"  checkpoint interval   : {args.save_steps}")
main_print(f"  warmup ratio          : {args.warmup_ratio}")
main_print(f"  max grad norm         : {args.max_grad_norm}")
main_print(f"  parameter log steps   : {args.parameter_log_steps}")
main_print(f"  histogram log steps   : {args.histogram_log_steps}")
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

# HARD FULL-RUN GUARD. The prior broken sampler silently ended at 14 steps.
# A run is not allowed to become the final adapter unless every expected
# optimizer step completed.
actual_steps = int(trainer.state.global_step)
if actual_steps != APPROX_STEPS:
    failure = {
        "status": "INCOMPLETE_DO_NOT_USE_AS_FINAL",
        "expected_steps": int(APPROX_STEPS),
        "actual_steps": actual_steps,
        "sampler": sampler_name,
        "generation_batch_size": generation_batch_size,
        "unique_training_cases": cache_unique_ids,
        "schedule_exposures": len(grpo_train),
    }
    if IS_MAIN:
        failure_path = Path(args.output_dir).parent / "FAILED_INCOMPLETE_RUN.json"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
    raise RuntimeError(
        f"Incomplete GRPO run: expected {APPROX_STEPS} optimizer steps, "
        f"got {actual_steps}. Final adapter will NOT be saved."
    )

# -------------------------------------------------------------------------
# Distributed-safe final save
#
# The previous implementation let non-main ranks enter an NCCL barrier while
# rank 0 serialised the adapter to disk. On slow shared storage, that barrier
# could exceed the 10-minute NCCL watchdog timeout even though training had
# completed successfully. Synchronise once before saving, unwrap the PEFT
# model, tear down NCCL on every rank, and then let rank 0 perform filesystem
# I/O without leaving the other ranks blocked in a collective.
# -------------------------------------------------------------------------
barrier()

unwrapped_model = (
    trainer.accelerator.unwrap_model(trainer.model)
    if IS_MAIN
    else None
)

if WORLD_SIZE > 1 and dist.is_initialized():
    dist.destroy_process_group()

if IS_MAIN:
    final_adapter = Path(args.final_adapter)
    final_adapter.mkdir(parents=True, exist_ok=True)

    if unwrapped_model is None:
        raise RuntimeError("Rank 0 could not unwrap the trained PEFT model.")

    save_kwargs = {"safe_serialization": True}
    if "selected_adapters" in inspect.signature(unwrapped_model.save_pretrained).parameters:
        save_kwargs["selected_adapters"] = ["default"]
    unwrapped_model.save_pretrained(str(final_adapter), **save_kwargs)
    tokenizer.save_pretrained(str(final_adapter))

    summary = {
        "world_size": WORLD_SIZE,
        "unique_training_cases": cache_unique_ids,
        "schedule_exposures": len(grpo_train),
        "padding_exposures": padding_count,
        "training_class_counts": dict(cache_counts),
        "num_generations": args.num_generations,
        "per_device_batch": args.per_device_batch,
        "gradient_accumulation": args.grad_accum,
        "optimizer_completion_batch": OPTIMIZER_COMPLETION_BATCH,
        "prompts_per_optimizer_step": PROMPTS_PER_OPTIMIZER_STEP,
        "rollout_generation_batch": ROLLOUT_GENERATION_BATCH,
        "rollout_prompts": ROLLOUT_PROMPTS,
        "steps_per_generation": EXPECTED_STEPS_PER_GENERATION,
        "optimizer_steps_per_rollout": OPTIMIZER_STEPS_PER_ROLLOUT,
        "expected_rollout_calls": EXPECTED_ROLLOUT_CALLS,
        "approximate_steps": APPROX_STEPS,
        "actual_steps": actual_steps,
        "early_stopping": False,
        "full_epoch_requested": True,
        "binary_only": True,
        "other_reference_count": 0,
        "sampler": sampler_name,
        "sampler_preflight_passed": True,
        "generation_batch_size": generation_batch_size,
        "source_prompts_per_generation_batch": source_prompts_per_generation_batch,
        "high_utilisation_profile": True,
        "seconds_per_step": (
            elapsed_hours * 3600.0 / max(1, trainer.state.global_step)
        ),
        "elapsed_hours": elapsed_hours,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "kl_reference": "frozen_initial_sft10k_adapter_loaded_from_disk",
        "reference_probe_max_logit_difference": reference_max_logit_difference,
        "reference_repeat_max_logit_difference": reference_repeat_max_logit_difference,
        "reference_tensor_max_difference": reference_tensor_max_difference,
        "reference_adapter_tensors": copied_reference_tensors,
        "peft_version": peft.__version__,
        "temperature": args.temperature,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "parameter_log_steps": args.parameter_log_steps,
        "histogram_log_steps": args.histogram_log_steps,
        "reward_weights": fc.REWARD_WEIGHTS,
        "reward_scheme": getattr(fc, "REWARD_SCHEME", "unknown"),
        "effective_reward_weights": getattr(fc, "EFFECTIVE_REWARD_WEIGHTS", None),
        "format_weight": fc.FORMAT_WEIGHT,
        "reasoning_weight": fc.REASONING_WEIGHT,
        "use_vllm": USE_VLLM,
        "vllm_memory_utilization": args.vllm_memory,
        "semantic_batch": args.semantic_batch,
        "reference_cache_device": str(getattr(scorer, "device", "unknown")),
        "dataloader_workers": args.dataloader_workers,
        "throughput_profile": (f"R1_outcome35_microbatch{args.per_device_batch}_accum{args.grad_accum}_"f"effective{OPTIMIZER_COMPLETION_BATCH}_rollout{ROLLOUT_GENERATION_BATCH}_"f"vllm{args.vllm_memory}_semantic{args.semantic_batch}_gc{int(args.gradient_checkpointing)}"),
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
