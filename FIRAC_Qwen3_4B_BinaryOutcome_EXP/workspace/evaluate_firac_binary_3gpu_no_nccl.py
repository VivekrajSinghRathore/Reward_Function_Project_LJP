"""Three-GPU independent-shard evaluation for the binary FIRAC experiment.

No NCCL process group is initialized. Each torchrun rank evaluates a disjoint
contiguous shard, writes a rank-local file, and rank 0 merges via shared disk.
The same binary soft-gated reward used for GRPO is used for SFT and GRPO
scoring. References are hard-checked to contain only ALLOWED/DISMISSED.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import firac_binary_soft_rca as fc

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", required=True)
parser.add_argument("--cache-dir", default="./firac_grpo_binary10k_rca/prepared")
parser.add_argument("--output-dir", default="./firac_grpo_binary10k_rca/results")
parser.add_argument("--tag", default="binary_grpo")
parser.add_argument("--limit", type=int, default=0, help="0 = full split")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--max-new-tokens", type=int, default=640)
parser.add_argument("--semantic-batch-size", type=int, default=64)
parser.add_argument("--max-prompt-length", type=int, default=1024)
args = parser.parse_args()

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this evaluator.")
if torch.cuda.device_count() < WORLD_SIZE:
    raise RuntimeError(f"WORLD_SIZE={WORLD_SIZE}, visible GPUs={torch.cuda.device_count()}")
torch.cuda.set_device(LOCAL_RANK)
IS_MAIN = RANK == 0


def main_print(*values, **kwargs):
    if IS_MAIN:
        print(*values, **kwargs, flush=True)


from datasets import load_from_disk  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

# PEFT 0.19.1 compatibility: same guarded no-TP patch used by training.
import peft  # noqa: E402
from peft import PeftModel  # noqa: E402
import transformers  # noqa: E402
import datasets as datasets_pkg  # noqa: E402
import peft.tuners.lora.awq as peft_lora_awq  # noqa: E402
peft_lora_awq.is_auto_awq_available = lambda: False

EXPECTED = {"transformers": "4.56.2", "peft": "0.19.1", "datasets": "4.0.0"}
ACTUAL = {
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "datasets": datasets_pkg.__version__,
}
if ACTUAL != EXPECTED:
    raise RuntimeError(f"Pinned evaluator environment mismatch: expected {EXPECTED}, got {ACTUAL}")


def install_peft_ddp_tp_noop_if_safe(model):
    """Apply the exact V1.5 PEFT 0.19.1 DDP/no-TP compatibility guard."""
    import peft.utils.save_and_load as peft_save_and_load

    def active_lora_tp_layers(candidate_model):
        active = []
        for name, module in candidate_model.named_modules():
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
        return "not_needed"
    except ImportError:
        pass

    def skip_tp_sharding_for_non_tp_ddp(peft_model, state_dict, adapter_name):
        active = active_lora_tp_layers(peft_model)
        if active:
            raise RuntimeError(
                "Active tensor parallelism detected; refusing the DDP-only PEFT compatibility patch."
            )
        return None

    peft_save_and_load._maybe_shard_state_dict_for_tp = skip_tp_sharding_for_non_tp_ddp
    return "patched_ddp_no_tp"


eval_path = Path(args.cache_dir) / "heldout_eval_binary"
if not eval_path.exists():
    raise FileNotFoundError(
        f"{eval_path} not found. Run the notebook binary data-preparation cell first."
    )
heldout = load_from_disk(str(eval_path))
if len(heldout) != fc.BINARY_HELDOUT_ROWS and args.limit == 0:
    raise RuntimeError(f"Expected {fc.BINARY_HELDOUT_ROWS} binary heldout rows, got {len(heldout)}")
if any(fc.normalise_label(x) not in fc.BINARY_LABELS for x in heldout["outcome_label"]):
    raise RuntimeError("OTHER/non-binary reference found in binary heldout cache.")
if args.limit > 0:
    heldout = heldout.select(range(min(args.limit, len(heldout))))

shard = heldout.shard(num_shards=WORLD_SIZE, index=RANK, contiguous=True)
main_print(f"Evaluating {len(heldout):,} binary rows across {WORLD_SIZE} independent GPU ranks.")

tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
load_kwargs = {
    "device_map": {"": LOCAL_RANK},
    "low_cpu_mem_usage": True,
    "attn_implementation": "sdpa",
}
try:
    base_model = AutoModelForCausalLM.from_pretrained(fc.MODEL_NAME, dtype=compute_dtype, **load_kwargs)
except (TypeError, ValueError) as error:
    if "dtype" not in str(error):
        raise
    base_model = AutoModelForCausalLM.from_pretrained(fc.MODEL_NAME, torch_dtype=compute_dtype, **load_kwargs)

patch_status = install_peft_ddp_tp_noop_if_safe(base_model)
print(f"[rank {RANK}] PEFT_DDP_TP_PATCH={patch_status}", flush=True)

# Accept only inactive forward-compatible PEFT fields seen in the SFT checkpoint.
import warnings
adapter_config_path = Path(args.adapter) / "adapter_config.json"
if adapter_config_path.exists():
    cfg_json = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    forward_only = {
        key: cfg_json.get(key) for key in ("monteclora_config", "velora_config")
        if key in cfg_json
    }
    active_unsupported = {
        key: value for key, value in forward_only.items()
        if value not in (None, False, {}, [])
    }
    if active_unsupported:
        raise RuntimeError(f"Unsupported active PEFT config fields: {active_unsupported}")
    if forward_only:
        warnings.filterwarnings(
            "ignore",
            message=r"Unexpected keyword arguments .*(monteclora_config|velora_config).*for class LoraConfig.*",
            category=UserWarning,
            module=r"peft\.config",
        )

model = PeftModel.from_pretrained(base_model, args.adapter)
model.eval()
model.config.use_cache = True

scorer = fc.SemanticScorer(device=f"cuda:{LOCAL_RANK}", batch_size=args.semantic_batch_size)
scorer.load()
scorer.precompute_references(fc.collect_reference_strings(shard), verbose=IS_MAIN)
reward_state = {}
reward_fn = fc.build_firac_reward(scorer, reward_state)

records = []
start = time.time()
for begin in range(0, len(shard), args.batch_size):
    batch = shard.select(range(begin, min(begin + args.batch_size, len(shard))))
    prompts = list(batch["prompt"])
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        add_special_tokens=False,
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_length = encoded["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    lengths = [int((row != tokenizer.pad_token_id).sum().item()) for row in new_tokens]

    rewards = reward_fn(
        completions=completions,
        reference_issues=list(batch["reference_issues"]),
        reference_rules=list(batch["reference_rules"]),
        reference_authorities=list(batch["reference_authorities"]),
        outcome_label=list(batch["outcome_label"]),
        case_id=list(batch["case_id"]),
        is_padding=[False] * len(batch),
    )

    for i, (completion, reward) in enumerate(zip(completions, rewards)):
        detail = reward_state["details"][i]
        parsed = fc.parse_firac_sections(completion)
        predicted = fc.extract_binary_outcome_from_conclusion(
            parsed["sections"].get("conclusion", "")
        )
        reference = fc.normalise_label(batch["outcome_label"][i])
        if reference not in fc.BINARY_LABELS:
            raise RuntimeError(f"Non-binary evaluation reference: {reference}")

        records.append({
            "case_id": str(batch["case_id"][i]),
            "input_prompt": prompts[i],
            "generated_output": completion,
            "parsed_conclusion": parsed["sections"].get("conclusion", ""),
            "present_headings": parsed["present_headings"],
            "reward": float(reward),
            "R_format_soft": detail["R_format_soft"],
            "R_format": detail["R_format"],
            "R_structure": detail["R_structure"],
            "R_coverage": detail["R_coverage"],
            "R_outcome": detail["R_outcome"],
            "R_issues": detail["R_issues"],
            "R_rules": detail["R_rules"],
            "R_authority": detail["R_authority"],
            "R_reasoning": detail["R_reasoning"],
            "predicted_outcome": predicted or "unknown",
            "reference_outcome": reference,
            "format_valid": bool(detail.get("format_valid", False)),
            "all_sections_present": bool(parsed["all_present"]),
            "truncated": lengths[i] >= args.max_new_tokens,
            "completion_tokens": lengths[i],
        })

    if IS_MAIN and (begin // args.batch_size) % 10 == 0:
        done = begin + len(batch)
        rate = done / max(1e-9, time.time() - start)
        main_print(f"  rank0 {done}/{len(shard)} ({rate:.2f} rows/s)")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
rank_path = output_dir / f".{args.tag}.rank{RANK}.json"
tmp = output_dir / f".{args.tag}.rank{RANK}.{uuid.uuid4().hex}.tmp"
with tmp.open("w", encoding="utf-8") as handle:
    json.dump(records, handle, ensure_ascii=False)
tmp.replace(rank_path)
print(f"[rank {RANK}] wrote {len(records):,} rows to {rank_path}", flush=True)

if not IS_MAIN:
    raise SystemExit(0)

deadline = time.time() + 8 * 3600
expected_paths = [output_dir / f".{args.tag}.rank{i}.json" for i in range(WORLD_SIZE)]
while not all(path.exists() for path in expected_paths):
    if time.time() > deadline:
        missing = [str(path) for path in expected_paths if not path.exists()]
        raise TimeoutError(f"Timed out waiting for evaluation rank files: {missing}")
    time.sleep(5)

records = []
for part_path in expected_paths:
    with part_path.open("r", encoding="utf-8") as handle:
        records.extend(json.load(handle))
    part_path.unlink(missing_ok=True)

n = len(records)
if args.limit == 0 and n != fc.BINARY_HELDOUT_ROWS:
    raise RuntimeError(f"Merged evaluation expected {fc.BINARY_HELDOUT_ROWS} rows, got {n}")
if len({r["case_id"] for r in records}) != n:
    raise RuntimeError("Duplicate case IDs found after evaluation shard merge.")


def mean(key):
    return sum(float(r[key]) for r in records) / max(1, n)


target_labels = ["allowed", "dismissed"]
prediction_labels = ["allowed", "dismissed", "unknown"]
y_true = [r["reference_outcome"] for r in records]
y_pred = [r["predicted_outcome"] for r in records]
if any(y not in target_labels for y in y_true):
    raise RuntimeError("Non-binary reference survived into final evaluation metrics.")

cm = confusion_matrix(y_true, y_pred, labels=prediction_labels)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=target_labels, zero_division=0,
)
per_class = {
    label: {
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
        "support": int(support[index]),
    }
    for index, label in enumerate(target_labels)
}
prediction_distribution = {
    label: int(sum(value == label for value in y_pred)) for label in prediction_labels
}

metrics = {
    "adapter": str(Path(args.adapter).resolve()),
    "tag": args.tag,
    "task": "binary_allowed_vs_dismissed",
    "rows": n,
    "world_size": WORLD_SIZE,
    "elapsed_hours_rank0": (time.time() - start) / 3600.0,
    "mean_R_total": mean("reward"),
    "mean_R_format_soft": mean("R_format_soft"),
    "mean_R_format": mean("R_format"),
    "mean_R_structure": mean("R_structure"),
    "mean_R_coverage": mean("R_coverage"),
    "mean_R_outcome": mean("R_outcome"),
    "mean_R_issues": mean("R_issues"),
    "mean_R_rules": mean("R_rules"),
    "mean_R_authority": mean("R_authority"),
    "mean_R_reasoning": mean("R_reasoning"),
    "outcome_accuracy": float(accuracy_score(y_true, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(y_true, y_pred, labels=target_labels, average="weighted", zero_division=0)),
    "format_completion_rate": sum(r["all_sections_present"] for r in records) / max(1, n),
    "format_valid_rate": sum(r["format_valid"] for r in records) / max(1, n),
    "unknown_rate": prediction_distribution["unknown"] / max(1, n),
    "truncation_rate": sum(r["truncated"] for r in records) / max(1, n),
    "mean_completion_tokens": mean("completion_tokens"),
    "prediction_distribution": prediction_distribution,
    "per_class": per_class,
    "confusion_labels": prediction_labels,
    "confusion_matrix": cm.tolist(),
}

with open(output_dir / f"metrics_{args.tag}.json", "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2, default=float)
with open(output_dir / f"per_case_{args.tag}.json", "w", encoding="utf-8") as handle:
    json.dump(records, handle, indent=2, default=float)

import pandas as pd  # noqa: E402
pd.DataFrame(records).to_csv(
    output_dir / f"per_case_{args.tag}.csv", index=False, encoding="utf-8-sig"
)

main_print("=" * 72)
main_print(f"BINARY EVALUATION: {args.tag}")
main_print("=" * 72)
for key, value in metrics.items():
    if isinstance(value, float):
        main_print(f"  {key:<28}: {value:.6f}")
    else:
        main_print(f"  {key:<28}: {value}")
