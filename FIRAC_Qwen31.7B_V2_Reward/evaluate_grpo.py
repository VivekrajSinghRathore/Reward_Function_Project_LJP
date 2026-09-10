"""
evaluate_grpo.py
================

Scores an adapter on the held-out 5K split with the same reward used for
training, plus outcome accuracy and a truncation rate.

    torchrun --standalone --nproc_per_node=3 evaluate_grpo.py --adapter ...

Generation here is greedy and single-sample, so this measures the deployed
behaviour rather than the training-time sampling distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import firac_common as fc

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", required=True)
parser.add_argument("--cache-dir", default="./firac_opt_v2/prepared")
parser.add_argument("--output-dir", default="./firac_opt_v2/results")
parser.add_argument("--tag", default="grpo")
parser.add_argument("--limit", type=int, default=0, help="0 = full split")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--max-new-tokens", type=int, default=640)
args = parser.parse_args()

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

torch.cuda.set_device(LOCAL_RANK)
if WORLD_SIZE > 1 and not dist.is_initialized():
    dist.init_process_group(backend="nccl")

IS_MAIN = RANK == 0


def main_print(*values, **kwargs):
    if IS_MAIN:
        print(*values, **kwargs, flush=True)


from datasets import load_from_disk  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402
from peft import PeftModel  # noqa: E402

eval_path = Path(args.cache_dir) / "heldout_eval"
if not eval_path.exists():
    raise FileNotFoundError(
        f"{eval_path} not found. Run train_grpo.py first, or run its dataset "
        "preparation step, to build the cached splits."
    )

heldout = load_from_disk(str(eval_path))
if args.limit > 0:
    heldout = heldout.select(range(min(args.limit, len(heldout))))

# Contiguous shard per rank; results are gathered at the end.
shard = heldout.shard(num_shards=WORLD_SIZE, index=RANK, contiguous=True)
main_print(f"Evaluating {len(heldout):,} rows across {WORLD_SIZE} ranks.")

tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

load_kwargs = {"device_map": {"": LOCAL_RANK}, "low_cpu_mem_usage": True}

try:
    base_model = AutoModelForCausalLM.from_pretrained(
        fc.MODEL_NAME, dtype=compute_dtype, **load_kwargs
    )
except (TypeError, ValueError) as error:
    if "dtype" not in str(error):
        raise
    base_model = AutoModelForCausalLM.from_pretrained(
        fc.MODEL_NAME, torch_dtype=compute_dtype, **load_kwargs
    )
model = PeftModel.from_pretrained(base_model, args.adapter)
model.eval()
model.config.use_cache = True

scorer = fc.SemanticScorer(device=f"cuda:{LOCAL_RANK}", batch_size=256)
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
        max_length=768,
        add_special_tokens=False,
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = encoded["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]

    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    lengths = [
        int((row != tokenizer.pad_token_id).sum().item()) for row in new_tokens
    ]

    rewards = reward_fn(
        completions=completions,
        reference_issues=list(batch["reference_issues"]),
        reference_rules=list(batch["reference_rules"]),
        reference_authorities=list(batch["reference_authorities"]),
        outcome_label=list(batch["outcome_label"]),
    )

    for i, (completion, reward) in enumerate(zip(completions, rewards)):
        detail = reward_state["details"][i]
        parsed = fc.parse_firac_sections(completion)
        predicted = fc.extract_outcome_from_conclusion(
            parsed["sections"].get("conclusion", "")
        )
        reference = fc.normalise_legal_text(batch["outcome_label"][i])

        records.append({
            "case_id": str(batch["case_id"][i])
                if "case_id" in batch.column_names else "",
            "input_prompt": prompts[i],
            "generated_output": completion,
            "parsed_conclusion": parsed["sections"].get("conclusion", ""),
            "present_headings": parsed["present_headings"],
            "reward": float(reward),
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
        main_print(f"  rank0 {done}/{len(shard)}  ({rate:.1f} rows/s)")

if WORLD_SIZE > 1:
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, records)
    records = [row for part in gathered for row in part]

if IS_MAIN:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)

    def mean(key):
        return sum(r[key] for r in records) / max(1, n)

    target_labels = ["allowed", "dismissed", "other"]
    prediction_labels = ["allowed", "dismissed", "other", "unknown"]
    y_true = [r["reference_outcome"] for r in records]
    y_pred = [r["predicted_outcome"] for r in records]

    cm = confusion_matrix(y_true, y_pred, labels=prediction_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=target_labels,
        zero_division=0,
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
        label: int(sum(value == label for value in y_pred))
        for label in prediction_labels
    }

    metrics = {
        "adapter": str(Path(args.adapter).resolve()),
        "tag": args.tag,
        "rows": n,
        "elapsed_hours": (time.time() - start) / 3600.0,
        "mean_R_total": mean("reward"),
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
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=target_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=target_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "format_completion_rate": sum(
            r["all_sections_present"] for r in records
        ) / max(1, n),
        "format_valid_rate": sum(
            r["format_valid"] for r in records
        ) / max(1, n),
        "unknown_rate": prediction_distribution["unknown"] / max(1, n),
        "truncation_rate": sum(r["truncated"] for r in records) / max(1, n),
        "mean_completion_tokens": mean("completion_tokens"),
        "prediction_distribution": prediction_distribution,
        "per_class": per_class,
        "confusion_labels": prediction_labels,
        "confusion_matrix": cm.tolist(),
    }

    with open(output_dir / f"metrics_{args.tag}.json", "w", encoding="utf-8") as h:
        json.dump(metrics, h, indent=2, default=float)

    with open(output_dir / f"per_case_{args.tag}.json", "w", encoding="utf-8") as h:
        json.dump(records, h, indent=2, default=float)

    import pandas as pd
    pd.DataFrame(records).to_csv(
        output_dir / f"per_case_{args.tag}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    main_print("=" * 60)
    main_print(f"EVALUATION: {args.tag}")
    main_print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            main_print(f"  {key:<26}: {value:.4f}")
        else:
            main_print(f"  {key:<26}: {value}")

if WORLD_SIZE > 1:
    dist.barrier()
    dist.destroy_process_group()
