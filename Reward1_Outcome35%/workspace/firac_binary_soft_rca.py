"""Binary (ALLOWED/DISMISSED) FIRAC soft-gated reward and split helpers.

This module intentionally reuses the validated parsing, semantic-scoring, prompt
construction helpers, and reference extraction from ``firac_common.py`` while
changing only what must change for the binary experiment:

* training/evaluation references are ALLOWED or DISMISSED only;
* the system prompt permits exactly ALLOWED or DISMISSED in Conclusion;
* a generated OTHER is treated as UNKNOWN/invalid for the binary task;
* the soft-gated reward remains 15% soft-format + 85% substantive reasoning;
* the 10,000 unique binary training set preserves the historical held-out
  ALLOWED/DISMISSED cases and adds new training examples after that held-out
  slice, preventing leakage;
* the physical GRPO schedule is padded from 10,000 to 10,008 rows (4 ALLOWED +
  4 DISMISSED repeats). V2.1 rolls out 24 prompts x 8 generations = 192
  completions at once and reuses that rollout across two optimizer steps; the
  unique training set remains exactly 10,000 cases (5,000/5,000).
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any, Dict, Optional

import numpy as np
import firac_common as base

MODEL_NAME = base.MODEL_NAME
EMBEDDING_MODEL_NAME = base.EMBEDDING_MODEL_NAME
SEMANTIC_THRESHOLD = base.SEMANTIC_THRESHOLD
DATASET_NAME = base.DATASET_NAME
DATASET_CONFIG = base.DATASET_CONFIG
SPLIT_SEED = base.SPLIT_SEED
SFT_SEED = base.SFT_SEED

BINARY_LABELS = {"allowed", "dismissed"}
BINARY_TRAIN_COUNTS = {"allowed": 5000, "dismissed": 5000}
BINARY_EVAL_COUNTS = {"allowed": 2250, "dismissed": 2250}
HISTORICAL_TRAIN_PER_CLASS = 4500
HISTORICAL_EVAL_PER_CLASS = 2250
EXTRA_TRAIN_PER_CLASS = 500
PAD_PER_CLASS = 4
UNIQUE_TRAIN_CASES = 10_000
PADDED_SCHEDULE_ROWS = 10_008
BINARY_HELDOUT_ROWS = 4_500

# Re-export stable helpers.
normalise_label = base.normalise_label
normalise_legal_text = base.normalise_legal_text
normalise_completion = base.normalise_completion
parse_firac_sections = base.parse_firac_sections
dedupe_items = base.dedupe_items
split_section_into_items = base.split_section_into_items
get_batch_value = base.get_batch_value
collect_reference_strings = base.collect_reference_strings
compute_authority_reward = base.compute_authority_reward


class SemanticScorer(base.SemanticScorer):
    """GPU-resident reference cache for high-throughput GRPO reward scoring.

    The archived scorer deliberately moved fixed reference embeddings to CPU.
    That is memory-conservative, but during GRPO it forces thousands of tiny
    CPU->GPU copies inside semantic_f1. For BGE-small and this 10K binary split,
    the reference cache is small enough to remain on each rank's GPU, removing
    that transfer bottleneck without changing any similarity computation.
    """

    def precompute_references(self, texts, verbose=False):
        unique = [
            t for t in dict.fromkeys(t for t in texts if t)
            if t not in self._reference_cache
        ]
        if not unique:
            return
        if verbose:
            print(f"Pre-encoding {len(unique):,} unique reference strings on GPU...", flush=True)
        chunk = max(self.batch_size * 8, 1024)
        for start in range(0, len(unique), chunk):
            block = unique[start:start + chunk]
            embeddings = self._encode(block)
            for i, text in enumerate(block):
                self._reference_cache[text] = embeddings[i].detach()
        if verbose:
            print(
                f"Reference cache size: {len(self._reference_cache):,} "
                f"(device-resident on {self.device})",
                flush=True,
            )

    def reference_map(self, texts):
        unique = list(dict.fromkeys(t for t in texts if t))
        missing = [t for t in unique if t not in self._reference_cache]
        if missing:
            embeddings = self._encode(missing)
            for i, text in enumerate(missing):
                self._reference_cache[text] = embeddings[i].detach()
        return {t: self._reference_cache[t] for t in unique}

BINARY_SYSTEM_PROMPT = """
Analyse the Supreme Court of India appeal and write a concise FIRAC judgment.

Use exactly these headings once and in this order:

Facts:
Issues:
Rules:
Analysis:
Conclusion:

Limits:
- Facts: maximum 70 words.
- Issues: maximum 2 numbered issues.
- Rules: maximum 4 numbered rules.
- Analysis: maximum 100 words.
- Conclusion: exactly one word: ALLOWED or DISMISSED.
- Do not write OTHER.
- Do not add punctuation or explanation after the Conclusion outcome.
- Do not invent provisions, cases, witnesses, or citations.
- Complete all five sections before stopping.
""".strip()

# Same soft-gated design as the corrected V5/V1.5 experiment.
REWARD_WEIGHTS = {
    "outcome": 0.15,
    "issues": 0.25,
    "rules": 0.40,
    "authority": 0.20,
}
FORMAT_WEIGHT = 0.15
REASONING_WEIGHT = 0.85
assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-9
assert abs(FORMAT_WEIGHT + REASONING_WEIGHT - 1.0) < 1e-9


def extract_binary_outcome_from_conclusion(conclusion_text: Any) -> str:
    """Strict binary parser: only the entire one-word conclusion is accepted."""
    text = normalise_legal_text(conclusion_text)
    match = re.fullmatch(r"(allowed|dismissed)", text)
    return match.group(1) if match else ""


def compute_binary_outcome_reward(parsed: Dict[str, Any], reference_outcome: Any, return_details=False):
    conclusion_text = parsed.get("sections", {}).get("conclusion", "")
    generated = extract_binary_outcome_from_conclusion(conclusion_text)
    reference = normalise_legal_text(reference_outcome)
    reward = float(reference in BINARY_LABELS and generated == reference)
    if not return_details:
        return reward
    return {
        "generated_outcome": generated,
        "reference_outcome": reference,
        "conclusion_text": conclusion_text,
        "reward": reward,
    }


def compute_binary_coverage_reward(parsed: Dict[str, Any]) -> float:
    """Coverage identical to the archived scorer except Conclusion is binary-only."""
    sections = parsed["sections"]
    issues = sections.get("issues", "")
    rules = sections.get("rules", "")
    analysis = sections.get("analysis", "")
    conclusion = sections.get("conclusion", "")

    issues_score = float(
        len(issues.split()) >= 6
        and ("whether" in issues.lower() or "?" in issues or base.sentence_count(issues) >= 1)
    )
    rules_score = float(len(rules.split()) >= 8 and bool(base.LEGAL_TERM_PATTERN.search(rules)))
    analysis_score = (
        0.40 * min(len(analysis.split()) / 80.0, 1.0)
        + 0.30 * min(base.sentence_count(analysis) / 3.0, 1.0)
        + 0.30 * float(base.has_reasoning_connector(analysis))
    )
    generated_outcome = extract_binary_outcome_from_conclusion(conclusion)
    conclusion_score = float(generated_outcome in BINARY_LABELS)
    connected_score = float(len(analysis.split()) >= 20 and generated_outcome in BINARY_LABELS)

    reward = (
        0.20 * issues_score
        + 0.20 * rules_score
        + 0.30 * analysis_score
        + 0.20 * conclusion_score
        + 0.10 * connected_score
    )
    return float(np.clip(reward, 0.0, 1.0))


def compute_binary_format_components(completion: Any) -> Dict[str, Any]:
    parsed = parse_firac_sections(completion)
    r_structure = base.compute_structure_reward(parsed)
    r_coverage = compute_binary_coverage_reward(parsed)
    r_format_soft = 0.5 * r_structure + 0.5 * r_coverage

    sections = parsed["sections"]
    all_non_empty = all(bool(sections.get(h, "").strip()) for h in base.REQUIRED_HEADINGS)
    conclusion_valid = extract_binary_outcome_from_conclusion(sections.get("conclusion", "")) in BINARY_LABELS
    format_valid = bool(
        parsed["all_present"]
        and parsed["correct_order"]
        and parsed["no_duplicates"]
        and all_non_empty
        and conclusion_valid
    )
    # R_format is retained as a diagnostic hard-validity score. R_total uses
    # R_format_soft, so semantic signal is NOT hard-gated.
    r_format = r_format_soft if format_valid else 0.0
    return {
        "parsed": parsed,
        "format_valid": format_valid,
        "R_structure": float(np.clip(r_structure, 0.0, 1.0)),
        "R_coverage": float(np.clip(r_coverage, 0.0, 1.0)),
        "R_format_soft": float(np.clip(r_format_soft, 0.0, 1.0)),
        "R_format": float(np.clip(r_format, 0.0, 1.0)),
    }


def combine_rewards(r_format_soft, r_outcome, r_issues, r_rules, r_authority):
    r_reasoning = (
        REWARD_WEIGHTS["outcome"] * float(r_outcome)
        + REWARD_WEIGHTS["issues"] * float(r_issues)
        + REWARD_WEIGHTS["rules"] * float(r_rules)
        + REWARD_WEIGHTS["authority"] * float(r_authority)
    )
    r_reasoning = float(np.clip(r_reasoning, 0.0, 1.0))
    r_format_soft = float(np.clip(r_format_soft, 0.0, 1.0))
    r_total = float(np.clip(
        FORMAT_WEIGHT * r_format_soft + REASONING_WEIGHT * r_reasoning,
        0.0,
        1.0,
    ))
    return {
        "R_outcome": float(r_outcome),
        "R_issues": float(r_issues),
        "R_rules": float(r_rules),
        "R_authority": float(r_authority),
        "R_reasoning": r_reasoning,
        "R_total": r_total,
    }


def build_firac_reward(scorer: "SemanticScorer", state: Optional[dict] = None):
    """TRL-compatible binary soft-gated reward with completion audit records."""
    if state is None:
        state = {}
    state.setdefault("details", [])
    state.setdefault("timings", {})
    state.setdefault("call_count", 0)

    def firac_binary_reward(
        completions,
        reference_issues=None,
        reference_rules=None,
        reference_authorities=None,
        outcome_label=None,
        case_id=None,
        is_padding=None,
        **kwargs,
    ):
        start = time.perf_counter()
        texts = [normalise_completion(c) for c in completions]
        format_results = [compute_binary_format_components(t) for t in texts]
        parsed_items = [r["parsed"] for r in format_results]

        gen_issues, gen_rules = [], []
        ref_issues, ref_rules = [], []
        authorities, outcomes, case_ids, padding_flags = [], [], [], []

        for index, parsed in enumerate(parsed_items):
            sections = parsed.get("sections", {})
            gen_issues.append(dedupe_items(split_section_into_items(sections.get("issues", ""), "issues")))
            gen_rules.append(dedupe_items(split_section_into_items(sections.get("rules", ""), "rules")))
            ref_issues.append(dedupe_items(get_batch_value(reference_issues, index, []) or []))
            ref_rules.append(dedupe_items(get_batch_value(reference_rules, index, []) or []))
            authorities.append(get_batch_value(reference_authorities, index, []) or [])
            outcomes.append(get_batch_value(outcome_label, index, "") or "")
            case_ids.append(str(get_batch_value(case_id, index, "") or ""))
            padding_flags.append(bool(get_batch_value(is_padding, index, False)))

        # References are a hard invariant for this experiment.
        bad_references = [normalise_label(x) for x in outcomes if normalise_label(x) not in BINARY_LABELS]
        if bad_references:
            raise RuntimeError(f"Non-binary reference labels reached the reward: {bad_references[:5]}")

        parse_seconds = time.perf_counter() - start
        semantic_start = time.perf_counter()
        generated_texts = [t for items in gen_issues + gen_rules for t in items]
        reference_texts = [t for items in ref_issues + ref_rules for t in items]
        generated_map = scorer.encode_map(generated_texts)
        reference_map = scorer.reference_map(reference_texts)

        issue_scores = [
            scorer.semantic_f1(gen_issues[i], ref_issues[i], generated_map, reference_map)
            for i in range(len(texts))
        ]
        rule_scores = [
            scorer.semantic_f1(gen_rules[i], ref_rules[i], generated_map, reference_map)
            for i in range(len(texts))
        ]
        semantic_seconds = time.perf_counter() - semantic_start

        rewards, details_list = [], []
        for index, (text, format_result, parsed) in enumerate(zip(texts, format_results, parsed_items)):
            r_outcome = float(compute_binary_outcome_reward(parsed, outcomes[index]))
            r_authority = float(compute_authority_reward(parsed, authorities[index]))
            combined = combine_rewards(
                r_format_soft=float(format_result["R_format_soft"]),
                r_outcome=r_outcome,
                r_issues=float(issue_scores[index]),
                r_rules=float(rule_scores[index]),
                r_authority=r_authority,
            )
            rewards.append(combined["R_total"])
            outcome_details = compute_binary_outcome_reward(parsed, outcomes[index], return_details=True)
            record = {
                "case_id": case_ids[index],
                "is_padding": padding_flags[index],
                "R_structure": float(format_result["R_structure"]),
                "R_coverage": float(format_result["R_coverage"]),
                "R_format_soft": float(format_result["R_format_soft"]),
                "R_format": float(format_result["R_format"]),
                "format_valid": bool(format_result["format_valid"]),
                "generated_outcome": outcome_details["generated_outcome"],
                "reference_outcome": outcome_details["reference_outcome"],
                "completion_index": index,
                "completion_chars": len(text),
                "completion_text": text[:6000],
            }
            record.update(combined)
            details_list.append(record)

        state["details"] = details_list
        state["timings"] = {
            "parse_seconds": parse_seconds,
            "semantic_seconds": semantic_seconds,
            "total_seconds": time.perf_counter() - start,
            "completion_count": len(texts),
            "reference_cache_size": scorer.reference_cache_size,
        }
        state["call_count"] += 1
        return rewards

    return firac_binary_reward


def make_binary_grpo_mapper(tokenizer):
    """Pre-render Qwen3 prompt with thinking disabled and binary instructions."""
    def _mapper(example):
        messages = [
            {"role": "system", "content": BINARY_SYSTEM_PROMPT},
            {"role": "user", "content": base.build_user_prompt(example)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        out = {"prompt": prompt_text}
        out.update(base.build_reference_fields(example))
        out["outcome_label"] = normalise_label(out["outcome_label"])
        out["case_id"] = str(example.get("case_id", ""))
        out["is_padding"] = bool(example.get("is_padding", False))
        return out
    return _mapper



def build_binary_short_sft_answer(example) -> Dict[str, str]:
    """Build the concise five-section SFT target with a strict binary conclusion.

    The Facts/Issues/Rules/Analysis construction is inherited from the validated
    FIRAC helper. The conclusion is then hard-checked to ALLOWED/DISMISSED so a
    malformed or OTHER target can never enter the fresh binary SFT run.
    """
    built = base.build_short_sft_answer(example)["short_sft_answer"]
    outcome = normalise_label(
        example.get("simplified_outcome_label")
        or (example.get("firac") or {}).get("conclusion", {}).get("simplified_label")
    )
    if outcome not in BINARY_LABELS:
        raise ValueError(f"Non-binary SFT target encountered: {outcome!r}")

    # The archived builder already emits a one-word conclusion. Rebuild the tail
    # anyway so the binary contract is explicit and cannot regress silently.
    parts = re.split(r"(?im)^Conclusion:\s*", built, maxsplit=1)
    if len(parts) != 2:
        raise RuntimeError("SFT target builder did not emit a Conclusion heading.")
    answer = parts[0].rstrip() + "\n\nConclusion:\n" + outcome.upper()
    return {"short_sft_answer": answer}


def make_binary_sft_mapper(tokenizer):
    """Return prompt/completion pairs for completion-only SFT.

    The prompt is byte-for-byte the same task instruction used for GRPO and
    Qwen3 thinking is disabled before rendering. The completion contains only
    the reference FIRAC answer plus EOS, allowing TRL completion_only_loss=True.
    """
    def _mapper(example):
        messages = [
            {"role": "system", "content": BINARY_SYSTEM_PROMPT},
            {"role": "user", "content": base.build_user_prompt(example)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        completion_text = build_binary_short_sft_answer(example)["short_sft_answer"]
        if tokenizer.eos_token and not completion_text.endswith(tokenizer.eos_token):
            completion_text += tokenizer.eos_token
        return {
            "prompt": prompt_text,
            "completion": completion_text,
            "case_id": str(example.get("case_id", "")),
            "outcome_label": normalise_label(example.get("simplified_outcome_label")),
            "is_padding": bool(example.get("is_padding", False)),
        }
    return _mapper


def build_binary_splits_preserving_historical_holdout(dataset, verify_historical=True):
    """Create 10K unique binary train + exact historical 4.5K binary heldout.

    Per class, the historical split was:
      0:4500     -> old training
      4500:6750  -> old held-out

    We keep that held-out untouched, and obtain 500 extra training cases from
    6750:7250. Thus no old held-out case is moved into binary training.
    """
    from datasets import concatenate_datasets

    raw_train = dataset["train"]
    train_parts, eval_parts = [], []
    labels = ["allowed", "dismissed"]

    for offset, label in enumerate(labels):
        pool = raw_train.filter(
            lambda ex, target=label: normalise_label(ex["simplified_outcome_label"]) == target
        ).shuffle(seed=SPLIT_SEED + offset)

        required = HISTORICAL_TRAIN_PER_CLASS + HISTORICAL_EVAL_PER_CLASS + EXTRA_TRAIN_PER_CLASS
        if len(pool) < required:
            raise ValueError(f"Not enough {label} cases: need {required}, found {len(pool)}")

        old_train = pool.select(range(0, HISTORICAL_TRAIN_PER_CLASS))
        heldout = pool.select(range(
            HISTORICAL_TRAIN_PER_CLASS,
            HISTORICAL_TRAIN_PER_CLASS + HISTORICAL_EVAL_PER_CLASS,
        ))
        extra_start = HISTORICAL_TRAIN_PER_CLASS + HISTORICAL_EVAL_PER_CLASS
        extra_train = pool.select(range(extra_start, extra_start + EXTRA_TRAIN_PER_CLASS))
        train_parts.append(concatenate_datasets([old_train, extra_train]))
        eval_parts.append(heldout)

    # Keep class pools deterministic; training schedule builder alternates them.
    binary_train = concatenate_datasets(train_parts)
    binary_eval = concatenate_datasets(eval_parts).shuffle(seed=SPLIT_SEED + 700)

    train_ids = set(map(str, binary_train["case_id"]))
    eval_ids = set(map(str, binary_eval["case_id"]))
    train_counts = Counter(normalise_label(x) for x in binary_train["simplified_outcome_label"])
    eval_counts = Counter(normalise_label(x) for x in binary_eval["simplified_outcome_label"])

    if len(binary_train) != UNIQUE_TRAIN_CASES or len(train_ids) != UNIQUE_TRAIN_CASES:
        raise RuntimeError("Binary training split is not exactly 10,000 unique cases.")
    if train_counts != Counter(BINARY_TRAIN_COUNTS):
        raise RuntimeError(f"Binary training counts wrong: {dict(train_counts)}")
    if len(binary_eval) != BINARY_HELDOUT_ROWS or len(eval_ids) != BINARY_HELDOUT_ROWS:
        raise RuntimeError("Binary held-out split is not exactly 4,500 unique cases.")
    if eval_counts != Counter(BINARY_EVAL_COUNTS):
        raise RuntimeError(f"Binary heldout counts wrong: {dict(eval_counts)}")
    if not train_ids.isdisjoint(eval_ids):
        raise RuntimeError("Binary train/heldout leakage detected.")

    if verify_historical:
        # Rebuild the archived split and prove that our binary held-out IDs are
        # exactly the ALLOWED/DISMISSED subset of the historical 5K heldout.
        _, archived_eval = base.build_balanced_splits(dataset)
        archived_binary_ids = {
            str(cid)
            for cid, label in zip(archived_eval["case_id"], archived_eval["simplified_outcome_label"])
            if normalise_label(label) in BINARY_LABELS
        }
        if eval_ids != archived_binary_ids:
            raise RuntimeError("Binary heldout no longer matches the historical heldout IDs.")

    return binary_train, binary_eval


def build_binary_training_schedule(binary_train):
    """Alternate 5K/5K classes and append 8 balanced padding exposures.

    TRL 0.23.1's native RepeatSampler is retained. The V2.1 high-utilisation
    profile uses microbatch=4, grad_accum=8 and an explicit global rollout batch
    of 192 completions = 24 unique prompts x 8 generations. The same 192
    completions are buffered across 16 microsteps = two optimizer steps.
    10,008 is divisible by both 24 rollout prompts and 12 prompts per optimizer
    step, while retaining exactly 10,000 unique case IDs.
    """
    from datasets import concatenate_datasets

    allowed = binary_train.filter(
        lambda ex: normalise_label(ex["simplified_outcome_label"]) == "allowed"
    )
    dismissed = binary_train.filter(
        lambda ex: normalise_label(ex["simplified_outcome_label"]) == "dismissed"
    )
    if len(allowed) != 5000 or len(dismissed) != 5000:
        raise RuntimeError("Expected exactly 5,000 examples per binary class.")

    # Deterministic within-class randomization, then strict alternation. This
    # makes every 24-prompt rollout exactly 12 ALLOWED / 12 DISMISSED.
    allowed = allowed.shuffle(seed=SPLIT_SEED + 801)
    dismissed = dismissed.shuffle(seed=SPLIT_SEED + 802)
    combined = concatenate_datasets([allowed, dismissed])
    alternating_indices = [idx for i in range(5000) for idx in (i, 5000 + i)]
    unique_schedule = combined.select(alternating_indices)
    unique_schedule = unique_schedule.add_column("is_padding", [False] * len(unique_schedule))

    tail_ids = set(map(str, unique_schedule["case_id"][-24:]))
    pad_a_idx = [i for i, cid in enumerate(map(str, allowed["case_id"])) if cid not in tail_ids][:PAD_PER_CLASS]
    pad_d_idx = [i for i, cid in enumerate(map(str, dismissed["case_id"])) if cid not in tail_ids][:PAD_PER_CLASS]
    if len(pad_a_idx) != PAD_PER_CLASS or len(pad_d_idx) != PAD_PER_CLASS:
        raise RuntimeError("Could not choose safe deterministic padding cases.")

    padding_order = [idx for ai, di in zip(pad_a_idx, pad_d_idx) for idx in (ai, 5000 + di)]
    padding = combined.select(padding_order).add_column("is_padding", [True] * (2 * PAD_PER_CLASS))
    schedule = concatenate_datasets([unique_schedule, padding])

    counts = Counter(normalise_label(x) for x in schedule["simplified_outcome_label"])
    unique_ids = len(set(map(str, schedule["case_id"])))
    padding_count = sum(bool(x) for x in schedule["is_padding"])
    if len(schedule) != PADDED_SCHEDULE_ROWS:
        raise RuntimeError(f"Expected {PADDED_SCHEDULE_ROWS} schedule rows, got {len(schedule)}")
    if unique_ids != UNIQUE_TRAIN_CASES:
        raise RuntimeError(f"Expected {UNIQUE_TRAIN_CASES} unique IDs, got {unique_ids}")
    if counts != Counter({"allowed": 5004, "dismissed": 5004}):
        raise RuntimeError(f"Unexpected padded schedule counts: {dict(counts)}")
    if padding_count != 8:
        raise RuntimeError(f"Expected 8 padding rows, got {padding_count}")

    # Verify every contiguous 24-row rollout source block is 12/12. This is
    # the order native RepeatSampler receives when shuffle_dataset=False.
    for start in range(0, len(schedule), 24):
        block = schedule.select(range(start, start + 24))
        block_counts = Counter(normalise_label(x) for x in block["simplified_outcome_label"])
        if block_counts != Counter({"allowed": 12, "dismissed": 12}):
            raise RuntimeError(f"Unbalanced 24-prompt rollout block at {start}: {dict(block_counts)}")
        if len(set(map(str, block["case_id"]))) != 24:
            raise RuntimeError(f"Duplicate case within one 24-prompt rollout block at {start}.")

    return schedule
