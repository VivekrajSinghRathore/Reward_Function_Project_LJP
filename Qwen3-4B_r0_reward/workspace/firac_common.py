"""
firac_common.py
===============

Shared definitions for the FIRAC GRPO pipeline.

This module is imported by BOTH the notebook and the torchrun training /
evaluation scripts. It replaces the previous `execute_cell(notebook, index)`
mechanism, which silently broke whenever a cell was inserted or removed.

Layout
------
Part A  Pure-python reward core. Imports only the standard library + numpy,
        so it can be unit-tested without a GPU, torch, or transformers.
Part B  Semantic scoring (BGE). Torch and sentence-transformers are imported
        lazily inside `SemanticScorer.load()`, never at module import time.
Part C  Dataset preparation and the balanced 10K / 5K split.

Nothing in this file has import-time side effects.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# =============================================================================
# PART A -- PURE PYTHON REWARD CORE (no torch)
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-4B"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SEMANTIC_THRESHOLD = 0.83

FIRAC_HEADINGS = ["Facts", "Issues", "Rules", "Analysis", "Conclusion"]
REQUIRED_HEADINGS = [h.lower() for h in FIRAC_HEADINGS]

REWARD_WEIGHTS = {
    "outcome": 0.40,
    "issues": 0.20,
    "rules": 0.25,
    "authority": 0.15,
}

assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-9

# The single system prompt used by BOTH SFT and GRPO. Any divergence between
# the two stages silently destroys the value of the SFT initialisation.
SHORT_SFT_SYSTEM_PROMPT = """
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
- Conclusion: exactly one word: ALLOWED, DISMISSED, or OTHER.
- Do not add punctuation or explanation after the Conclusion outcome.
- Do not invent provisions, cases, witnesses, or citations.
- Complete all five sections before stopping.
""".strip()

USER_PROMPT_SUFFIX = "\n\nWrite the FIRAC judgment."


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalise_legal_text(text: Any) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).casefold()
    text = re.sub(r"\bversus\b|\bvs\.?\b|\bv\.\b", " v ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_label(value: Any) -> str:
    return str(value or "").strip().lower()


def format_numbered_items(items: Optional[Iterable[Any]]) -> str:
    cleaned = [clean_text(item) for item in (items or []) if clean_text(item)]
    return "\n".join(f"{i}. {item}" for i, item in enumerate(cleaned, start=1))


def normalise_completion(completion: Any) -> str:
    """Accept a str, a message dict, or a list of message dicts."""
    if completion is None:
        return ""

    if isinstance(completion, str):
        return completion.strip()

    if isinstance(completion, dict):
        return str(completion.get("content", completion)).strip()

    if isinstance(completion, list):
        contents = []
        for item in completion:
            if isinstance(item, dict) and item.get("content"):
                contents.append(str(item["content"]))
            elif isinstance(item, str):
                contents.append(item)
        return "\n".join(contents).strip()

    return str(completion).strip()


# Qwen3 emits a reasoning block when the chat template is applied with the
# default `enable_thinking=True`. We disable thinking at render time, but a
# stray block in a resumed run must never be scored as FIRAC content.
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
DANGLING_THINK_PATTERN = re.compile(r"^.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = THINK_BLOCK_PATTERN.sub(" ", text)
    if "</think>" in text.lower():
        text = DANGLING_THINK_PATTERN.sub(" ", text)
    return text.strip()


# -----------------------------------------------------------------------------
# FIRAC parser (Markdown compatible)
# -----------------------------------------------------------------------------

HEADING_PATTERN = re.compile(
    r"(?im)^"
    r"[ \t]*"
    r"(?:\*\*|__)?"
    r"(Facts|Issues|Rules|Analysis|Conclusion)"
    r"(?:\*\*|__)?"
    r"[ \t]*:"
    r"[ \t]*"
    r"(?:\*\*|__)?"
    r"[ \t]*"
)


def parse_firac_sections(completion: Any) -> Dict[str, Any]:
    """Parse plain or Markdown FIRAC headings into sections."""
    text = strip_thinking(normalise_completion(completion))

    matches = list(HEADING_PATTERN.finditer(text))
    present_headings = [m.group(1).lower() for m in matches]

    heading_counts = {h: present_headings.count(h) for h in REQUIRED_HEADINGS}

    first_occurrence_order: List[str] = []
    for heading in present_headings:
        if heading not in first_occurrence_order:
            first_occurrence_order.append(heading)

    sections = {h: "" for h in REQUIRED_HEADINGS}

    for index, match in enumerate(matches):
        heading = match.group(1).lower()
        content_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        if not sections[heading]:
            sections[heading] = text[match.end():content_end].strip()

    return {
        "text": text,
        "sections": sections,
        "heading_counts": heading_counts,
        "present_headings": present_headings,
        "correct_order": first_occurrence_order == REQUIRED_HEADINGS,
        "all_present": all(heading_counts[h] >= 1 for h in REQUIRED_HEADINGS),
        "no_duplicates": all(heading_counts[h] == 1 for h in REQUIRED_HEADINGS),
    }


# -----------------------------------------------------------------------------
# Format rewards
# -----------------------------------------------------------------------------

def compute_structure_reward(parsed: Dict[str, Any]) -> float:
    counts = parsed["heading_counts"]
    sections = parsed["sections"]
    n = len(REQUIRED_HEADINGS)

    heading_presence = sum(counts.get(h, 0) >= 1 for h in REQUIRED_HEADINGS) / n
    order_score = float(parsed["correct_order"])
    uniqueness_score = sum(counts.get(h, 0) == 1 for h in REQUIRED_HEADINGS) / n
    non_empty_score = sum(
        bool(sections.get(h, "").strip()) for h in REQUIRED_HEADINGS
    ) / n

    reward = (
        0.25 * heading_presence
        + 0.25 * order_score
        + 0.25 * uniqueness_score
        + 0.25 * non_empty_score
    )
    return float(max(0.0, min(1.0, reward)))


def sentence_count(text: Any) -> int:
    pieces = re.split(r"(?:[.!?]+(?:\s+|$)|\n+)", str(text or "").strip())
    return sum(len(piece.strip().split()) >= 3 for piece in pieces)


REASONING_CONNECTOR_PATTERN = re.compile(
    r"\b(because|therefore|thus|hence|consequently|accordingly|since|"
    r"as a result|in view of|applying|applied|however|although|whereas)\b",
    flags=re.IGNORECASE,
)

OUTCOME_LANGUAGE_PATTERN = re.compile(
    r"\b(allowed|dismissed|disposed|partly allowed|allowed in part|"
    r"dismissed in part|set aside|remanded|withdrawn|other)\b",
    flags=re.IGNORECASE,
)

LEGAL_TERM_PATTERN = re.compile(
    r"\b(section|article|act|code|rule|regulation|constitution|court|"
    r"precedent|principle|held|statute|case|v\.|versus)\b",
    flags=re.IGNORECASE,
)


def has_reasoning_connector(text: Any) -> bool:
    return bool(REASONING_CONNECTOR_PATTERN.search(str(text or "")))


def conclusion_has_outcome_language(text: Any) -> bool:
    return bool(OUTCOME_LANGUAGE_PATTERN.search(str(text or "")))


def compute_coverage_reward(parsed: Dict[str, Any]) -> float:
    sections = parsed["sections"]
    issues = sections.get("issues", "")
    rules = sections.get("rules", "")
    analysis = sections.get("analysis", "")
    conclusion = sections.get("conclusion", "")

    issues_score = float(
        len(issues.split()) >= 6
        and (
            "whether" in issues.lower()
            or "?" in issues
            or sentence_count(issues) >= 1
        )
    )

    rules_score = float(
        len(rules.split()) >= 8 and bool(LEGAL_TERM_PATTERN.search(rules))
    )

    analysis_score = (
        0.40 * min(len(analysis.split()) / 80.0, 1.0)
        + 0.30 * min(sentence_count(analysis) / 3.0, 1.0)
        + 0.30 * float(has_reasoning_connector(analysis))
    )

    generated_outcome = extract_outcome_from_conclusion(conclusion)
    conclusion_score = float(generated_outcome in OUTCOME_LABELS)

    connected_score = float(
        len(analysis.split()) >= 20 and generated_outcome in OUTCOME_LABELS
    )

    reward = (
        0.20 * issues_score
        + 0.20 * rules_score
        + 0.30 * analysis_score
        + 0.20 * conclusion_score
        + 0.10 * connected_score
    )
    return float(max(0.0, min(1.0, reward)))


def compute_format_components(completion: Any) -> Dict[str, Any]:
    parsed = parse_firac_sections(completion)
    r_structure = compute_structure_reward(parsed)
    r_coverage = compute_coverage_reward(parsed)
    soft_format = 0.5 * r_structure + 0.5 * r_coverage

    sections = parsed["sections"]
    all_non_empty = all(bool(sections.get(h, "").strip()) for h in REQUIRED_HEADINGS)
    conclusion_valid = (
        extract_outcome_from_conclusion(sections.get("conclusion", ""))
        in OUTCOME_LABELS
    )
    format_valid = bool(
        parsed["all_present"]
        and parsed["correct_order"]
        and parsed["no_duplicates"]
        and all_non_empty
        and conclusion_valid
    )

    # Hard gate: malformed FIRAC receives no reasoning reward.
    r_format = soft_format if format_valid else 0.0

    return {
        "parsed": parsed,
        "format_valid": format_valid,
        "R_structure": r_structure,
        "R_coverage": r_coverage,
        "R_format_soft": float(max(0.0, min(1.0, soft_format))),
        "R_format": float(max(0.0, min(1.0, r_format))),
    }


# -----------------------------------------------------------------------------
# Item splitting for semantic F1
# -----------------------------------------------------------------------------

LIST_MARKER_PATTERN = re.compile(r"^\s*(?:[-*\u2022]+|\d+[.)]|[a-zA-Z][.)])\s+")

ISSUE_SPLIT_PATTERN = re.compile(
    r"(?<=\?)\s+(?=(?:Whether|What|When|How|Can|Does|Did|Is|Are|Was|Were)\b)",
    flags=re.IGNORECASE,
)


def clean_generated_item(text: Any) -> str:
    text = str(text or "").strip()
    text = LIST_MARKER_PATTERN.sub("", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


def group_wrapped_list_lines(section_text: Any) -> List[str]:
    text = str(section_text or "").strip()
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    has_markers = any(LIST_MARKER_PATTERN.match(line) for line in lines)

    if has_markers:
        grouped: List[str] = []
        current: List[str] = []

        for line in lines:
            if LIST_MARKER_PATTERN.match(line):
                if current:
                    grouped.append(" ".join(current))
                current = [clean_generated_item(line)]
            else:
                current.append(line)

        if current:
            grouped.append(" ".join(current))

        return [
            clean_generated_item(item)
            for item in grouped
            if clean_generated_item(item)
        ]

    return [
        clean_generated_item(item)
        for item in re.split(r"\n\s*\n+", text)
        if clean_generated_item(item)
    ]


def split_section_into_items(section_text: Any, section_type: str) -> List[str]:
    grouped_items = group_wrapped_list_lines(section_text)
    extracted: List[str] = []

    for item in grouped_items:
        if section_type == "issues":
            extracted.extend(ISSUE_SPLIT_PATTERN.split(item))

        elif section_type == "rules":
            parts = [
                clean_generated_item(part)
                for part in re.split(r"\s*;\s*", item)
            ]
            if len(parts) > 1 and all(len(p.split()) >= 5 for p in parts):
                extracted.extend(parts)
            else:
                extracted.append(item)

        else:
            extracted.append(item)

    unique_items: List[str] = []
    seen = set()

    for item in extracted:
        item = clean_generated_item(item)
        if len(item.split()) < 3:
            continue
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    return unique_items


def dedupe_items(items: Optional[Iterable[Any]]) -> List[str]:
    cleaned: List[str] = []
    seen = set()

    for item in items or []:
        item = clean_generated_item(item)
        if not item:
            continue
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)

    return cleaned


def maximum_bipartite_matches(similarity_matrix, threshold: float):
    """Greedy highest-similarity-first one-to-one matching.

    Vectorised: the original implementation looped over every (i, j) pair in
    Python, which dominated reward time once completion counts grew.
    """
    matrix = np.asarray(similarity_matrix, dtype=np.float32)
    if matrix.size == 0:
        return []

    rows, cols = np.nonzero(matrix >= threshold)
    if rows.size == 0:
        return []

    scores = matrix[rows, cols]
    order = np.argsort(-scores, kind="stable")

    used_generated = set()
    used_reference = set()
    matches = []

    for k in order:
        i = int(rows[k])
        j = int(cols[k])
        if i in used_generated or j in used_reference:
            continue
        used_generated.add(i)
        used_reference.add(j)
        matches.append((i, j, float(scores[k])))

    return matches


def f1_from_matches(n_matched: int, n_generated: int, n_reference: int) -> float:
    if n_generated == 0 or n_reference == 0:
        return 0.0
    precision = n_matched / n_generated
    recall = n_matched / n_reference
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


# -----------------------------------------------------------------------------
# Outcome reward
# -----------------------------------------------------------------------------

OUTCOME_LABELS = {"allowed", "dismissed", "other"}

ALLOWED_PATTERNS = [
    r"\bappeal is allowed in part\b",
    r"\ballowed in part\b",
    r"\bpartly allowed\b",
    r"\bappeal is allowed\b",
    r"\bappeal allowed\b",
    r"\ballowed\b",
]

DISMISSED_PATTERNS = [
    r"\bappeal is dismissed in part\b",
    r"\bdismissed in part\b",
    r"\bpartly dismissed\b",
    r"\bappeal is dismissed\b",
    r"\bappeal dismissed\b",
    r"\bdismissed\b",
]

OTHER_PATTERNS = [
    r"\bdisposed of\b",
    r"\bdisposed\b",
    r"\bremanded\b",
    r"\bremitted\b",
    r"\bwithdrawn\b",
    r"\binfructuous\b",
    r"\bother\b",
]


def extract_outcome_from_conclusion(conclusion_text: Any) -> str:
    """Return an outcome only when the entire Conclusion is one valid word."""
    text = normalise_legal_text(conclusion_text)
    if not text:
        return ""

    match = re.fullmatch(r"(allowed|dismissed|other)", text)
    return match.group(1) if match else ""


def compute_outcome_reward(
    parsed: Dict[str, Any],
    reference_outcome: Any,
    return_details: bool = False,
):
    conclusion_text = parsed.get("sections", {}).get("conclusion", "")
    generated_outcome = extract_outcome_from_conclusion(conclusion_text)
    reference = normalise_legal_text(reference_outcome)

    reward = float(
        reference in OUTCOME_LABELS and generated_outcome == reference
    )

    if not return_details:
        return reward

    return {
        "generated_outcome": generated_outcome,
        "reference_outcome": reference,
        "conclusion_text": conclusion_text,
        "reward": reward,
    }


# -----------------------------------------------------------------------------
# Authority reward
# -----------------------------------------------------------------------------

STATUTORY_PATTERN = re.compile(
    r"\b(?:section|sections|s\.|article|articles|art\.|order|rule|regulation|"
    r"clause|schedule)\s+"
    r"([0-9]+[a-zA-Z]*(?:\s*\(\s*[0-9a-zA-Z]+\s*\))*)"
    r"(?:\s+(?:of|under)\s+(?:the\s+)?([A-Z][A-Za-z0-9,'\-\s]{2,60}?"
    r"(?:Act|Code|Constitution|Rules|Regulations|Ordinance)"
    r"(?:,?\s*[0-9]{4})?))?",
    flags=re.IGNORECASE,
)

CASE_PATTERN = re.compile(
    r"([A-Z][A-Za-z.&'\-]*(?:\s+[A-Z][A-Za-z.&'\-]*){0,5})"
    r"\s+(?:v\.?|vs\.?|versus)\s+"
    r"([A-Z][A-Za-z.&'\-]*(?:\s+[A-Z][A-Za-z.&'\-]*){0,5})"
)

CITATION_PATTERN = re.compile(
    r"\(?\b(1[89]\d{2}|20\d{2})\b\)?\s*"
    r"\(?\d*\)?\s*"
    r"(SCC|SCR|AIR|SCC\s+OnLine|All\s+ER|QB|KB)\b[\s.]*\d*",
    flags=re.IGNORECASE,
)

STATUTORY_TYPES = {"statute", "constitutional provision", "rule", "regulation"}


def extract_statutory_authorities_from_item(item: str) -> List[str]:
    found = []
    for match in STATUTORY_PATTERN.finditer(item or ""):
        provision = clean_text(match.group(0))
        if provision:
            found.append(provision)
    return found


def extract_case_authorities_from_item(item: str) -> List[str]:
    found = []
    for match in CASE_PATTERN.finditer(item or ""):
        found.append(clean_text(match.group(0)))
    return found


def extract_citations(item: str) -> List[str]:
    return [clean_text(m.group(0)) for m in CITATION_PATTERN.finditer(item or "")]


def extract_generated_authorities(rules_text: Any) -> List[str]:
    rule_items = split_section_into_items(rules_text, "rules")

    extracted: List[str] = []
    for rule_item in rule_items:
        extracted.extend(extract_statutory_authorities_from_item(rule_item))
        extracted.extend(extract_case_authorities_from_item(rule_item))
        extracted.extend(extract_citations(rule_item))

    unique: List[str] = []
    seen = set()
    for authority in extracted:
        authority = normalise_legal_text(authority)
        if authority and authority not in seen:
            seen.add(authority)
            unique.append(authority)

    return unique


def prepare_reference_authorities(reference_authorities) -> List[str]:
    prepared: List[str] = []

    for authority in reference_authorities or []:
        parts = [p.strip() for p in str(authority).split("|") if p.strip()]
        if not parts:
            continue

        authority_type = normalise_legal_text(parts[0])

        if authority_type in STATUTORY_TYPES:
            reference = parts[1] if len(parts) > 1 else ""
            provision = parts[2] if len(parts) > 2 else ""
            natural_form = (
                f"{provision} of {reference}"
                if provision and reference
                else provision or reference
            )
            if natural_form:
                prepared.append(normalise_legal_text(natural_form))

        elif authority_type == "precedent":
            if len(parts) > 1:
                prepared.append(normalise_legal_text(parts[1]))
            if len(parts) > 2:
                prepared.append(normalise_legal_text(parts[2]))

        elif len(parts) > 1:
            prepared.append(normalise_legal_text(parts[1]))

    unique: List[str] = []
    seen = set()
    for authority in prepared:
        if authority and authority not in seen:
            seen.add(authority)
            unique.append(authority)

    return unique


def authority_pair_matches(generated_authority: str, reference_authority: str) -> bool:
    generated = normalise_legal_text(generated_authority)
    reference = normalise_legal_text(reference_authority)

    if not generated or not reference:
        return False

    if generated == reference:
        return True

    return (
        min(len(generated.split()), len(reference.split())) >= 3
        and (generated in reference or reference in generated)
    )


def compute_authority_reward(
    parsed: Dict[str, Any],
    reference_authorities,
    return_details: bool = False,
):
    rules_text = parsed.get("sections", {}).get("rules", "")
    generated = extract_generated_authorities(rules_text)
    reference = prepare_reference_authorities(reference_authorities)

    candidates = []
    for gi, gitem in enumerate(generated):
        for ri, ritem in enumerate(reference):
            if authority_pair_matches(gitem, ritem):
                candidates.append((gi, ri, float(gitem == ritem)))

    candidates.sort(key=lambda item: item[2], reverse=True)

    used_generated = set()
    used_reference = set()
    accepted = []

    for gi, ri, _ in candidates:
        if gi in used_generated or ri in used_reference:
            continue
        used_generated.add(gi)
        used_reference.add(ri)
        accepted.append((gi, ri))

    f1 = f1_from_matches(len(accepted), len(generated), len(reference))

    if not return_details:
        return f1

    return {
        "generated_authorities": generated,
        "reference_authorities": reference,
        "matches": accepted,
        "f1": f1,
    }


def combine_rewards(
    r_format: float,
    r_outcome: float,
    r_issues: float,
    r_rules: float,
    r_authority: float,
) -> Dict[str, float]:
    r_reasoning = (
        REWARD_WEIGHTS["outcome"] * r_outcome
        + REWARD_WEIGHTS["issues"] * r_issues
        + REWARD_WEIGHTS["rules"] * r_rules
        + REWARD_WEIGHTS["authority"] * r_authority
    )
    r_reasoning = float(max(0.0, min(1.0, r_reasoning)))
    r_total = float(max(0.0, min(1.0, r_format * r_reasoning)))

    return {
        "R_outcome": float(r_outcome),
        "R_issues": float(r_issues),
        "R_rules": float(r_rules),
        "R_authority": float(r_authority),
        "R_reasoning": r_reasoning,
        "R_total": r_total,
    }


# =============================================================================
# PART B -- SEMANTIC SCORING (torch imported lazily)
# =============================================================================

class SemanticScorer:
    """BGE-based semantic F1 with a persistent reference-embedding cache.

    Reference texts are fixed for the whole run, so they are embedded once
    (ideally via `precompute_references`) and then reused. Only generated
    text needs to be embedded during training.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 256,
        threshold: float = SEMANTIC_THRESHOLD,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.threshold = threshold
        self.model = None
        self._reference_cache: Dict[str, Any] = {}

    def load(self):
        if self.model is not None:
            return self.model

        import torch
        from sentence_transformers import SentenceTransformer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.eval()
        # bf16 halves embedding time on Ampere with no measurable F1 change.
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported():
            self.model = self.model.to(torch.bfloat16)

        return self.model

    def _encode(self, texts: Sequence[str]):
        import torch

        self.load()
        with torch.inference_mode():
            return self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def encode_map(self, texts: Sequence[str]) -> Dict[str, Any]:
        unique = list(dict.fromkeys(t for t in texts if t))
        if not unique:
            return {}
        embeddings = self._encode(unique)
        return {text: embeddings[i] for i, text in enumerate(unique)}

    def precompute_references(self, texts: Sequence[str], verbose: bool = False):
        """Embed every reference string once, up front, and cache on CPU."""
        unique = [
            t for t in dict.fromkeys(t for t in texts if t)
            if t not in self._reference_cache
        ]
        if not unique:
            return

        if verbose:
            print(f"Pre-encoding {len(unique):,} unique reference strings...", flush=True)

        chunk = max(self.batch_size * 8, 1024)
        for start in range(0, len(unique), chunk):
            block = unique[start:start + chunk]
            embeddings = self._encode(block)
            for i, text in enumerate(block):
                self._reference_cache[text] = embeddings[i].detach().to("cpu")

        if verbose:
            print(f"Reference cache size: {len(self._reference_cache):,}", flush=True)

    def reference_map(self, texts: Sequence[str]) -> Dict[str, Any]:
        unique = list(dict.fromkeys(t for t in texts if t))
        missing = [t for t in unique if t not in self._reference_cache]

        if missing:
            embeddings = self._encode(missing)
            for i, text in enumerate(missing):
                self._reference_cache[text] = embeddings[i].detach().to("cpu")

        return {t: self._reference_cache[t] for t in unique}

    @property
    def reference_cache_size(self) -> int:
        return len(self._reference_cache)

    def semantic_f1(
        self,
        generated_items: Sequence[str],
        reference_items: Sequence[str],
        generated_map: Dict[str, Any],
        reference_map: Dict[str, Any],
    ) -> float:
        import torch

        generated_items = dedupe_items(generated_items)
        reference_items = dedupe_items(reference_items)

        if not generated_items or not reference_items:
            return 0.0

        try:
            generated_matrix = torch.stack(
                [generated_map[item] for item in generated_items]
            )
            reference_matrix = torch.stack(
                [reference_map[item] for item in reference_items]
            ).to(generated_matrix.device)
        except KeyError:
            # Defensive: an item missing from a map must never crash training.
            return 0.0

        similarity = (
            generated_matrix.float() @ reference_matrix.float().T
        ).cpu().numpy()

        matches = maximum_bipartite_matches(similarity, self.threshold)
        return f1_from_matches(
            len(matches), len(generated_items), len(reference_items)
        )


def get_batch_value(values, index, default=None):
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        return values[index] if index < len(values) else default
    return values


def build_firac_reward(scorer: "SemanticScorer", state: Optional[dict] = None):
    """Return a TRL-compatible reward function closed over `scorer`.

    `state` is an optional dict that receives the most recent per-completion
    breakdown and timings, for the TensorBoard callback to read.
    """
    import time

    if state is None:
        state = {}

    state.setdefault("details", [])
    state.setdefault("timings", {})
    state.setdefault("call_count", 0)

    def firac_grpo_reward(
        completions,
        reference_issues=None,
        reference_rules=None,
        reference_authorities=None,
        outcome_label=None,
        **kwargs,
    ):
        start = time.perf_counter()

        texts = [normalise_completion(c) for c in completions]
        format_results = [compute_format_components(t) for t in texts]
        parsed_items = [r["parsed"] for r in format_results]

        gen_issues, gen_rules = [], []
        ref_issues, ref_rules = [], []
        authorities, outcomes = [], []

        for index, parsed in enumerate(parsed_items):
            sections = parsed.get("sections", {})

            gen_issues.append(
                dedupe_items(
                    split_section_into_items(sections.get("issues", ""), "issues")
                )
            )
            gen_rules.append(
                dedupe_items(
                    split_section_into_items(sections.get("rules", ""), "rules")
                )
            )
            ref_issues.append(
                dedupe_items(get_batch_value(reference_issues, index, []) or [])
            )
            ref_rules.append(
                dedupe_items(get_batch_value(reference_rules, index, []) or [])
            )
            authorities.append(
                get_batch_value(reference_authorities, index, []) or []
            )
            outcomes.append(get_batch_value(outcome_label, index, "") or "")

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

        rewards = []
        details_list = []

        for index, (text, format_result, parsed) in enumerate(
            zip(texts, format_results, parsed_items)
        ):
            r_outcome = float(compute_outcome_reward(parsed, outcomes[index]))
            r_authority = float(
                compute_authority_reward(parsed, authorities[index])
            )

            combined = combine_rewards(
                r_format=float(format_result["R_format"]),
                r_outcome=r_outcome,
                r_issues=float(issue_scores[index]),
                r_rules=float(rule_scores[index]),
                r_authority=r_authority,
            )

            rewards.append(combined["R_total"])

            outcome_details = compute_outcome_reward(
                parsed, outcomes[index], return_details=True
            )
            record = {
                "R_structure": float(format_result["R_structure"]),
                "R_coverage": float(format_result["R_coverage"]),
                "R_format_soft": float(format_result["R_format_soft"]),
                "R_format": float(format_result["R_format"]),
                "format_valid": bool(format_result["format_valid"]),
                "generated_outcome": outcome_details["generated_outcome"],
                "reference_outcome": outcome_details["reference_outcome"],
            }
            record.update(combined)
            record["completion_index"] = index
            record["completion_chars"] = len(text)
            details_list.append(record)

        total_seconds = time.perf_counter() - start

        state["details"] = details_list
        state["timings"] = {
            "parse_seconds": parse_seconds,
            "semantic_seconds": semantic_seconds,
            "total_seconds": total_seconds,
            "completion_count": len(texts),
            "reference_cache_size": scorer.reference_cache_size,
        }
        state["call_count"] += 1

        return rewards

    return firac_grpo_reward


# =============================================================================
# PART C -- DATASET PREPARATION
# =============================================================================

DATASET_NAME = "mborcin/firac-appeals"
DATASET_CONFIG = "combined"

SPLIT_SEED = 42
SFT_SEED = 142

TRAIN_COUNTS = {"allowed": 4500, "dismissed": 4500, "other": 1000}
EVAL_COUNTS = {"allowed": 2250, "dismissed": 2250, "other": 500}
SFT_COUNTS = {"allowed": 900, "dismissed": 900, "other": 900}
SFT_VALIDATION_PER_CLASS = 100
GRPO_OTHER_REPEAT = 2


def format_reference_rules(rules) -> List[str]:
    formatted = []

    for rule in rules or []:
        if not isinstance(rule, dict):
            continue

        parts = []
        reference = clean_text(rule.get("reference"))

        for field, label in [
            ("type", "Type"),
            ("reference", "Reference"),
            ("provision", "Provision"),
            ("case_name", "Case"),
            ("citation", "Citation"),
            ("relevance", "Rule"),
        ]:
            value = clean_text(rule.get(field))
            if not value:
                continue
            if field == "case_name" and value.lower() == reference.lower():
                continue
            parts.append(f"{label}: {value}")

        if parts:
            formatted.append(". ".join(parts))

    return formatted


def extract_reference_authorities(rules) -> List[str]:
    authorities = []

    for rule in rules or []:
        if not isinstance(rule, dict):
            continue

        parts = []
        reference = clean_text(rule.get("reference"))

        for field in ["type", "reference", "provision", "case_name", "citation"]:
            value = clean_text(rule.get(field))
            if not value:
                continue
            if field == "case_name" and value.lower() == reference.lower():
                continue
            parts.append(value)

        if parts:
            authorities.append(" | ".join(parts))

    return authorities


def build_compact_case_facts(facts, max_words: int = 350, max_items: int = 12) -> str:
    selected = []
    total_words = 0

    for fact in facts or []:
        fact = clean_text(fact)
        if not fact:
            continue

        remaining = max_words - total_words
        if remaining <= 0:
            break

        words = fact.split()
        if len(words) > remaining:
            fact = " ".join(words[:remaining])

        selected.append(fact)
        total_words += len(fact.split())

        if len(selected) >= max_items:
            break

    return "\n".join(f"{i}. {fact}" for i, fact in enumerate(selected, start=1))


def build_user_prompt(example) -> str:
    firac = example.get("firac") or {}
    compact_facts = build_compact_case_facts(firac.get("facts") or [])
    return "Case facts:\n\n" + compact_facts + USER_PROMPT_SUFFIX


def build_reference_fields(example) -> Dict[str, Any]:
    firac = example.get("firac") or {}
    conclusion = firac.get("conclusion") or {}

    return {
        "reference_issues": [
            clean_text(i) for i in (firac.get("issues") or []) if clean_text(i)
        ],
        "reference_rules": format_reference_rules(firac.get("rules") or []),
        "reference_authorities": extract_reference_authorities(
            firac.get("rules") or []
        ),
        "reference_conclusion_text": clean_text(conclusion.get("text")),
        "outcome_label": clean_text(
            example.get("simplified_outcome_label")
            or conclusion.get("simplified_label")
        ).lower(),
    }


def make_grpo_mapper(tokenizer):
    """Return a `.map` function producing a PRE-RENDERED text prompt.

    This is the fix for the Qwen3 thinking-mode mismatch. TRL applies the
    chat template itself for conversational prompts, using the template's
    defaults -- which for Qwen3 means `enable_thinking=True`. SFT was done
    with `enable_thinking=False`, so the policy was being rolled out under a
    template it was never trained on, and the reasoning block consumed the
    completion budget before `Conclusion:` was ever written.

    By rendering to a plain string here, TRL treats the prompt as
    non-conversational and passes it through verbatim.
    """
    def _mapper(example):
        messages = [
            {"role": "system", "content": SHORT_SFT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(example)},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        out = {"prompt": prompt_text}
        out.update(build_reference_fields(example))
        return out

    return _mapper


def collect_reference_strings(dataset) -> List[str]:
    """Every string the reward will ever need to embed on the reference side."""
    texts: List[str] = []
    for issues, rules in zip(
        dataset["reference_issues"], dataset["reference_rules"]
    ):
        texts.extend(dedupe_items(issues))
        texts.extend(dedupe_items(rules))
    return list(dict.fromkeys(texts))


# -----------------------------------------------------------------------------
# SFT target construction
# -----------------------------------------------------------------------------

def truncate_words(text: Any, max_words: int) -> str:
    return " ".join(clean_text(text).split()[:max_words])


def compact_rule_text(rule) -> str:
    """One structured rule -> a concise line keeping provision, authority,
    citation, and principle."""
    if not isinstance(rule, dict):
        return ""

    reference = clean_text(rule.get("reference"))
    provision = clean_text(rule.get("provision"))
    case_name = clean_text(rule.get("case_name"))
    citation = clean_text(rule.get("citation"))
    relevance = truncate_words(rule.get("relevance"), 28)

    authority_parts = []
    if provision:
        authority_parts.append(provision)
    if reference:
        authority_parts.append(reference)
    if case_name and case_name.lower() != reference.lower():
        authority_parts.append(case_name)
    if citation:
        authority_parts.append(citation)

    authority = " \u2014 ".join(authority_parts)

    if authority and relevance:
        return f"{authority}: {relevance}"

    return authority or relevance


def build_short_sft_answer(example) -> Dict[str, str]:
    """Concise reference FIRAC target with the outcome first in Conclusion.

    The word limits here mirror SHORT_SFT_SYSTEM_PROMPT exactly. If you change
    one, change the other, or the model is being taught to violate its own
    instructions.
    """
    firac = example.get("firac") or {}
    conclusion = firac.get("conclusion") or {}

    facts = [clean_text(f) for f in (firac.get("facts") or []) if clean_text(f)]
    facts_text = truncate_words(" ".join(facts), 70)

    issues = [
        truncate_words(issue, 30)
        for issue in (firac.get("issues") or [])
        if clean_text(issue)
    ][:2]
    issues_text = "\n".join(
        f"{i}. {issue}" for i, issue in enumerate(issues, start=1)
    )

    compact_rules = [
        rule for rule in
        (compact_rule_text(r) for r in (firac.get("rules") or []))
        if rule
    ][:4]
    rules_text = "\n".join(
        f"{i}. {rule}" for i, rule in enumerate(compact_rules, start=1)
    )

    analysis_parts: List[str] = []
    for item in firac.get("analysis") or []:
        if not isinstance(item, dict):
            continue
        analysis_parts.extend(
            clean_text(s) for s in (item.get("steps") or []) if clean_text(s)
        )
        issue_conclusion = clean_text(item.get("conclusion"))
        if issue_conclusion:
            analysis_parts.append(issue_conclusion)

    analysis_text = truncate_words(" ".join(analysis_parts), 100)

    outcome = clean_text(
        example.get("simplified_outcome_label")
        or conclusion.get("simplified_label")
    ).upper()

    if outcome not in {"ALLOWED", "DISMISSED", "OTHER"}:
        outcome = "OTHER"

    # The target Conclusion is deliberately one word. This removes the
    # ambiguity that corrupted the first GRPO experiment.
    conclusion_output = outcome

    answer = (
        f"Facts:\n{facts_text}\n\n"
        f"Issues:\n{issues_text}\n\n"
        f"Rules:\n{rules_text}\n\n"
        f"Analysis:\n{analysis_text}\n\n"
        f"Conclusion:\n{conclusion_output}"
    ).strip()

    return {"short_sft_answer": answer}


def make_sft_mapper(tokenizer):
    """Create a standard prompt-completion example.

    The prompt is pre-rendered with Qwen3 thinking disabled. The completion
    contains only the assistant FIRAC target plus EOS. TRL must therefore mask
    every prompt token and compute SFT loss only on the completion.
    """
    def _mapper(example):
        prompt_messages = [
            {"role": "system", "content": SHORT_SFT_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(example)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        completion_text = example["short_sft_answer"]
        if tokenizer.eos_token and not completion_text.endswith(tokenizer.eos_token):
            completion_text += tokenizer.eos_token

        return {
            "prompt": prompt_text,
            "completion": completion_text,
        }

    return _mapper


def build_sft_pool(raw_grpo_train, heldout_eval=None):
    """Build a class-balanced, stratified SFT train/validation split.

    For each outcome, 900 unique cases are selected. The first 100 are reserved
    for validation and the remaining 800 are used for SFT training.
    """
    from datasets import concatenate_datasets

    with_targets = raw_grpo_train.map(
        build_short_sft_answer, desc="Building one-word SFT targets"
    )

    train_parts = []
    validation_parts = []

    for offset, (label, count) in enumerate(SFT_COUNTS.items()):
        pool = with_targets.filter(
            lambda ex, target=label: normalise_label(
                ex["simplified_outcome_label"]
            ) == target
        ).shuffle(seed=SFT_SEED + offset)

        if len(pool) < count:
            raise ValueError(
                f"Not enough '{label}' cases for SFT: need {count}, "
                f"found {len(pool)}"
            )
        if count <= SFT_VALIDATION_PER_CLASS:
            raise ValueError("SFT count must exceed validation count.")

        selected = pool.select(range(count))
        validation_parts.append(
            selected.select(range(SFT_VALIDATION_PER_CLASS))
        )
        train_parts.append(
            selected.select(range(SFT_VALIDATION_PER_CLASS, count))
        )

    sft_train = concatenate_datasets(train_parts).shuffle(seed=SFT_SEED)
    sft_validation = concatenate_datasets(validation_parts).shuffle(
        seed=SFT_SEED + 1
    )

    if heldout_eval is not None:
        eval_ids = set(map(str, heldout_eval["case_id"]))
        for name, split in (
            ("train", sft_train),
            ("validation", sft_validation),
        ):
            ids = set(map(str, split["case_id"]))
            if not ids.isdisjoint(eval_ids):
                raise ValueError(f"SFT {name} split overlaps the held-out set.")

    train_ids = set(map(str, sft_train["case_id"]))
    validation_ids = set(map(str, sft_validation["case_id"]))
    if not train_ids.isdisjoint(validation_ids):
        raise ValueError("SFT train and validation splits overlap.")

    return sft_train, sft_validation


def build_grpo_training_schedule(raw_grpo_train):
    """Keep all 10K unique cases and repeat OTHER cases once.

    This changes exposure frequency without removing any Allowed or Dismissed
    case. Expected schedule: 4,500 Allowed, 4,500 Dismissed, 2,000 Other.
    """
    from datasets import concatenate_datasets

    other = raw_grpo_train.filter(
        lambda ex: normalise_label(ex["simplified_outcome_label"]) == "other"
    )
    repeats = [raw_grpo_train] + [other] * max(0, GRPO_OTHER_REPEAT - 1)
    return concatenate_datasets(repeats).shuffle(seed=SPLIT_SEED + 99)


def build_balanced_splits(dataset):
    """Deterministic, disjoint 10K training / 5K held-out split."""
    from datasets import concatenate_datasets

    raw_train = dataset["train"]
    labels = ["allowed", "dismissed", "other"]

    train_parts, eval_parts = [], []

    for offset, label in enumerate(labels):
        pool = raw_train.filter(
            lambda ex, target=label: normalise_label(
                ex["simplified_outcome_label"]
            ) == target
        ).shuffle(seed=SPLIT_SEED + offset)

        required = TRAIN_COUNTS[label] + EVAL_COUNTS[label]
        if len(pool) < required:
            raise ValueError(
                f"Not enough '{label}' cases: need {required}, found {len(pool)}"
            )

        train_parts.append(pool.select(range(TRAIN_COUNTS[label])))
        eval_parts.append(
            pool.select(
                range(TRAIN_COUNTS[label], TRAIN_COUNTS[label] + EVAL_COUNTS[label])
            )
        )

    raw_grpo_train = concatenate_datasets(train_parts).shuffle(seed=SPLIT_SEED)
    raw_heldout_eval = concatenate_datasets(eval_parts).shuffle(seed=SPLIT_SEED)

    train_ids = set(map(str, raw_grpo_train["case_id"]))
    eval_ids = set(map(str, raw_heldout_eval["case_id"]))

    if not train_ids.isdisjoint(eval_ids):
        raise ValueError("Train and held-out splits overlap.")

    return raw_grpo_train, raw_heldout_eval
