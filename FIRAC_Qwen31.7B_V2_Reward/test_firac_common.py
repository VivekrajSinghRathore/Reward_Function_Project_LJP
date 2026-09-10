"""Fail-closed tests for the corrected FIRAC reward and data contract."""

import firac_common as fc


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        raise AssertionError(name)


GOOD = """
Facts:
The appellant challenged the conviction after the High Court reappraised the
evidence.

Issues:
1. Whether the conviction was legally sustainable?

Rules:
1. Section 302 of the Indian Penal Code governs punishment for murder.

Analysis:
Applying the statutory rule, the Court found that the material evidence was
not reliable because material contradictions remained. Therefore the
conviction could not safely stand.

Conclusion:
ALLOWED
""".strip()

EXTRA_CONCLUSION = GOOD.replace(
    "Conclusion:\nALLOWED",
    "Conclusion:\nALLOWED. The appeal succeeds.",
)

MISSING_RULES = GOOD.replace(
    "Rules:\n1. Section 302 of the Indian Penal Code governs punishment for murder.\n\n",
    "",
)

# ---------------------------------------------------------------------------
# Strict one-word outcome parser
# ---------------------------------------------------------------------------
parser_cases = [
    ("ALLOWED", "allowed"),
    ("DISMISSED", "dismissed"),
    ("OTHER", "other"),
    (" allowed ", "allowed"),
    ("DISMISSED. The High Court had allowed the claim.", ""),
    ("OTHER. Although the earlier appeal was allowed.", ""),
    ("The appeal is dismissed", ""),
    ("", ""),
]
for text, expected in parser_cases:
    actual = fc.extract_outcome_from_conclusion(text)
    check(f"strict outcome {text!r}", actual == expected)

# ---------------------------------------------------------------------------
# FIRAC parser and hard format gate
# ---------------------------------------------------------------------------
parsed = fc.parse_firac_sections(GOOD)
check("all headings present", parsed["all_present"])
check("headings ordered", parsed["correct_order"])
check("headings unique", parsed["no_duplicates"])

good_format = fc.compute_format_components(GOOD)
check("valid FIRAC passes hard gate", good_format["format_valid"])
check("valid FIRAC has positive format reward", good_format["R_format"] > 0)

extra_format = fc.compute_format_components(EXTRA_CONCLUSION)
check("extra conclusion text fails hard gate", not extra_format["format_valid"])
check("extra conclusion receives zero format reward", extra_format["R_format"] == 0)

missing_format = fc.compute_format_components(MISSING_RULES)
check("missing section fails hard gate", not missing_format["format_valid"])
check("missing section receives zero format reward", missing_format["R_format"] == 0)

# ---------------------------------------------------------------------------
# Outcome reward
# ---------------------------------------------------------------------------
good_parsed = fc.parse_firac_sections(GOOD)
check(
    "matching outcome reward",
    fc.compute_outcome_reward(good_parsed, "allowed") == 1.0,
)
check(
    "mismatching outcome reward",
    fc.compute_outcome_reward(good_parsed, "dismissed") == 0.0,
)

# ---------------------------------------------------------------------------
# Reward weights and gate
# ---------------------------------------------------------------------------
check("reward weights sum to one", abs(sum(fc.REWARD_WEIGHTS.values()) - 1.0) < 1e-9)
check("outcome weight is 0.40", fc.REWARD_WEIGHTS["outcome"] == 0.40)

combined = fc.combine_rewards(1.0, 1.0, 1.0, 1.0, 1.0)
check("perfect reward equals one", combined["R_total"] == 1.0)

gated = fc.combine_rewards(0.0, 1.0, 1.0, 1.0, 1.0)
check("hard format gate zeroes total", gated["R_total"] == 0.0)

# ---------------------------------------------------------------------------
# One-word SFT target and evidence budget
# ---------------------------------------------------------------------------
dummy = {
    "simplified_outcome_label": "dismissed",
    "firac": {
        "facts": ["Fact " + str(i) for i in range(20)],
        "issues": ["Whether the appeal should succeed?"],
        "rules": [{"reference": "Section 1", "relevance": "Applicable rule"}],
        "analysis": [{"steps": ["The rule applies because the evidence fails."]}],
        "conclusion": {
            "simplified_label": "dismissed",
            "text": "The appeal is dismissed although an earlier claim was allowed.",
        },
    },
}
target = fc.build_short_sft_answer(dummy)["short_sft_answer"]
target_parsed = fc.parse_firac_sections(target)
check(
    "SFT Conclusion is exactly one word",
    target_parsed["sections"]["conclusion"] == "DISMISSED",
)

facts = fc.build_compact_case_facts(dummy["firac"]["facts"])
check("fact item cap", len(facts.splitlines()) <= 12)
check("fact word cap", len(facts.split()) <= 350 + 12)  # includes numbering

check("SFT classes balanced", len(set(fc.SFT_COUNTS.values())) == 1)
check("SFT Other count is 900", fc.SFT_COUNTS["other"] == 900)

print("\nAll V2 FIRAC preflight tests passed.")
