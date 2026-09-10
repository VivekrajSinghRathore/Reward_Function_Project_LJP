"""R1 controlled reward ablation: effective weights 5/35/15/30/15."""
import firac_binary_soft_rca as _base

REWARD_SCHEME = "R1_outcome35"
FORMAT_WEIGHT = 0.05
REASONING_WEIGHT = 0.95
REWARD_WEIGHTS = {
    "outcome": 7.0 / 19.0,
    "issues": 3.0 / 19.0,
    "rules": 6.0 / 19.0,
    "authority": 3.0 / 19.0,
}
EFFECTIVE_REWARD_WEIGHTS = {
    "format": 0.05,
    "outcome": 0.35,
    "issues": 0.15,
    "rules": 0.30,
    "authority": 0.15,
}

assert abs(sum(REWARD_WEIGHTS.values()) - 1.0) < 1e-12
assert abs(FORMAT_WEIGHT + REASONING_WEIGHT - 1.0) < 1e-12
assert abs(sum(EFFECTIVE_REWARD_WEIGHTS.values()) - 1.0) < 1e-12

_base.FORMAT_WEIGHT = FORMAT_WEIGHT
_base.REASONING_WEIGHT = REASONING_WEIGHT
_base.REWARD_WEIGHTS = REWARD_WEIGHTS

from firac_binary_soft_rca import *  # noqa: F401,F403,E402

FORMAT_WEIGHT = 0.05
REASONING_WEIGHT = 0.95
REWARD_WEIGHTS = {
    "outcome": 7.0 / 19.0,
    "issues": 3.0 / 19.0,
    "rules": 6.0 / 19.0,
    "authority": 3.0 / 19.0,
}
REWARD_SCHEME = "R1_outcome35"
EFFECTIVE_REWARD_WEIGHTS = {
    "format": 0.05,
    "outcome": 0.35,
    "issues": 0.15,
    "rules": 0.30,
    "authority": 0.15,
}
