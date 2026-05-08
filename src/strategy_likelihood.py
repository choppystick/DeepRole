"""
Strategy-aware likelihood — the ∏_i π_i^σ(I_i(h, ρ)) factor in Equation 1 /
CALCTERMINALBELIEF (Algorithm 2 line 19).

Replaces the uniform-random `uniform_likelihood` of Section 2 when CFR is the
one driving belief updates. Inside the CFR recursion we maintain per-player
per-info-set reach probs as we descend; at any leaf this function gathers
them by assignment to produce the per-ρ likelihood factor.

The consistency mask term (1 − 1{h ⊢ ¬ρ}) is NOT included here — it lives in
consistency.py and is multiplied in separately by the caller.
"""
from __future__ import annotations

import numpy as np

from src.assignments import INFO_SET_INDEX, NUM_ASSIGNMENTS
from src.game import NUM_PLAYERS


_PLAYER_INDICES = np.arange(NUM_PLAYERS)[:, None]


def strategy_aware_likelihood(reach_probs: np.ndarray) -> np.ndarray:
    """Per-assignment ∏_i reach_probs[i, I_i(ρ)].

    Args:
      reach_probs: shape (NUM_PLAYERS, 15) — per-player per-info-set reach
        probability from the subgame root to the current public node.
    Returns:
      (60,) array; element a equals ∏_i reach_probs[i, INFO_SET_INDEX[i, a]].
    """
    assert reach_probs.shape == (NUM_PLAYERS, 15), reach_probs.shape
    gathered = reach_probs[_PLAYER_INDICES, INFO_SET_INDEX]   # (5, 60)
    return gathered.prod(axis=0)
