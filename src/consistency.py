"""
Computes the (1 - 1{h ⊢ ¬ρ}) term in Eq. 1 of Serrino et al. (2019):
returns a (60,) boolean mask where True means assignment ρ is logically
consistent with the public observation h, False means impossible.

We deduce only from mission outcomes:
  A mission with team T showing `f` fails rules out any ρ where fewer than
  `f` players in T are evil. (Arthur-side players must succeed by game
  rule, so all `f` fails came from evil players on the team.)

Vote outcomes are NOT used for deduction: any role can mechanically vote
either way without revealing information. Strategic vote interpretations
belong in the strategy / likelihood module, not here.
"""
from __future__ import annotations

import numpy as np

from src.assignments import ASSIGNMENTS, NUM_ASSIGNMENTS, evil_indices


def consistency_mask(observation: dict) -> np.ndarray:
    """Return a (60,) bool array: True for assignments consistent with `h`.

    Args:
      observation: a dict produced by GameState.observation().
    Returns:
      np.ndarray of shape (60,), dtype bool.
    """
    mask = np.ones(NUM_ASSIGNMENTS, dtype=bool)

    for round_rec in observation["rounds"]:
        if round_rec.get("mission_fails") is None:
            continue  # mission not yet played for this round
        approved = [p for p in round_rec["proposals"] if p.get("approved")]
        assert len(approved) == 1, (
            "expected exactly one approved proposal per played round"
        )
        team = set(approved[0]["team"])
        f = round_rec["mission_fails"]

        for a_idx, rho in enumerate(ASSIGNMENTS):
            if not mask[a_idx]:
                continue
            if len(evil_indices(rho) & team) < f:
                mask[a_idx] = False

    return mask
