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

def print_consistency_mask(mask, observation=None):
    print(f"Consistent assignments: {mask.sum()} / {len(mask)}\n")
    print(f"{'idx':<5} {'Seat0':<12} {'Seat1':<12} {'Seat2':<12} {'Seat3':<12} {'Seat4':<12} {'Evil seats'}")
    print("-" * 85)
    for a_idx, (rho, valid) in enumerate(zip(ASSIGNMENTS, mask)):
        if not valid:
            continue
        roles = [r.name for r in rho]
        evils = sorted(evil_indices(rho))
        print(
            f"{a_idx:<5} "
            f"{roles[0]:<12} {roles[1]:<12} {roles[2]:<12} "
            f"{roles[3]:<12} {roles[4]:<12} {evils}"
        )
        
# from game import Role, new_game
# from assignments import ASSIGNMENTS, evil_indices
# from scenarios import make_obs_with_mission
# rho = (Role.LS, Role.LS, Role.MERLIN, Role.DS, Role.ASSASSIN)
# obs = make_obs_with_mission(team=(1,3,4), fails=2, assignment=rho, round_idx=1)
# mask = consistency_mask(obs)

# print_consistency_mask(mask)
