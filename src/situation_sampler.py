"""
Situation sampler — Algorithm 4 of Serrino et al. (2019).

Generates training situations (proposer, belief) for a target game-part
indexed by (s, f). Used to feed CFR data generation in Algorithm 3.

Algorithm 4 (paper, p. 14):
  1: INPUT s: number of succeeds.
  2: INPUT f: number of fails.
  3: OUTPUT p, b: a random game situation.
  4: procedure SAMPLE_SITUATION(s, f)
  5:   I ← SAMPLE_FAILED_MISSIONS(s, f)         ⊳ uniformly sample f failed missions
  6:   E ← EVIL_PLAYERS(I)                       ⊳ evil teams consistent with I
  7:   P(E) ∼ Dir(1_|E|)
  8:   P(M) ∼ Dir(1_n)
  9:   b ← P(E) ⨂ P(M)
 10:   p ∼ unif{1, n}
 11:   return p, b
 12: end procedure

The paper underspecifies SAMPLE_FAILED_MISSIONS — what teams were on the
failed missions. We sample uniformly from the round-by-round team-size
schedule: pick which f of the s+f played missions failed, then for each of
those failed-mission slots sample a random team of TEAM_SIZES[round] from
all C(5, team_size) possibilities.

E (consistent evil pairs) = {(i, j) : ∀ failed team T, {i, j} ∩ T ≠ ∅}.

The belief b is over the 60 ASSIGNMENTS. For each assignment ρ:
  b[ρ] = P(E)[evil_pair(ρ)] · P(M)[merlin_seat(ρ)]
followed by renormalization (since P(M) over all 5 seats includes the
2 evil seats, which never have Merlin — those ρ get zeroed before renorm).
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from src.assignments import (
    ASSIGNMENTS,
    NUM_ASSIGNMENTS,
    evil_indices,
    merlin_index,
)
from src.game import NUM_PLAYERS, TEAM_SIZES


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

# All 10 possible evil pairs (unordered) {i, j} ⊂ {0..4}.
_EVIL_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    combinations(range(NUM_PLAYERS), 2)
)
assert len(_EVIL_PAIRS) == 10


def _evil_pair_of_assignment(rho) -> Tuple[int, int]:
    return tuple(sorted(evil_indices(rho)))


def _consistent_evil_pairs(failed_teams: List[Tuple[int, ...]]) -> List[Tuple[int, int]]:
    """Evil pairs (i, j) with at least one of {i, j} on every failed-mission team.

    Reasoning: a failed mission requires ≥1 evil player on the team. If neither
    i nor j was on a failed team, that team had 0 evils and can't have failed.
    """
    consistent = []
    for pair in _EVIL_PAIRS:
        ok = all(any(p in T for p in pair) for T in failed_teams)
        if ok:
            consistent.append(pair)
    return consistent


def _sample_failed_team_for_round(round_idx: int, rng: np.random.Generator) -> Tuple[int, ...]:
    """Sample a uniform random team of TEAM_SIZES[round_idx]."""
    team_size = TEAM_SIZES[round_idx]
    teams = list(combinations(range(NUM_PLAYERS), team_size))
    idx = int(rng.integers(0, len(teams)))
    return teams[idx]


def _sample_failed_missions(
    s: int, f: int, rng: np.random.Generator
) -> List[Tuple[int, ...]]:
    """Algorithm 4 line 5. Sample f failed mission teams from the s+f past missions.

    The s+f past missions occupy round indices 0..(s+f-1). We pick which f of
    those s+f rounds were failures (uniformly), then sample a random team for
    each failed-round slot.
    """
    n_past = s + f
    if f == 0:
        return []
    failed_rounds = rng.choice(n_past, size=f, replace=False)
    return [_sample_failed_team_for_round(int(r), rng) for r in failed_rounds]


# ---------------------------------------------------------------------------
# SAMPLE_SITUATION (Algorithm 4)
# ---------------------------------------------------------------------------

def sample_situation(
    s: int,
    f: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, np.ndarray]:
    """Sample a (proposer, belief) for the (s, f) game-part.

    Args:
      s: number of successful missions (0..2 valid; networks exist for these).
      f: number of failed missions   (0..2 valid).
      rng: optional Generator for determinism.
    Returns:
      proposer: int ∈ {0..NUM_PLAYERS-1}.
      belief: (60,) float64, normalized to sum to 1.
    """
    assert 0 <= s < 3 and 0 <= f < 3, f"unsupported (s, f): ({s}, {f})"
    if rng is None:
        rng = np.random.default_rng()

    # Step 5: sample failed missions; step 6: derive consistent evil pairs.
    failed_teams = _sample_failed_missions(s, f, rng)
    consistent_pairs = _consistent_evil_pairs(failed_teams)
    assert len(consistent_pairs) > 0, (
        "no consistent evil pair — sampler bug"
    )

    # Step 7: P(E) ∼ Dir(1_|E|).
    # Step 8: P(M) ∼ Dir(1_n) over all 5 seats.
    p_evil_pair = rng.dirichlet(np.ones(len(consistent_pairs)))
    p_merlin = rng.dirichlet(np.ones(NUM_PLAYERS))

    # Step 9: b[ρ] = P(E)[evil_pair(ρ)] · P(M)[merlin_seat(ρ)].
    # ρ where evil_pair(ρ) ∉ consistent_pairs gets 0.
    # ρ where merlin is on an evil seat is impossible by role multiset (Merlin
    # is good), so this never happens — every ρ has Merlin on a non-evil seat.
    pair_idx = {pair: i for i, pair in enumerate(consistent_pairs)}
    belief = np.zeros(NUM_ASSIGNMENTS, dtype=np.float64)
    for a_idx, rho in enumerate(ASSIGNMENTS):
        pair = _evil_pair_of_assignment(rho)
        if pair not in pair_idx:
            continue
        m_seat = merlin_index(rho)
        belief[a_idx] = p_evil_pair[pair_idx[pair]] * p_merlin[m_seat]
    total = belief.sum()
    if total <= 0:
        # Degenerate Dirichlet draw (mass landed entirely on evil seats for M).
        # Re-normalize against ρ-consistent mass: redistribute uniformly over
        # the consistent assignments.
        for a_idx, rho in enumerate(ASSIGNMENTS):
            if _evil_pair_of_assignment(rho) in pair_idx:
                belief[a_idx] = 1.0
        total = belief.sum()
    belief /= total

    # Step 10: proposer ∼ uniform.
    proposer = int(rng.integers(0, NUM_PLAYERS))
    return proposer, belief
