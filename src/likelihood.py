"""
We implement the uniform-random strategy ("logic-only baseline") as a test distribution:
every legal action is equally likely. Under this regime:

  - Vote contributions are 1/2 per vote regardless of role -> constant across
    assignments, cancels after renormalization.
  - Proposal contributions are 1/C(5, m) regardless of role -> constant,
    cancels.
  - Mission outcomes are the only term that distinguishes assignments.

For a mission where team T plays and `f` fails are observed:
  Under uniform random, each evil player on T independently chose succeed/fail
  with probability 1/2 (good players have only one legal action: succeed).
  Summing over the C(k, f) ways `f` of the `k` evil players on T might have
  failed (the deductive ambiguity sum from Sec. 3.1 of the paper):

    P(observe f fails | k evil on team) = C(k, f) / 2^k    if k >= f else 0.

  When k < f the likelihood is zero, which redundantly mirrors the consistency
  mask. We rely on math.comb(k, f) = 0 for k < f to handle this.

This module returns ONLY the mission contribution. Vote/proposal terms are
omitted because they cancel under uniform random.
"""
from __future__ import annotations

from math import comb

import numpy as np

from src.assignments import ASSIGNMENTS, NUM_ASSIGNMENTS, evil_indices


def uniform_likelihood(observation: dict) -> np.ndarray:
    """Per-assignment likelihood ∏_i π_i^σ under the uniform-random strategy.

    Args:
      observation: a dict produced by GameState.observation().
    Returns:
      np.ndarray of shape (60,), dtype float64. Values in [0, 1]. Inconsistent
      assignments get 0.0 automatically.
    """
    lik = np.ones(NUM_ASSIGNMENTS, dtype=np.float64)

    for round_rec in observation["rounds"]:
        if round_rec.get("mission_fails") is None:
            continue
        approved = [p for p in round_rec["proposals"] if p.get("approved")]
        assert len(approved) == 1
        team = set(approved[0]["team"])
        f = round_rec["mission_fails"]

        for a_idx, rho in enumerate(ASSIGNMENTS):
            k = len(evil_indices(rho) & team)
            # comb(k, f) = 0 when k < f, which zeroes inconsistent ρ.
            lik[a_idx] *= comb(k, f) / (2 ** k)

    return lik
