"""
Stateful class implementing Equation 1 of Serrino et al. (2019):

    b(ρ|h) ∝ b(ρ) · (1 - 1{h ⊢ ¬ρ}) · ∏_i π_i^σ(I_i(h, ρ))

Maintains a single global joint posterior b(ρ|h) over the 60 role assignments.
Per-player views (e.g. for CFR's counterfactual values) are derived from this
joint by conditioning — see condition_on_role().

API:
  tracker = BeliefTracker()                # uniform prior over 60 assignments
  tracker.observe(game.observation())      # mutates internal _belief
  tracker.belief                           # current b(ρ|h), shape (60,)
  tracker.marginal_evil(player)            # P(player is evil | h)
  tracker.marginal_role(player, role)      # P(player has role | h)
  tracker.condition_on_role(player, role)  # b(ρ|h, ρ[player]=role)
  tracker.reset()                          # restore prior

Implementation notes:
  - observe() recomputes the belief from scratch given the full public history.
    Avalon games are short (≤25 events), so this is cheap and bug-free.
  - Strategy is uniform-random for now. When CFR exists, we'll
    swap in a strategy-aware likelihood without changing this class's API.
  - This class never accesses GameState.assignment. Pass in observation()
    output, never the GameState itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.assignments import ASSIGNMENTS, NUM_ASSIGNMENTS
from src.consistency import consistency_mask
from src.likelihood import uniform_likelihood


class BeliefTracker:
    """Stateful belief tracker maintaining b(ρ|h) over the 60 assignments."""

    def __init__(self, prior: Optional[np.ndarray] = None) -> None:
        if prior is None:
            prior = np.full(NUM_ASSIGNMENTS, 1.0 / NUM_ASSIGNMENTS)
        prior = np.asarray(prior, dtype=np.float64)
        assert prior.shape == (NUM_ASSIGNMENTS,), prior.shape
        assert np.all(prior >= 0), "prior must be non-negative"
        s = prior.sum()
        assert s > 0, "prior must have positive total mass"
        self._prior = prior / s
        self._belief = self._prior.copy()

    # ---- core API -------------------------------------------------------

    @property
    def belief(self) -> np.ndarray:
        """Current b(ρ|h), shape (60,). Returns a copy to prevent mutation."""
        return self._belief.copy()

    def reset(self) -> None:
        """Restore the prior."""
        self._belief = self._prior.copy()

    def observe(self, observation: dict) -> None:
        """Recompute b(ρ|h) from scratch given the public observation.

        Args:
          observation: dict from GameState.observation().
        """
        mask = consistency_mask(observation).astype(np.float64)
        lik = uniform_likelihood(observation)
        unnorm = self._prior * mask * lik
        Z = unnorm.sum()
        if Z <= 0:
            raise RuntimeError(
                "All assignments have zero posterior. Likely a bug in "
                "consistency.py or likelihood.py, or an inconsistent "
                "observation was passed in."
            )
        self._belief = unnorm / Z

    # ---- query helpers --------------------------------------------------

    def marginal_role(self, player: int, role) -> float:
        """P(player has the given role | h)."""
        return float(sum(
            self._belief[i] for i, rho in enumerate(ASSIGNMENTS)
            if rho[player] is role
        ))

    def marginal_evil(self, player: int) -> float:
        """P(player is evil | h). Equals marginal_role(player, SPY) +
        marginal_role(player, ASSASSIN)."""
        return float(sum(
            self._belief[i] for i, rho in enumerate(ASSIGNMENTS)
            if rho[player].is_evil()
        ))

    def condition_on_role(self, player: int, role) -> np.ndarray:
        """Belief conditioned on the constraint that `player` has `role`.

        Used to derive a player's subjective belief from the global b(ρ|h):
        a player who privately knows their own role conditions the global
        belief on that fact. Returns a normalized (60,) array.
        """
        cond = np.array([
            self._belief[i] if rho[player] is role else 0.0
            for i, rho in enumerate(ASSIGNMENTS)
        ], dtype=np.float64)
        Z = cond.sum()
        if Z <= 0:
            raise ValueError(
                f"P(player {player} has role {role}) = 0 under current belief"
            )
        return cond / Z
