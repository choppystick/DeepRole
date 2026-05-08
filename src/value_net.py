"""
Section 3 — value network and win-probability readout.

ValueNetwork (Appendix C of the paper):
  Input:  65 = 5-onehot proposer + 60-dim belief
  Hidden: 2 ReLU layers of 80 units each
  Output: 60-dim sigmoid -> w[a] = P(good wins | assignment a)

win_prob_readout (Figure 2 of the paper):
  V_i[j] = sum_a M_i[j, a] * b[a] * payoff_i(a)
  payoff_i(a) = w[a]       if player i is good under ASSIGNMENTS[a]
              = 1 - w[a]   if player i is evil

This is the unnormalized form (no division by sum_a M_i[j,a] * b[a]):
CFR Section 4 needs the unnormalized counterfactual values.

Dtype boundary: net runs in float32; readout promotes to float64 to combine
cleanly with the float64 belief from Section 2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.assignments import ASSIGNMENTS, M_MATRIX, NUM_ASSIGNMENTS
from src.game import NUM_PLAYERS


# ---------------------------------------------------------------------------
# GOOD_MASK[i, a] = 1.0 if player i is good under ASSIGNMENTS[a].
# Used by the readout to pick payoff = w or 1-w per (player, assignment).
# ---------------------------------------------------------------------------

GOOD_MASK: np.ndarray = np.array(
    [[1.0 if rho[i].is_good() else 0.0 for rho in ASSIGNMENTS]
     for i in range(NUM_PLAYERS)],
    dtype=np.float32,
)
assert GOOD_MASK.shape == (NUM_PLAYERS, NUM_ASSIGNMENTS)

# Torch views for callers of the readout.
GOOD_MASK_T: torch.Tensor = torch.from_numpy(GOOD_MASK)               # (5, 60), f32
M_MATRIX_T: torch.Tensor = torch.from_numpy(M_MATRIX)                 # (5, 15, 60), f32


# ---------------------------------------------------------------------------
# ValueNetwork
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """MLP: (proposer one-hot, belief) -> per-assignment P(good wins).

    forward(proposer, belief) returns a (60,) tensor in [0, 1]. The proposer
    one-hot is built internally; callers pass the integer index plus the
    60-dim belief vector.
    """

    INPUT_DIM = NUM_PLAYERS + NUM_ASSIGNMENTS         # 65
    HIDDEN_DIM = 80
    OUTPUT_DIM = NUM_ASSIGNMENTS                      # 60

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, proposer: int, belief: torch.Tensor) -> torch.Tensor:
        assert 0 <= proposer < NUM_PLAYERS, proposer
        assert belief.shape == (NUM_ASSIGNMENTS,), belief.shape
        b32 = belief.to(torch.float32)
        proposer_onehot = torch.zeros(NUM_PLAYERS, dtype=torch.float32,
                                      device=b32.device)
        proposer_onehot[proposer] = 1.0
        x = torch.cat([proposer_onehot, b32])         # (65,)
        return self.net(x)                            # (60,)


# ---------------------------------------------------------------------------
# Win-probability readout (free function — zero learnable parameters)
# ---------------------------------------------------------------------------

def win_prob_readout(
    w: torch.Tensor,
    belief: torch.Tensor,
    M: torch.Tensor = M_MATRIX_T,
    good_mask: torch.Tensor = GOOD_MASK_T,
) -> torch.Tensor:
    """Convert per-assignment P(good wins) to per-info-set values.

    Args:
      w:         (60,)         net output, P(good wins | rho).
      belief:    (60,)         posterior b(rho|h).
      M:         (5, 15, 60)   info-set indicator (M_MATRIX).
      good_mask: (5, 60)       1.0 if player i is good under rho, else 0.0.
    Returns:
      (5, 15) float64 tensor: V[i, j] = sum_a M[i,j,a] * b[a] * payoff_i(a).
      Promoted to float64 to combine cleanly with the float64 belief.
    """
    assert w.shape == (NUM_ASSIGNMENTS,), w.shape
    assert belief.shape == (NUM_ASSIGNMENTS,), belief.shape
    assert M.shape == (NUM_PLAYERS, 15, NUM_ASSIGNMENTS), M.shape
    assert good_mask.shape == (NUM_PLAYERS, NUM_ASSIGNMENTS), good_mask.shape

    w64 = w.to(torch.float64)
    b64 = belief.to(torch.float64)
    M64 = M.to(torch.float64)
    g64 = good_mask.to(torch.float64)

    # payoff[i, a] = g[i,a]*w[a] + (1-g[i,a])*(1-w[a])
    payoff = g64 * w64 + (1.0 - g64) * (1.0 - w64)    # (5, 60)
    weighted = payoff * b64                            # (5, 60), broadcast over i
    return torch.einsum("ija,ia->ij", M64, weighted)  # (5, 15)
