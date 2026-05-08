"""
Regret matching+ (Tammelin et al., 2015), the per-info-set primitive used in
Algorithm 1 line 20 of Serrino et al. (2019).

Strategy from regrets:
  σ(a) = max(r(a), 0) / Σ_b max(r(b), 0)   if denominator > 0
  σ(a) = 1 / |A|                           otherwise (uniform fallback)

CFR+ additionally floors *cumulative* regrets at zero after each iteration.
That floor is applied in cfr.py — this module is only the strategy readout.
"""
from __future__ import annotations

import numpy as np


def regret_matching_plus(regrets: np.ndarray) -> np.ndarray:
    """Convert a regret table to a strategy distribution.

    Args:
      regrets: (..., |A|) array of cumulative regrets. Trailing dim is actions.
    Returns:
      Same shape; each row of length |A| sums to 1.
    """
    assert regrets.ndim >= 1
    pos = np.maximum(regrets, 0.0)
    total = pos.sum(axis=-1, keepdims=True)
    n_actions = regrets.shape[-1]
    safe_total = np.where(total > 0, total, 1.0)
    normalized = pos / safe_total
    uniform = np.full_like(normalized, 1.0 / n_actions)
    return np.where(total > 0, normalized, uniform)
