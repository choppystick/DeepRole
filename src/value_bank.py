"""
ValueBank: 45 ValueNetwork instances indexed by (proposer, s, f).

Per the paper, every proposal-phase history h maps to one of 45 networks based
on (proposer, successes_so_far, failures_so_far). Each network has its own
parameters; there is no weight sharing.

Key set: 5 proposers * 3 success counts (0,1,2) * 3 failure counts (0,1,2) = 45.
States with s=3 or f=3 are terminal (Arthur reaches assassination, or Mordred
wins) and have no associated network.
"""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from src.game import NUM_PLAYERS
from src.value_net import ValueNetwork


VALID_KEYS: Tuple[Tuple[int, int, int], ...] = tuple(
    (p, s, f) for p in range(NUM_PLAYERS) for s in range(3) for f in range(3)
)
assert len(VALID_KEYS) == 45


class ValueBank(nn.Module):
    """Holds 45 ValueNetwork submodules, dispatched by (proposer, s, f).

    Subclasses nn.Module so PyTorch sees each network as a submodule
    (parameters() walks them, .to(device) propagates).
    """

    def __init__(self) -> None:
        super().__init__()
        self.nets = nn.ModuleDict({
            self._key_str(p, s, f): ValueNetwork()
            for (p, s, f) in VALID_KEYS
        })

    @staticmethod
    def _key_str(proposer: int, successes: int, failures: int) -> str:
        return f"{proposer}_{successes}_{failures}"

    def __contains__(self, key: Tuple[int, int, int]) -> bool:
        try:
            return self._key_str(*key) in self.nets
        except TypeError:
            return False

    def get(self, proposer: int, successes: int, failures: int) -> ValueNetwork:
        k = self._key_str(proposer, successes, failures)
        if k not in self.nets:
            raise KeyError(
                f"No value network for (proposer={proposer}, s={successes}, "
                f"f={failures}). Valid: 0 <= proposer < 5, 0 <= s < 3, 0 <= f < 3."
            )
        return self.nets[k]

    def keys(self) -> Tuple[Tuple[int, int, int], ...]:
        return VALID_KEYS
