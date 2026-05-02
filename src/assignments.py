"""
Builds on game.py. Provides:
  - Per-assignment role queries (evil_indices, merlin_index, ...)
  - Info-set machinery: each player has 15 distinct info sets at game start
    (1 Resistance + 6 Merlin + 4 Spy + 4 Assassin). See paper, p.5.
  - INFO_SET_INDEX[i, a]: integer in [0, 14] giving player i's info set under
    ASSIGNMENTS[a].
  - M_MATRIX[i]: the (15, 60) one-hot mapping from Fig. 2 of the paper.
"""
from __future__ import annotations

from typing import FrozenSet, List, Tuple

import numpy as np

from src.game import ASSIGNMENTS, NUM_PLAYERS, Role


NUM_ASSIGNMENTS = len(ASSIGNMENTS)
INFO_SETS_PER_PLAYER_COUNT = 15

assert NUM_ASSIGNMENTS == 60, "expected 60 distinct role assignments"


# ---------------------------------------------------------------------------
# Per-assignment role queries
# ---------------------------------------------------------------------------

def evil_indices(rho: Tuple[Role, ...]) -> FrozenSet[int]:
    """Set of player indices whose role is evil (DS or Assassin)."""
    return frozenset(i for i, r in enumerate(rho) if r.is_evil())


def merlin_index(rho: Tuple[Role, ...]) -> int:
    return rho.index(Role.MERLIN)


def assassin_index(rho: Tuple[Role, ...]) -> int:
    return rho.index(Role.ASSASSIN)


def disloyal_index(rho: Tuple[Role, ...]) -> int: # disloyal servant
    return rho.index(Role.DS)


# ---------------------------------------------------------------------------
# Info-set keys
# ---------------------------------------------------------------------------

def info_set_key(player: int, rho: Tuple[Role, ...]) -> tuple:
    """A hashable key for player's starting info set under assignment rho.

    Two assignments yield the same key iff player cannot distinguish them
    from their private role observation alone (before any actions).

      Resistance: knows only own role.                  -> 1 info set
      Merlin   : own role + the 2 evil players (as an unordered set; cannot
                 tell Spy from Assassin).               -> C(4,2) = 6 info sets
      Spy      : own role + Assassin's position.        -> 4 info sets
      Assassin : own role + Spy's position.             -> 4 info sets
    """
    role = rho[player]
    if role is Role.LS:
        return (Role.LS,)
    if role is Role.MERLIN:
        return (Role.MERLIN, evil_indices(rho))
    if role is Role.DS:
        return (Role.DS, assassin_index(rho))
    if role is Role.ASSASSIN:
        return (Role.ASSASSIN, disloyal_index(rho))
    raise ValueError(f"Unknown role: {role!r}")


# ---------------------------------------------------------------------------
# Info-set tables
# ---------------------------------------------------------------------------
# INFO_SETS_PER_PLAYER[i] : list of 15 info-set keys for player i.
# INFO_SET_INDEX[i, a]    : index of player i's info set under ASSIGNMENTS[a].
# M_MATRIX[i]             : (15, 60) one-hot matrix; column a has a 1 in the
#                           row that is player i's info set under assignment a.

INFO_SETS_PER_PLAYER: List[List[tuple]] = []
INFO_SET_INDEX = np.zeros((NUM_PLAYERS, NUM_ASSIGNMENTS), dtype=np.int32)

for _player in range(NUM_PLAYERS):
    _keys: List[tuple] = []
    _key_to_idx: dict = {}
    for _a_idx, _rho in enumerate(ASSIGNMENTS):
        _k = info_set_key(_player, _rho)
        if _k not in _key_to_idx:
            _key_to_idx[_k] = len(_keys)    # assign next integer index
            _keys.append(_k)
        INFO_SET_INDEX[_player, _a_idx] = _key_to_idx[_k]
    INFO_SETS_PER_PLAYER.append(_keys)
    assert len(_keys) == INFO_SETS_PER_PLAYER_COUNT, (
        f"player {_player} produced {len(_keys)} info sets, expected "
        f"{INFO_SETS_PER_PLAYER_COUNT}"
    )

M_MATRIX = np.zeros(
    (NUM_PLAYERS, INFO_SETS_PER_PLAYER_COUNT, NUM_ASSIGNMENTS), dtype=np.float32
)

for _player in range(NUM_PLAYERS):
    for _a_idx in range(NUM_ASSIGNMENTS):
        M_MATRIX[_player, INFO_SET_INDEX[_player, _a_idx], _a_idx] = 1.0


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------

def print_info_sets(player: int) -> None:
    print(f"\n=== Player {player} Info Sets ===")
    
    # Group assignments by info set index
    from collections import defaultdict
    groups = defaultdict(list)
    for a_idx, rho in enumerate(ASSIGNMENTS):
        i = INFO_SET_INDEX[player, a_idx]
        groups[i].append((a_idx, rho))
        
    keys = INFO_SETS_PER_PLAYER[player]
    for i, key in enumerate(keys):
        print(f"\n  Info Set {i:2d} | Key: {fmt_key(key)}")
        print(f"  {'Idx':>4}  Assignment")
        for a_idx, rho in groups[i]:
            print(f"  {a_idx:>4}  {fmt_rho(rho)}")

def fmt_key(key: tuple) -> str:
    """Human-readable info set key."""
    role = key[0]
    if role is Role.LS:
        return "LS (sees nothing)"
    if role is Role.MERLIN:
        return f"Merlin | evil seats = {set(key[1])}"
    if role is Role.DS:
        return f"Spy    | assassin @ seat {key[1]}"
    if role is Role.ASSASSIN:
        return f"Assassin | spy @ seat {key[1]}"

def fmt_rho(rho: tuple) -> str:
    """Print assignment as seat->role mapping."""
    return "  ".join(f"s{i}:{r.name[:3]}" for i, r in enumerate(rho))