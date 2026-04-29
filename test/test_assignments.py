"""Tests for assignments.py — info-set structure and M_MATRIX."""
import numpy as np
import pytest

from src.game import Role
from src.assignments import (
    ASSIGNMENTS,
    NUM_ASSIGNMENTS,
    INFO_SETS_PER_PLAYER,
    INFO_SET_INDEX,
    M_MATRIX,
    evil_indices,
    merlin_index,
    assassin_index,
    disloyal_index,
    info_set_key,
)


def test_assignment_count():
    assert NUM_ASSIGNMENTS == 60
    assert len(ASSIGNMENTS) == 60


def test_role_multiset_in_each_assignment():
    """Every assignment has exactly the multiset {M, R, R, S, A}."""
    expected = sorted([
        Role.MERLIN, Role.LS, Role.LS,
        Role.DS, Role.ASSASSIN,
    ], key=lambda r: r.value)
    for rho in ASSIGNMENTS:
        actual = sorted(rho, key=lambda r: r.value)
        assert actual == expected


def test_evil_indices_size_2():
    """Every assignment has exactly 2 evil players."""
    for rho in ASSIGNMENTS:
        assert len(evil_indices(rho)) == 2


def test_role_position_counts():
    """Across the 60 assignments, each player position holds each role with
    counts: M=12, R=24, S=12, A=12."""
    for player in range(5):
        counts = {Role.MERLIN: 0, Role.LS: 0, Role.DS: 0, Role.ASSASSIN: 0}
        for rho in ASSIGNMENTS:
            counts[rho[player]] += 1
        assert counts == {
            Role.MERLIN: 12, Role.LS: 24,
            Role.DS: 12, Role.ASSASSIN: 12,
        }, f"player {player}: {counts}"


def test_15_info_sets_per_player_with_role_breakdown():
    """Each player has 15 info sets: 1 LS + 6 Merlin + 4 DS + 4 Assassin."""
    for player in range(5):
        keys = INFO_SETS_PER_PLAYER[player]
        assert len(keys) == 15
        n_L = sum(1 for k in keys if k[0] is Role.LS)
        n_M = sum(1 for k in keys if k[0] is Role.MERLIN)
        n_D = sum(1 for k in keys if k[0] is Role.DS)
        n_A = sum(1 for k in keys if k[0] is Role.ASSASSIN)
        assert (n_L, n_M, n_D, n_A) == (1, 6, 4, 4), (n_L, n_M, n_D, n_A)


def test_M_matrix_shape_and_one_hot():
    """M_MATRIX has shape (5, 15, 60); each column has exactly one 1."""
    assert M_MATRIX.shape == (5, 15, 60)
    col_sums = M_MATRIX.sum(axis=1)         # (5, 60)
    assert np.all(col_sums == 1.0)
    # Each player's full matrix sums to 60 (one entry per assignment).
    for p in range(5):
        assert M_MATRIX[p].sum() == 60.0


def test_assignments_per_info_set():
    """Within each player's 15 info sets, the assignment counts are:
       LS: 24, Merlin: 2 each, DS: 3 each, Assassin: 3 each."""
    for player in range(5):
        per_set_counts = M_MATRIX[player].sum(axis=1)  # (15,)
        sorted_counts = sorted(per_set_counts.astype(int).tolist())
        # 1 set of size 24 + 6 of size 2 + 4 of size 3 + 4 of size 3
        expected = sorted([24] + [2] * 6 + [3] * 4 + [3] * 4)
        assert sorted_counts == expected


def test_info_set_key_consistency_with_INFO_SET_INDEX():
    """Two assignments share an INFO_SET_INDEX iff they share an info_set_key."""
    for player in range(5):
        for a in range(NUM_ASSIGNMENTS):
            for b in range(NUM_ASSIGNMENTS):
                same_idx = INFO_SET_INDEX[player, a] == INFO_SET_INDEX[player, b]
                same_key = info_set_key(player, ASSIGNMENTS[a]) == info_set_key(
                    player, ASSIGNMENTS[b]
                )
                assert same_idx == same_key


def test_merlin_spy_assassin_index_helpers():
    """The role-position helpers actually find the right player."""
    for rho in ASSIGNMENTS:
        assert rho[merlin_index(rho)] is Role.MERLIN
        assert rho[disloyal_index(rho)] is Role.DS
        assert rho[assassin_index(rho)] is Role.ASSASSIN
