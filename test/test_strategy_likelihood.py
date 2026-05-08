"""Tests for strategy_likelihood.py — gather-and-product over (player, info-set)."""
import numpy as np
import pytest

from src.assignments import INFO_SET_INDEX, NUM_ASSIGNMENTS
from src.game import NUM_PLAYERS
from src.strategy_likelihood import strategy_aware_likelihood


def test_all_ones_yields_all_ones():
    """Reach prob = 1 everywhere → likelihood factor = 1 per ρ."""
    rp = np.ones((NUM_PLAYERS, 15), dtype=np.float64)
    lik = strategy_aware_likelihood(rp)
    np.testing.assert_array_equal(lik, np.ones(NUM_ASSIGNMENTS))


def test_zero_in_one_info_set_zeros_those_assignments():
    """Zeroing player 0's info-set 0 zeros every ρ where INFO_SET_INDEX[0, ρ] = 0."""
    rp = np.ones((NUM_PLAYERS, 15), dtype=np.float64)
    rp[0, 0] = 0.0
    lik = strategy_aware_likelihood(rp)
    for a in range(NUM_ASSIGNMENTS):
        if INFO_SET_INDEX[0, a] == 0:
            assert lik[a] == 0.0
        else:
            assert lik[a] == 1.0


def test_factorizes_per_player():
    """Independent player factors multiply: lik(ρ) = prod_i rp[i, INFO_SET_INDEX[i, ρ]]."""
    rng = np.random.default_rng(0)
    rp = rng.uniform(0.5, 1.5, size=(NUM_PLAYERS, 15))
    lik = strategy_aware_likelihood(rp)
    for a in range(NUM_ASSIGNMENTS):
        expected = 1.0
        for i in range(NUM_PLAYERS):
            expected *= rp[i, INFO_SET_INDEX[i, a]]
        assert lik[a] == pytest.approx(expected)


def test_shape_and_dtype():
    rp = np.full((NUM_PLAYERS, 15), 0.5, dtype=np.float64)
    lik = strategy_aware_likelihood(rp)
    assert lik.shape == (NUM_ASSIGNMENTS,)
    assert lik.dtype == np.float64


def test_rejects_wrong_shape():
    with pytest.raises(AssertionError):
        strategy_aware_likelihood(np.ones((NUM_PLAYERS, 14)))
    with pytest.raises(AssertionError):
        strategy_aware_likelihood(np.ones((4, 15)))
