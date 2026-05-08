"""Tests for regret_matching.py — positive-part normalization + uniform fallback."""
import numpy as np
import pytest

from src.regret_matching import regret_matching_plus


def test_all_zero_regrets_uniform():
    r = np.zeros(4)
    s = regret_matching_plus(r)
    np.testing.assert_allclose(s, np.full(4, 0.25))


def test_all_negative_regrets_uniform():
    r = np.array([-1.0, -2.0, -3.0])
    s = regret_matching_plus(r)
    np.testing.assert_allclose(s, np.full(3, 1 / 3))


def test_pure_positive_normalize():
    r = np.array([1.0, 3.0, 0.0, 0.0])
    s = regret_matching_plus(r)
    np.testing.assert_allclose(s, np.array([0.25, 0.75, 0.0, 0.0]))


def test_mixed_signs_zero_negatives():
    r = np.array([-1.0, 2.0, -5.0, 6.0])
    s = regret_matching_plus(r)
    np.testing.assert_allclose(s, np.array([0.0, 0.25, 0.0, 0.75]))


def test_strategy_sums_to_one_per_row():
    rng = np.random.default_rng(0)
    r = rng.standard_normal((10, 5))
    s = regret_matching_plus(r)
    assert s.shape == (10, 5)
    np.testing.assert_allclose(s.sum(axis=-1), np.ones(10), atol=1e-12)


def test_batch_shape_preserved():
    """Works on (batch1, batch2, |A|) input."""
    rng = np.random.default_rng(42)
    r = rng.standard_normal((3, 7, 4))
    s = regret_matching_plus(r)
    assert s.shape == (3, 7, 4)
    np.testing.assert_allclose(s.sum(axis=-1), np.ones((3, 7)), atol=1e-12)


def test_strategy_in_unit_simplex():
    rng = np.random.default_rng(1)
    r = rng.standard_normal((20, 6))
    s = regret_matching_plus(r)
    assert np.all(s >= 0.0)
    assert np.all(s <= 1.0 + 1e-12)
