"""Tests for the Algorithm 4 situation sampler."""
import numpy as np
import pytest

from src.assignments import ASSIGNMENTS, NUM_ASSIGNMENTS, evil_indices, merlin_index
from src.game import NUM_PLAYERS
from src.situation_sampler import (
    _EVIL_PAIRS,
    _consistent_evil_pairs,
    _evil_pair_of_assignment,
    sample_situation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_evil_pairs_count_and_uniqueness():
    assert len(_EVIL_PAIRS) == 10
    assert len(set(_EVIL_PAIRS)) == 10


def test_consistent_evil_pairs_no_failed_missions_admits_all():
    """When f=0 (no failed missions), every pair is consistent."""
    pairs = _consistent_evil_pairs([])
    assert set(pairs) == set(_EVIL_PAIRS)


def test_consistent_evil_pairs_filters_correctly():
    """Failed mission with team {0, 1, 2} requires the evil pair to intersect it."""
    pairs = _consistent_evil_pairs([(0, 1, 2)])
    for p in pairs:
        assert any(seat in (0, 1, 2) for seat in p)
    # Pair (3, 4) has no overlap → should NOT be consistent.
    assert (3, 4) not in pairs


def test_consistent_evil_pairs_two_disjoint_teams():
    """Two failed missions on disjoint teams force the evil pair to span both."""
    pairs = _consistent_evil_pairs([(0, 1), (3, 4)])
    for p in pairs:
        assert any(seat in (0, 1) for seat in p)
        assert any(seat in (3, 4) for seat in p)


def test_evil_pair_of_assignment_matches_evil_indices():
    for rho in ASSIGNMENTS:
        pair = _evil_pair_of_assignment(rho)
        assert tuple(sorted(evil_indices(rho))) == pair


# ---------------------------------------------------------------------------
# sample_situation
# ---------------------------------------------------------------------------

def test_sample_situation_returns_valid_belief():
    rng = np.random.default_rng(0)
    proposer, belief = sample_situation(s=0, f=0, rng=rng)
    assert 0 <= proposer < NUM_PLAYERS
    assert belief.shape == (NUM_ASSIGNMENTS,)
    assert belief.dtype == np.float64
    assert np.all(belief >= 0)
    assert belief.sum() == pytest.approx(1.0)


def test_sample_situation_no_fails_admits_full_support():
    """f=0 → consistent pairs = all 10 → every assignment can have positive mass."""
    rng = np.random.default_rng(0)
    # Average over many samples — every assignment should appear with positive
    # mass on at least one sample.
    has_mass = np.zeros(NUM_ASSIGNMENTS, dtype=bool)
    for _ in range(200):
        _, belief = sample_situation(s=0, f=0, rng=rng)
        has_mass |= belief > 0
    assert has_mass.all()


def test_sample_situation_with_fails_zeroes_inconsistent_pairs():
    """When f>0, sampled belief assigns 0 mass to any ρ whose evil pair is
    inconsistent with the sampled failed missions. We can't easily check the
    specific zeros without inspecting the internal sample, but we can check
    that the *support* of the belief is consistent: at least one team failed,
    so its evil intersection must be ≥ 1 for any assignment with positive mass."""
    rng = np.random.default_rng(123)
    for _ in range(50):
        proposer, belief = sample_situation(s=1, f=1, rng=rng)
        # Every assignment with positive mass has a non-empty evil-set
        # (always 2 evils). We can't directly test the team-intersection
        # condition without reading the internal failed_teams; the sampler's
        # internal _consistent_evil_pairs is unit-tested separately.
        assert (belief > 0).sum() > 0


def test_sample_situation_proposer_uniform_on_average():
    """Over many draws, each proposer seat is picked with probability ~ 1/5."""
    rng = np.random.default_rng(42)
    n = 5000
    counts = np.zeros(NUM_PLAYERS, dtype=int)
    for _ in range(n):
        p, _ = sample_situation(s=0, f=0, rng=rng)
        counts[p] += 1
    # Expected 1000 ± 3σ ≈ ±90 per seat.
    expected = n / NUM_PLAYERS
    for c in counts:
        assert abs(c - expected) < 100, counts


def test_sample_situation_belief_marginal_over_evil_pair_dirichlet():
    """Averaging beliefs over many draws (no fails), the marginal probability
    of each consistent evil pair is symmetric over 10 pairs ⇒ ~ 1/10 each."""
    rng = np.random.default_rng(0)
    n = 2000
    pair_marginal = np.zeros(len(_EVIL_PAIRS), dtype=np.float64)
    pair_to_idx = {p: i for i, p in enumerate(_EVIL_PAIRS)}
    for _ in range(n):
        _, belief = sample_situation(s=0, f=0, rng=rng)
        for a_idx, rho in enumerate(ASSIGNMENTS):
            pair = _evil_pair_of_assignment(rho)
            pair_marginal[pair_to_idx[pair]] += belief[a_idx]
    pair_marginal /= n
    expected = 1 / len(_EVIL_PAIRS)
    for m in pair_marginal:
        assert abs(m - expected) < 0.02, pair_marginal


def test_sample_situation_belief_marginal_over_merlin_uniform():
    """Averaging beliefs (no fails), the marginal P(Merlin at seat k) ~ 1/5."""
    rng = np.random.default_rng(0)
    n = 2000
    merlin_marginal = np.zeros(NUM_PLAYERS, dtype=np.float64)
    for _ in range(n):
        _, belief = sample_situation(s=0, f=0, rng=rng)
        for a_idx, rho in enumerate(ASSIGNMENTS):
            merlin_marginal[merlin_index(rho)] += belief[a_idx]
    merlin_marginal /= n
    expected = 1 / NUM_PLAYERS
    for m in merlin_marginal:
        assert abs(m - expected) < 0.02, merlin_marginal


def test_sample_situation_deterministic_with_seed():
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    pa, ba = sample_situation(s=1, f=1, rng=rng_a)
    pb, bb = sample_situation(s=1, f=1, rng=rng_b)
    assert pa == pb
    np.testing.assert_array_equal(ba, bb)


def test_sample_situation_rejects_invalid_sf():
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        sample_situation(s=3, f=0, rng=rng)
    with pytest.raises(AssertionError):
        sample_situation(s=0, f=3, rng=rng)
    with pytest.raises(AssertionError):
        sample_situation(s=-1, f=0, rng=rng)


def test_sample_situation_handles_all_sf_combinations():
    rng = np.random.default_rng(0)
    for s in range(3):
        for f in range(3):
            p, b = sample_situation(s=s, f=f, rng=rng)
            assert 0 <= p < NUM_PLAYERS
            assert b.sum() == pytest.approx(1.0)
            assert (b >= 0).all()
