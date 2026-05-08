"""Regression tests for paper-faithful joint-factorized mission deduction.

Replaces the marginal-uniform approximation for the (k_I=2, f=1) ambiguous
case with the responsibility-weighted joint factorization from
cfr_plus.cpp::my_single_pass_responsibility (Detry322/DeepRole).
"""
import numpy as np
import pytest

from src.assignments import INFO_SETS_PER_PLAYER
from src.cfr import (
    _AMBIGUOUS_SENTINEL,
    _evil_action_dist,
    _partner_viewpoint,
    _responsibility_factor,
    MissionState,
    deduce_actions,
)
from src.game import Role


# ---------------------------------------------------------------------------
# Responsibility factor — invariants from the C++ reference
# ---------------------------------------------------------------------------

def test_responsibility_factorization_reconstructs_joint():
    """f_me * f_partner == outcome_prob exactly, by construction."""
    cases = [
        (0.5, 0.5),
        (0.3, 0.7),
        (0.0, 0.5),
        (1.0, 0.5),
        (0.99, 0.01),
        (0.01, 0.99),
        (0.4, 0.4),
        (0.25, 0.75),
    ]
    for my_pass, partner_pass in cases:
        outcome = my_pass * (1 - partner_pass) + (1 - my_pass) * partner_pass
        my_f = _responsibility_factor(my_pass, partner_pass)
        partner_f = _responsibility_factor(partner_pass, my_pass)
        assert abs(my_f * partner_f - outcome) < 1e-12, (
            my_pass, partner_pass, my_f, partner_f, outcome
        )


def test_responsibility_factorization_random_pairs():
    """100 random (my_pass, partner_pass) pairs all satisfy the invariant."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        my, par = float(rng.uniform()), float(rng.uniform())
        outcome = my * (1 - par) + (1 - my) * par
        my_f = _responsibility_factor(my, par)
        par_f = _responsibility_factor(par, my)
        assert abs(my_f * par_f - outcome) < 1e-12


def test_responsibility_factor_symmetric_case_matches_sqrt():
    """When my_pass == partner_pass, my_resp == partner_resp == 0.5,
    so my_factor = outcome_prob ** 0.5 = sqrt(2p(1-p))."""
    for p in [0.2, 0.5, 0.7]:
        outcome = 2 * p * (1 - p)
        my_f = _responsibility_factor(p, p)
        assert abs(my_f - outcome ** 0.5) < 1e-12


def test_responsibility_factor_zero_outcome_returns_zero():
    """If both players are deterministic in the same direction, outcome_prob=0."""
    assert _responsibility_factor(0.0, 0.0) == 0.0   # both fail → 2 fails, not 1
    assert _responsibility_factor(1.0, 1.0) == 0.0   # both succeed → 0 fails, not 1


def test_responsibility_factor_one_deterministic_other_random():
    """If my_pass=1 deterministically, all responsibility goes to partner."""
    # my_pass=1, partner_pass=0.3 → outcome = 1*0.7 + 0*0.3 = 0.7
    # my_var = 1, partner_var = 0.58, my_exponent = 1/1.58 ≈ 0.633
    # my_factor = 0.7^0.633 ≈ 0.798
    # partner_factor = 0.7^0.367 ≈ 0.877
    # product = 0.798 * 0.877 ≈ 0.7 ✓
    my_f = _responsibility_factor(1.0, 0.3)
    par_f = _responsibility_factor(0.3, 1.0)
    assert abs(my_f * par_f - 0.7) < 1e-12


# ---------------------------------------------------------------------------
# _evil_action_dist
# ---------------------------------------------------------------------------

def test_evil_action_dist_returns_none_only_for_k2_f1():
    """The only ambiguous case in 5p Avalon is (f=1, k_I=2)."""
    assert _evil_action_dist(1, 2) is None
    # All other (f, k_I) combinations with f, k_I ∈ {0, 1, 2} return tuples.
    determined = [
        (0, 0), (0, 1), (0, 2),
        (1, 1),
        (2, 1), (2, 2),
    ]
    for f, k in determined:
        result = _evil_action_dist(f, k)
        assert result is not None, (f, k)
        assert isinstance(result, tuple) and len(result) == 2


def test_evil_action_dist_raises_on_unexpected_input():
    """5p Avalon allows f, k_I in {0, 1, 2}. Anything else is a bug."""
    with pytest.raises(ValueError):
        _evil_action_dist(1, 3)
    with pytest.raises(ValueError):
        _evil_action_dist(3, 5)


# ---------------------------------------------------------------------------
# _partner_viewpoint — symmetric-pair lookup
# ---------------------------------------------------------------------------

def test_partner_viewpoint_round_trips():
    """If player p_a is in info-set j_a (key (DS, p_b)), then p_b's symmetric
    info-set has key (ASSASSIN, p_a) — and round-tripping back yields j_a."""
    p_a = 0
    for j_a, key in enumerate(INFO_SETS_PER_PLAYER[p_a]):
        if key[0] not in (Role.DS, Role.ASSASSIN):
            continue
        p_b = key[1]
        j_b = _partner_viewpoint(p_a, j_a, p_b)
        # Partner's key should be (other_role, p_a).
        partner_key = INFO_SETS_PER_PLAYER[p_b][j_b]
        assert partner_key[1] == p_a
        assert partner_key[0] != key[0]   # role swapped
        # Round trip back.
        j_a_back = _partner_viewpoint(p_b, j_b, p_a)
        assert j_a_back == j_a


def test_partner_viewpoint_rejects_non_evil_role():
    """Calling on an LS / Merlin info-set is a bug (no partner relation)."""
    p_a = 0
    for j_a, key in enumerate(INFO_SETS_PER_PLAYER[p_a]):
        if key[0] is Role.LS:
            with pytest.raises(ValueError):
                _partner_viewpoint(p_a, j_a, partner=1)
            return
    pytest.fail("no LS info-set found for player 0")


# ---------------------------------------------------------------------------
# deduce_actions: sentinel placement at ambiguous mission rows
# ---------------------------------------------------------------------------

def test_deduce_mission_sentinel_only_in_k2_f1_evil_rows():
    """Sentinel rows appear at exactly the (k_I=2, f=1) evil info-sets."""
    team = (0, 1)
    s = MissionState(round_idx=0, proposer=0, successes=0, failures=0, team=team)
    actions = deduce_actions(s, 1)
    for player_idx in team:
        a = actions[player_idx]
        for j, key in enumerate(INFO_SETS_PER_PLAYER[player_idx]):
            role = key[0]
            has_sentinel = (a[j] == _AMBIGUOUS_SENTINEL).any()
            if role in (Role.DS, Role.ASSASSIN):
                partner_seat = key[1]
                k_I = 1 + (1 if partner_seat in team else 0)
                if k_I == 2:
                    assert has_sentinel
                else:
                    assert not has_sentinel
            else:
                # Good roles never carry sentinel.
                assert not has_sentinel


def test_deduce_mission_no_sentinels_when_observation_unambiguous():
    """f=0 or f=2 are deterministic for all info-sets; no sentinels emitted."""
    s = MissionState(round_idx=0, proposer=0, successes=0, failures=0, team=(0, 1))
    for f in (0, 2):
        actions = deduce_actions(s, f)
        for a in actions.values():
            assert not (a == _AMBIGUOUS_SENTINEL).any(), f
