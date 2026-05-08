"""Tests for the assassinate solver added in Section 5.

Replaces the placeholder TerminalState(winner="assassinate_pending") with a
proper game-theoretic resolution: a MissionState's third success transitions
to AssassinateState (decision node, 5 targets), and each target leads to an
AssassinateOutcomeState whose per-ρ utility is determined by whether
target == seat-of-Merlin(ρ).
"""
import numpy as np
import pytest
import torch

from src.assignments import ASSIGNMENTS, INFO_SETS_PER_PLAYER, NUM_ASSIGNMENTS
from src.cfr import (
    AssassinateOutcomeState,
    AssassinateState,
    CFRSolver,
    MissionState,
    ProposeState,
    TerminalState,
    _ASSASSIN_UTIL_CACHE,
    child_state,
    deduce_actions,
    is_leaf,
    legal_observations,
    moving_players,
    num_actions,
)
from src.game import NUM_PLAYERS, Role
from src.value_bank import ValueBank


# ---------------------------------------------------------------------------
# Transition: MissionState → AssassinateState
# ---------------------------------------------------------------------------

def test_mission_with_third_success_routes_to_assassinate():
    """The pre-Section-5 placeholder TerminalState(winner='assassinate_pending')
    no longer exists. The transition now goes to AssassinateState."""
    s = MissionState(round_idx=2, proposer=0, successes=2, failures=0, team=(0, 1, 2))
    child = child_state(s, 0)
    assert isinstance(child, AssassinateState)
    assert not isinstance(child, TerminalState)


def test_mission_with_third_failure_still_terminates_spy():
    """3 failures still goes to a true terminal (spy win), not assassinate."""
    s = MissionState(round_idx=2, proposer=0, successes=1, failures=2, team=(0, 1, 2))
    child = child_state(s, 1)
    assert isinstance(child, TerminalState)
    assert child.winner == "spy"


# ---------------------------------------------------------------------------
# AssassinateState as a decision node
# ---------------------------------------------------------------------------

def test_assassinate_state_is_not_a_leaf():
    assert not is_leaf(AssassinateState())


def test_assassinate_outcome_state_is_a_leaf():
    assert is_leaf(AssassinateOutcomeState(target=0))


def test_assassinate_state_legal_observations_are_5_targets():
    assert legal_observations(AssassinateState()) == (0, 1, 2, 3, 4)


def test_assassinate_state_all_5_players_are_movers():
    """All 5 players are nominally movers since the actual Assassin depends on ρ."""
    assert moving_players(AssassinateState()) == tuple(range(NUM_PLAYERS))


def test_assassinate_state_num_actions_per_player_is_5():
    for i in range(NUM_PLAYERS):
        assert num_actions(AssassinateState(), i) == 5


def test_assassinate_state_child_is_outcome_state():
    for target in range(NUM_PLAYERS):
        c = child_state(AssassinateState(), target)
        assert isinstance(c, AssassinateOutcomeState)
        assert c.target == target


# ---------------------------------------------------------------------------
# deduce_actions: per-info-set behavior at AssassinateState
# ---------------------------------------------------------------------------

def test_deduce_actions_assassin_info_set_one_hot_at_target():
    """In info-sets where i is the Assassin, ⃗a_i[j, target] = 1, others 0."""
    target = 3
    actions = deduce_actions(AssassinateState(), target)
    for i in range(NUM_PLAYERS):
        a = actions[i]
        for j, key in enumerate(INFO_SETS_PER_PLAYER[i]):
            if key[0] is Role.ASSASSIN:
                assert a[j, target] == 1.0
                # All other action columns are zero in Assassin info-sets.
                others = [k for k in range(NUM_PLAYERS) if k != target]
                assert (a[j, others] == 0.0).all()


def test_deduce_actions_non_assassin_info_set_is_all_ones():
    """In info-sets where i is NOT the Assassin, ⃗a_i[j, :] = 1 (no-op)."""
    actions = deduce_actions(AssassinateState(),2)
    for i in range(NUM_PLAYERS):
        a = actions[i]
        for j, key in enumerate(INFO_SETS_PER_PLAYER[i]):
            if key[0] is not Role.ASSASSIN:
                np.testing.assert_array_equal(a[j], np.ones(NUM_PLAYERS))


def test_deduce_actions_factor_is_one_for_non_assassin_info_sets():
    """Factor (σ · a).sum should equal 1 in non-Assassin info-sets when σ
    is a probability distribution. This is the no-op semantics — these
    info-sets don't move at AssassinateState."""
    actions = deduce_actions(AssassinateState(),0)
    rng = np.random.default_rng(0)
    for i in range(NUM_PLAYERS):
        a = actions[i]
        # Random probability distribution per info-set.
        sigma = rng.dirichlet(np.ones(NUM_PLAYERS), size=15)
        factor = (sigma * a).sum(axis=-1)   # (15,)
        for j, key in enumerate(INFO_SETS_PER_PLAYER[i]):
            if key[0] is not Role.ASSASSIN:
                assert factor[j] == pytest.approx(1.0)


def test_deduce_actions_factor_is_strategy_at_target_for_assassin_info_sets():
    """In Assassin info-sets, factor = σ_i[j, target]."""
    target = 1
    actions = deduce_actions(AssassinateState(), target)
    rng = np.random.default_rng(0)
    for i in range(NUM_PLAYERS):
        a = actions[i]
        sigma = rng.dirichlet(np.ones(NUM_PLAYERS), size=15)
        factor = (sigma * a).sum(axis=-1)
        for j, key in enumerate(INFO_SETS_PER_PLAYER[i]):
            if key[0] is Role.ASSASSIN:
                assert factor[j] == pytest.approx(sigma[j, target])


# ---------------------------------------------------------------------------
# _ASSASSIN_UTIL_CACHE — per-target per-(player, ρ) utility
# ---------------------------------------------------------------------------

def test_assassin_util_cache_has_5_targets():
    assert set(_ASSASSIN_UTIL_CACHE.keys()) == set(range(NUM_PLAYERS))
    for util in _ASSASSIN_UTIL_CACHE.values():
        assert util.shape == (NUM_PLAYERS, NUM_ASSIGNMENTS)
        assert util.dtype == np.float64


def test_assassin_util_cache_consistent_with_outcome_rule():
    """For each (target, ρ): if Merlin(ρ).seat == target, evil wins (util[i,ρ]=+1
    iff i evil); else resistance wins (util[i,ρ]=+1 iff i good)."""
    for target in range(NUM_PLAYERS):
        util = _ASSASSIN_UTIL_CACHE[target]
        for a_idx, rho in enumerate(ASSIGNMENTS):
            merlin_seat = rho.index(Role.MERLIN)
            spy_wins = (merlin_seat == target)
            for i in range(NUM_PLAYERS):
                expected = (
                    (+1.0 if rho[i].is_evil() else -1.0)
                    if spy_wins
                    else (+1.0 if rho[i].is_good() else -1.0)
                )
                assert util[i, a_idx] == expected


def test_assassin_util_zero_sum_per_assignment():
    """In every (target, ρ) pair, sum across players = winners − losers.
    With 3 good + 2 evil: spy wins ⇒ +2 + (−3) = −1; resistance wins ⇒ +3 + (−2) = +1."""
    for target in range(NUM_PLAYERS):
        util = _ASSASSIN_UTIL_CACHE[target]
        for a_idx, rho in enumerate(ASSIGNMENTS):
            merlin_seat = rho.index(Role.MERLIN)
            spy_wins = (merlin_seat == target)
            expected_sum = -1.0 if spy_wins else +1.0
            assert util[:, a_idx].sum() == pytest.approx(expected_sum)


# ---------------------------------------------------------------------------
# CFR solve through assassinate
# ---------------------------------------------------------------------------

def _root_with_assassinate_reachable() -> ProposeState:
    """Subgame from (round=4, s=2, f=0) — one mission left, success = assassinate.

    Round 4 has team_size 3, max_rejections at 4 means only one shot to vote;
    and a 0-fail outcome takes us into AssassinateState.
    """
    return ProposeState(
        round_idx=4, proposer=0, successes=2, failures=0,
        rejected_count=4,   # one attempt left
    )


def test_cfr_dag_includes_assassinate_states_when_reachable():
    """The state DAG built from the assassinate-reachable root contains
    both AssassinateState and AssassinateOutcomeState nodes."""
    bank = ValueBank()
    solver = CFRSolver(_root_with_assassinate_reachable(), bank)
    states = list(solver._states.values())
    has_assassinate = any(isinstance(s, AssassinateState) for s, _ in states)
    has_outcome = any(isinstance(s, AssassinateOutcomeState) for s, _ in states)
    assert has_assassinate
    assert has_outcome
    n_outcomes = sum(1 for s, _ in states if isinstance(s, AssassinateOutcomeState))
    # 5 targets × possibly distinct masks. With this subgame the mission has
    # 4 outcomes (team-of-3 with f=0..3), but only f=0 leads to assassinate;
    # so all 5 outcomes share the SAME mask (f=0 doesn't constrain).
    assert n_outcomes == 5


def test_cfr_runs_through_assassinate_without_crashing():
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_root_with_assassinate_reachable(), bank)
    avg = solver.solve(n_iters=3)
    # Strategies remain valid distributions including at AssassinateState.
    for key, sigma in avg.items():
        assert np.all(sigma >= 0)
        np.testing.assert_allclose(sigma.sum(axis=-1), np.ones(sigma.shape[0]), atol=1e-9)


def test_assassin_strategy_picks_likely_merlin_at_assassinate():
    """When the Assassin's info-set strongly identifies Merlin, the
    converged strategy should put high mass on guessing that seat.

    Construct: a prior belief that puts almost all mass on a specific ρ
    where Merlin sits at seat 2. Run enough CFR iterations and confirm the
    Assassin (whoever it is in the high-mass ρ) picks seat 2.
    """
    torch.manual_seed(0)
    bank = ValueBank()
    # Prior concentrated on (LS, LS, MERLIN, DS, ASSASSIN) — seats 0,1=LS,
    # seat 2=Merlin, seat 3=DS (Spy), seat 4=ASSASSIN.
    prior = np.full(NUM_ASSIGNMENTS, 1e-10, dtype=np.float64)
    target_rho = (Role.LS, Role.LS, Role.MERLIN, Role.DS, Role.ASSASSIN)
    target_idx = ASSIGNMENTS.index(target_rho)
    prior[target_idx] = 1.0
    prior /= prior.sum()

    solver = CFRSolver(_root_with_assassinate_reachable(), bank,
                       prior_belief=prior)
    avg = solver.solve(n_iters=30)

    # Find the AssassinateState's regret table for player 4 (the Assassin).
    # Look up player 4's Assassin info-set when Spy is at seat 3.
    assassin_info_set_key = (Role.ASSASSIN, 3)
    j = INFO_SETS_PER_PLAYER[4].index(assassin_info_set_key)
    # Find the AssassinateState's average strategy for player 4.
    assassinate_keys = [
        k for k, _ in avg.items()
        if isinstance(k[0][0], AssassinateState) and k[1] == 4
    ]
    assert len(assassinate_keys) == 1
    sigma = avg[assassinate_keys[0]]
    # σ[j, target=2] should dominate.
    pick = int(np.argmax(sigma[j]))
    assert pick == 2, f"Assassin picked seat {pick}, expected seat 2 (Merlin); σ[j]={sigma[j]}"
