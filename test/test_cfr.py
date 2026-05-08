"""Tests for cfr.py — public-state machinery, deduceActions, end-to-end CFR."""
import numpy as np
import pytest
import torch

from src.assignments import ASSIGNMENTS, INFO_SETS_PER_PLAYER, NUM_ASSIGNMENTS, evil_indices
from src.cfr import (
    CFRSolver,
    MissionState,
    ProposeState,
    TerminalState,
    VOTE_PROFILES,
    ValueNetLeafState,
    VoteState,
    _AMBIGUOUS_SENTINEL,
    _evil_action_dist,
    child_state,
    deduce_actions,
    legal_observations,
    majority_approves,
    moving_players,
    num_actions,
    teams_for_size,
    update_consistency_mask,
)
from src.game import MAX_PROPOSALS, NUM_PLAYERS, Role, TEAM_SIZES
from src.value_bank import ValueBank


# ---------------------------------------------------------------------------
# State plumbing
# ---------------------------------------------------------------------------

def test_propose_moving_players_just_proposer():
    s = ProposeState(round_idx=0, proposer=2, successes=0, failures=0, rejected_count=0)
    assert moving_players(s) == (2,)


def test_vote_moving_players_all_five():
    s = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                  rejected_count=0, team=(0, 1))
    assert moving_players(s) == tuple(range(5))


def test_mission_moving_players_team_only():
    s = MissionState(round_idx=0, proposer=0, successes=0, failures=0, team=(1, 3))
    assert moving_players(s) == (1, 3)


def test_propose_observations_are_team_combinations():
    s = ProposeState(round_idx=0, proposer=0, successes=0, failures=0, rejected_count=0)
    obs = legal_observations(s)
    assert obs == teams_for_size(TEAM_SIZES[0])
    assert len(obs) == 10  # C(5, 2)


def test_vote_observations_full_32_profiles():
    s = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                  rejected_count=0, team=(0, 1))
    obs = legal_observations(s)
    assert obs == VOTE_PROFILES
    assert len(obs) == 32
    # Every profile is a 5-tuple of booleans, all distinct.
    assert len(set(obs)) == 32
    for p in obs:
        assert len(p) == NUM_PLAYERS
        assert all(isinstance(v, bool) for v in p)


def test_mission_observations_zero_to_team_size():
    s = MissionState(round_idx=0, proposer=0, successes=0, failures=0, team=(0, 1, 2))
    assert legal_observations(s) == (0, 1, 2, 3)


def test_propose_to_vote_transition():
    s = ProposeState(round_idx=0, proposer=2, successes=1, failures=1, rejected_count=0)
    child = child_state(s, (0, 1))
    assert isinstance(child, VoteState)
    assert child.team == (0, 1)
    assert child.proposer == 2
    assert child.rejected_count == 0


def test_vote_approve_profile_to_mission():
    s = VoteState(round_idx=0, proposer=2, successes=0, failures=0,
                  rejected_count=0, team=(0, 1))
    approving = (True, True, True, False, False)  # 3-2 majority
    child = child_state(s, approving)
    assert isinstance(child, MissionState)
    assert child.team == (0, 1)


def test_vote_reject_profile_advances_proposer():
    s = VoteState(round_idx=0, proposer=2, successes=0, failures=0,
                  rejected_count=0, team=(0, 1))
    rejecting = (True, True, False, False, False)  # 2-3 minority
    child = child_state(s, rejecting)
    assert isinstance(child, ProposeState)
    assert child.proposer == 3
    assert child.rejected_count == 1


def test_vote_fifth_reject_terminates():
    s = VoteState(round_idx=0, proposer=2, successes=0, failures=0,
                  rejected_count=MAX_PROPOSALS - 1, team=(0, 1))
    rejecting = (False, False, False, False, False)
    child = child_state(s, rejecting)
    assert isinstance(child, TerminalState)
    assert child.winner == "spy"


def test_vote_majority_approves_helper():
    assert majority_approves((True, True, True, False, False)) is True
    assert majority_approves((True, True, False, False, False)) is False
    assert majority_approves((True, True, True, True, True)) is True
    assert majority_approves((False, False, False, False, False)) is False
    assert majority_approves((True, True, False, True, False)) is True
    assert majority_approves((True, False, True, False, False)) is False


def test_mission_pass_to_value_net_leaf():
    s = MissionState(round_idx=0, proposer=0, successes=0, failures=0, team=(0, 1))
    child = child_state(s, 0)
    assert isinstance(child, ValueNetLeafState)
    assert child.successes == 1
    assert child.failures == 0
    assert child.proposer == 1
    assert child.round_idx == 1


def test_mission_three_fails_terminal_spy():
    s = MissionState(round_idx=2, proposer=0, successes=1, failures=2, team=(0, 1, 2))
    child = child_state(s, 1)
    assert isinstance(child, TerminalState)
    assert child.winner == "spy"


def test_mission_third_success_routes_to_assassinate():
    """Third successful mission no longer routes to a "pending" terminal —
    it routes to AssassinateState, where the Assassin actually picks."""
    from src.cfr import AssassinateState
    s = MissionState(round_idx=2, proposer=0, successes=2, failures=0, team=(0, 1, 2))
    child = child_state(s, 0)
    assert isinstance(child, AssassinateState)


# ---------------------------------------------------------------------------
# deduceActions
# ---------------------------------------------------------------------------

def test_deduce_propose_one_hot_on_team_index():
    s = ProposeState(round_idx=0, proposer=2, successes=0, failures=0, rejected_count=0)
    teams = teams_for_size(TEAM_SIZES[0])
    target = teams[3]  # arbitrary
    actions = deduce_actions(s, target)
    assert set(actions.keys()) == {2}
    a = actions[2]
    assert a.shape == (15, len(teams))
    # One-hot at the chosen team's index, identical across info sets.
    expected = np.zeros((15, len(teams)))
    expected[:, 3] = 1.0
    np.testing.assert_array_equal(a, expected)


def test_deduce_vote_per_player_action_from_profile():
    """Each player's action distribution is one-hot at their bit in the profile."""
    s = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                  rejected_count=0, team=(0, 1))
    profile = (True, False, True, False, True)  # 0,2,4 yes; 1,3 no
    actions = deduce_actions(s, profile)
    assert set(actions.keys()) == set(range(5))
    for i in range(5):
        expected_action = 0 if profile[i] else 1
        np.testing.assert_array_equal(actions[i][:, expected_action], np.ones(15))
        np.testing.assert_array_equal(actions[i][:, 1 - expected_action], np.zeros(15))


def test_deduce_vote_unanimous_profiles():
    s = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                  rejected_count=0, team=(0, 1))
    all_yes = (True,) * 5
    actions = deduce_actions(s, all_yes)
    for i in range(5):
        np.testing.assert_array_equal(actions[i][:, 0], np.ones(15))
        np.testing.assert_array_equal(actions[i][:, 1], np.zeros(15))
    all_no = (False,) * 5
    actions = deduce_actions(s, all_no)
    for i in range(5):
        np.testing.assert_array_equal(actions[i][:, 1], np.ones(15))
        np.testing.assert_array_equal(actions[i][:, 0], np.zeros(15))


def test_deduce_mission_zero_fails_all_succeed():
    """f=0: every team member, in every info-set, is determined to have played S."""
    s = MissionState(round_idx=0, proposer=0, successes=0, failures=0, team=(0, 1))
    actions = deduce_actions(s, 0)
    for player_idx in (0, 1):
        a = actions[player_idx]
        # Action 0 = succeed.
        np.testing.assert_array_equal(a[:, 0], np.ones(15))
        np.testing.assert_array_equal(a[:, 1], np.zeros(15))


def test_deduce_mission_two_fails_evil_played_F():
    """f=2 on team of 2 ⇒ both played F. Good info-sets are unaffected; for the
    Spy info-set with Assassin on team, ⃗a[F]=1; for Spy info-set with Assassin
    OFF the team, k_I=1 < f=2 ⇒ ⃗a is zero (impossible info-set)."""
    team = (0, 1)
    s = MissionState(round_idx=0, proposer=0, successes=0, failures=0, team=team)
    actions = deduce_actions(s, 2)
    for player_idx in team:
        a = actions[player_idx]
        for j, key in enumerate(INFO_SETS_PER_PLAYER[player_idx]):
            role = key[0]
            if role is Role.LS or role is Role.MERLIN:
                # Good — forced S, even though this info-set is inconsistent
                # with f=2 (consistency mask zeros it later).
                assert a[j, 0] == 1.0
                assert a[j, 1] == 0.0
            elif role is Role.DS:
                assassin_seat = key[1]
                k_I = 1 + (assassin_seat in team)
                if k_I == 2:
                    assert a[j, 1] == 1.0  # F
                    assert a[j, 0] == 0.0
                else:
                    # k_I=1 < 2 ⇒ inconsistent info-set, distribution is zero.
                    assert a[j, 0] == 0.0
                    assert a[j, 1] == 0.0
            elif role is Role.ASSASSIN:
                spy_seat = key[1]
                k_I = 1 + (spy_seat in team)
                if k_I == 2:
                    assert a[j, 1] == 1.0
                    assert a[j, 0] == 0.0
                else:
                    assert a[j, 0] == 0.0
                    assert a[j, 1] == 0.0


def test_deduce_mission_one_fail_emits_sentinel_when_ambiguous():
    """f=1, k_I=2 ⇒ deduce_actions emits the ambiguous sentinel for that row.
    The CFR loop replaces it with the conditional dist using the current strategy."""
    team = (0, 1)
    s = MissionState(round_idx=0, proposer=0, successes=0, failures=0, team=team)
    actions = deduce_actions(s, 1)
    a = actions[0]
    found_ambiguous = False
    for j, key in enumerate(INFO_SETS_PER_PLAYER[0]):
        role = key[0]
        if role is Role.DS:
            assassin_seat = key[1]
            k_I = 1 + (assassin_seat in team)
            if k_I == 2:
                # Sentinel: both columns set to _AMBIGUOUS_SENTINEL.
                assert a[j, 0] == _AMBIGUOUS_SENTINEL
                assert a[j, 1] == _AMBIGUOUS_SENTINEL
                found_ambiguous = True
            else:
                # k_I=1, f=1 ⇒ deducible: i played F.
                assert a[j, 1] == 1.0
                assert a[j, 0] == 0.0
    assert found_ambiguous


def test_evil_action_dist_edge_cases():
    assert _evil_action_dist(0, 0) == (0.0, 0.0)         # impossible info-set
    assert _evil_action_dist(2, 1) == (0.0, 0.0)         # k < f
    assert _evil_action_dist(0, 2) == (1.0, 0.0)         # all played S
    assert _evil_action_dist(2, 2) == (0.0, 1.0)         # all played F
    assert _evil_action_dist(1, 2) is None               # ambiguous (sentinel)
    assert _evil_action_dist(1, 1) == (0.0, 1.0)         # k=1, f=1 ⇒ forced F
    assert _evil_action_dist(0, 1) == (1.0, 0.0)         # k=1, f=0 ⇒ forced S


# ---------------------------------------------------------------------------
# Consistency mask
# ---------------------------------------------------------------------------

def test_consistency_mask_zero_fails_unchanged():
    mask = np.ones(NUM_ASSIGNMENTS, dtype=bool)
    out = update_consistency_mask(mask, (0, 1), 0)
    np.testing.assert_array_equal(out, mask)


def test_consistency_mask_two_fails_team_of_two_pinpoints():
    mask = np.ones(NUM_ASSIGNMENTS, dtype=bool)
    out = update_consistency_mask(mask, (0, 1), 2)
    # Survivors: ρ where both players 0 and 1 are evil.
    for a, rho in enumerate(ASSIGNMENTS):
        ev = evil_indices(rho)
        survives = (0 in ev) and (1 in ev)
        assert out[a] == survives


# ---------------------------------------------------------------------------
# End-to-end CFR sanity checks
# ---------------------------------------------------------------------------

def _root_propose_state(rejected_count=MAX_PROPOSALS - 1):
    """Smallest non-trivial subgame: only one more proposal allowed before forfeit."""
    return ProposeState(
        round_idx=0,
        proposer=0,
        successes=0,
        failures=0,
        rejected_count=rejected_count,
    )


def test_cfr_runs_one_iteration_returns_strategy():
    """Smoke test: 1 iteration of CFR on tiny subgame produces strategies of valid shape."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_root_propose_state(), bank)
    avg = solver.solve(n_iters=1, averaging_delay=0)
    # The root proposer's strategy must exist; root_strategy() abstracts the key format.
    root_sigma = solver.root_strategy(player=0)
    assert root_sigma.shape == (15, 10)  # 15 info-sets × 10 teams
    np.testing.assert_allclose(root_sigma.sum(axis=-1), np.ones(15), atol=1e-9)


def test_cfr_strategies_are_distributions():
    """All average strategies sum to 1 along the action axis and are non-negative."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_root_propose_state(), bank)
    avg = solver.solve(n_iters=3, averaging_delay=0)
    for key, sigma in avg.items():
        assert np.all(sigma >= 0.0), key
        np.testing.assert_allclose(sigma.sum(axis=-1), np.ones(sigma.shape[0]), atol=1e-9)


def test_cfr_regrets_non_negative():
    """CFR+ floors regrets at zero after every update."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_root_propose_state(), bank)
    solver.solve(n_iters=3, averaging_delay=0)
    for key, r in solver.regrets.items():
        assert np.all(r >= 0.0), key


def test_cfr_root_strategy_helper():
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_root_propose_state(), bank)
    solver.solve(n_iters=2)
    sigma = solver.root_strategy(player=0)
    assert sigma.shape == (15, 10)
    np.testing.assert_allclose(sigma.sum(axis=-1), np.ones(15), atol=1e-9)


def test_cfr_calc_terminal_belief_factorized():
    """At the root with reach=1 and all-true mask, calc_terminal_belief is the prior."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_root_propose_state(), bank)
    reach = np.ones((NUM_PLAYERS, 15), dtype=np.float64)
    mask = np.ones(NUM_ASSIGNMENTS, dtype=bool)
    bterm = solver._calc_terminal_belief(solver.prior_belief, reach, mask)
    np.testing.assert_allclose(bterm, solver.prior_belief)


def test_cfr_terminal_cfvs_sign_for_spy_winner():
    """At a TerminalState(spy) with reach=1, evil players get +1, good get −1
    when summed across all assignments (uniform prior). Specifically:
      Σ_j v[i, j] = Σ_a util[i, a] * b[a]
    With uniform b = 1/60 and util[i, a] = ±1, sum equals (#evil ρ - #good ρ)/60
    for player i — depends on per-position role distribution."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_root_propose_state(), bank)
    reach = np.ones((NUM_PLAYERS, 15), dtype=np.float64)
    mask = np.ones(NUM_ASSIGNMENTS, dtype=bool)
    term = TerminalState(winner="spy")
    v = solver._terminal_cfvs(term, solver.prior_belief, reach, mask)
    # Σ_j v[i, j] = (P(i evil) - P(i good)) under uniform prior over 60.
    # Per-position: 24 evil ρ (12 Spy + 12 Assassin), 36 good (12 Merlin + 24 LS).
    # Sum = (24 - 36)/60 = -0.2 for every player.
    for i in range(NUM_PLAYERS):
        assert v[i].sum() == pytest.approx(-0.2)


def test_cfr_neural_cfvs_runs_without_error():
    """Smoke test: NEURALCFVS on a fresh net returns a (5, 15) finite tensor."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_root_propose_state(), bank)
    reach = np.ones((NUM_PLAYERS, 15), dtype=np.float64)
    mask = np.ones(NUM_ASSIGNMENTS, dtype=bool)
    leaf = ValueNetLeafState(round_idx=1, proposer=1, successes=1, failures=0)
    v = solver._neural_cfvs(leaf, solver.prior_belief, reach, mask)
    assert v.shape == (NUM_PLAYERS, 15)
    assert np.all(np.isfinite(v))
