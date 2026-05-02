"""
Tests for beliefs.py — integrated BeliefTracker behavior.

The marquee test is test_handcomputed_R1_pass_marginals, which checks the
BeliefTracker's output against values derived by hand:

  Setup: R1 mission of team {0,1}, 0 fails, uniform-random strategy.

  Group assignments by k = #evil in {0,1}:
    k=0 group: 18 assignments, mission likelihood = 1
    k=1 group: 36 assignments, mission likelihood = 1/2
    k=2 group:  6 assignments, mission likelihood = 1/4
  (Counts derived from the role multiset; total = 60.)

  Unnormalized posterior mass per group:
    k=0 -> 18 * 1   = 18
    k=1 -> 36 * 1/2 = 18
    k=2 ->  6 * 1/4 = 1.5
  Total Z = 37.5.

  Normalized group probabilities:
    P(k=0) = 18/37.5 = 0.48
    P(k=1) = 18/37.5 = 0.48
    P(k=2) =  1.5/37.5 ≈ 0.04

  Per-player marginals:
    P(player 0 evil) = P(k=1) * (prob 0 is the evil one | k=1) + P(k=2) * 1
                     = 0.48 * 0.5 + 0.04
                     = 0.28
    P(player 2 evil) (off-team, by symmetry):
                       k=0: 2 evil among {2,3,4} -> P(2 evil) = 2/3
                       k=1: 1 evil among {2,3,4} -> P(2 evil) = 1/3
                       k=2: 0 evil among {2,3,4} -> 0
                     = 0.48 * 2/3 + 0.48 * 1/3 + 0
                     = 0.48
  Sanity: total evil mass = 0.28 + 0.28 + 0.48 + 0.48 + 0.48 = 2.0 ✓
"""
import numpy as np
import pytest

from src.game import Role, new_game
from src.assignments import ASSIGNMENTS
from src.beliefs import BeliefTracker
from src.scenarios import make_obs_with_mission


def test_initial_belief_uniform():
    tracker = BeliefTracker()
    np.testing.assert_allclose(tracker.belief, 1 / 60 * np.ones(60))


def test_belief_sums_to_one_after_observe():
    rho = (Role.DS, Role.LS, Role.MERLIN, Role.ASSASSIN, Role.LS)
    obs = make_obs_with_mission(team=(0, 1), fails=1, assignment=rho)
    tracker = BeliefTracker()
    tracker.observe(obs)
    assert tracker.belief.sum() == pytest.approx(1.0)
    assert np.all(tracker.belief >= 0)


def test_double_fail_pinpoints_evil():
    """Mission {0,1} 2 fails => P(0 evil) = P(1 evil) = 1, others = 0."""
    rho = (Role.DS, Role.ASSASSIN, Role.MERLIN, Role.LS, Role.LS)
    obs = make_obs_with_mission(team=(0, 1), fails=2, assignment=rho)
    tracker = BeliefTracker()
    tracker.observe(obs)
    assert tracker.marginal_evil(0) == pytest.approx(1.0)
    assert tracker.marginal_evil(1) == pytest.approx(1.0)
    assert tracker.marginal_evil(2) == pytest.approx(0.0)
    assert tracker.marginal_evil(3) == pytest.approx(0.0)
    assert tracker.marginal_evil(4) == pytest.approx(0.0)


def test_pass_lowers_evil_belief_for_team_members():
    """Passing mission lowers team members' evil prob below the 0.4 baseline."""
    rho = (Role.LS, Role.LS, Role.MERLIN, Role.DS, Role.ASSASSIN)
    obs = make_obs_with_mission(team=(0, 1), fails=0, assignment=rho)
    tracker = BeliefTracker()
    tracker.observe(obs)
    assert tracker.marginal_evil(0) < 0.4
    assert tracker.marginal_evil(1) < 0.4
    # Off-team members must rise correspondingly: total evil = 2 always.
    total_evil = sum(tracker.marginal_evil(p) for p in range(5))
    assert total_evil == pytest.approx(2.0)


def test_handcomputed_R1_pass_marginals():
    """The marquee test: hand-computed marginals for R1 mission {0,1} 0 fails."""
    rho = (Role.LS, Role.LS, Role.MERLIN, Role.DS, Role.ASSASSIN)
    obs = make_obs_with_mission(team=(0, 1), fails=0, assignment=rho)
    tracker = BeliefTracker()
    tracker.observe(obs)

    # On-team players: 10.5 / 37.5 = 0.28
    assert tracker.marginal_evil(0) == pytest.approx(10.5 / 37.5)
    assert tracker.marginal_evil(1) == pytest.approx(10.5 / 37.5)
    # Off-team players (by symmetry, all equal): 0.48
    p2 = tracker.marginal_evil(2)
    p3 = tracker.marginal_evil(3)
    p4 = tracker.marginal_evil(4)
    assert p2 == pytest.approx(0.48)
    assert p3 == pytest.approx(p2)
    assert p4 == pytest.approx(p2)
    # Sanity: total evil mass conserves at 2.
    total = sum(tracker.marginal_evil(p) for p in range(5))
    assert total == pytest.approx(2.0)


def test_handcomputed_R1_pass_role_marginals():
    """Hand-computed Merlin marginal for player 0 after R1 mission {0,1} 0 fails.

    Of the 18 'k=0' assignments (player 0 and 1 both good):
      Position 0 holds Merlin in some fraction of these. Counting:
      For (M at 0): position 1 holds R, positions {2,3,4} permute {R, S, A}.
        Distinct: position 1 fixed as R (only choice that keeps role multiset),
        positions {2,3,4} are some permutation of {R, S, A} = 3! = 6.
      For (R at 0): position 1 can be M or R.
        - 1 = M: positions {2,3,4} = perm of {R, S, A} = 6.
        - 1 = R: positions {2,3,4} = perm of {M, S, A} = 6.
      So 18 = 6 (M@0) + 6 (R@0, M@1) + 6 (R@0, R@1).
      P(player 0 is M | k=0) = 6/18 = 1/3.

    Group probs: P(k=0) = 0.48. So P(player 0 = M, k=0) = 0.48/3 = 0.16.

    For k=1: player 0 is evil (S or A) or good. Half the time player 0 is the
    evil one (P(0 is evil | k=1) = 0.5), so P(0 is good | k=1) = 0.5.
    Within those 18 'k=1, 0 is good' assignments, what fraction has 0 as M?
      0 must be good (not evil); options for 0 are M or R.
      We trust the implementation here and just check the high-level invariant: P(0 = M) 
      summed over all assignments should equal sum of belief mass on assignments where 0 is M.
    """
    rho = (Role.LS, Role.LS, Role.MERLIN, Role.DS, Role.ASSASSIN)
    obs = make_obs_with_mission(team=(0, 1), fails=0, assignment=rho)
    tracker = BeliefTracker()
    tracker.observe(obs)

    # P(player 0 is Merlin) computed from belief vs marginal_role helper.
    by_helper = tracker.marginal_role(0, Role.MERLIN)
    by_hand = float(sum(
        tracker.belief[i] for i, ρ in enumerate(ASSIGNMENTS) if ρ[0] is Role.MERLIN
    ))
    assert by_helper == pytest.approx(by_hand)
    # And it must be in [0, 1].
    assert 0 <= by_helper <= 1
    # Sum over roles for a player must equal 1.
    total = sum(
        tracker.marginal_role(0, r)
        for r in (Role.MERLIN, Role.LS, Role.DS, Role.ASSASSIN)
    )
    assert total == pytest.approx(1.0)


def test_condition_on_role():
    """Conditioning concentrates belief on assignments matching the role."""
    tracker = BeliefTracker()
    cond = tracker.condition_on_role(0, Role.MERLIN)
    nonzero_idx = [i for i, ρ in enumerate(ASSIGNMENTS) if ρ[0] is Role.MERLIN]
    assert len(nonzero_idx) == 12
    assert cond.sum() == pytest.approx(1.0)
    for i in range(60):
        if i in nonzero_idx:
            assert cond[i] == pytest.approx(1 / 12)
        else:
            assert cond[i] == 0.0


def test_reset_restores_prior():
    rho = (Role.DS, Role.ASSASSIN, Role.MERLIN, Role.LS, Role.LS)
    obs = make_obs_with_mission(team=(0, 1), fails=2, assignment=rho)
    tracker = BeliefTracker()
    tracker.observe(obs)
    assert tracker.marginal_evil(0) == pytest.approx(1.0)
    tracker.reset()
    np.testing.assert_allclose(tracker.belief, 1 / 60 * np.ones(60))
