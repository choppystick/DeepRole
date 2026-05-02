"""Tests for likelihood.py — uniform-random mission likelihoods."""
import numpy as np
import pytest

from src.game import Role, new_game
from src.assignments import ASSIGNMENTS, evil_indices
from src.likelihood import uniform_likelihood
from src.scenarios import make_obs_with_mission


def test_initial_state_likelihood_one():
    """No mission observed => no constraint => likelihood 1 for all assignments."""
    g = new_game(assignment=ASSIGNMENTS[0])
    lik = uniform_likelihood(g.observation())
    assert np.all(lik == 1.0)


def test_zero_fail_likelihood_by_evil_count():
    """Mission {0,1} 0 fails: likelihood = 1/2^k for k evil on team."""
    rho = (Role.LS, Role.LS, Role.MERLIN, Role.DS, Role.ASSASSIN)
    obs = make_obs_with_mission(team=(0, 1), fails=0, assignment=rho)
    lik = uniform_likelihood(obs)
    for i, ρ in enumerate(ASSIGNMENTS):
        k = len({0, 1} & evil_indices(ρ))
        expected = 1.0 / (2 ** k)
        assert lik[i] == pytest.approx(expected)


def test_one_fail_likelihood_by_evil_count():
    """Mission {0,1} 1 fail: lik = C(k,1)/2^k = k/2^k.
       k=0: 0; k=1: 1/2; k=2: 2/4 = 1/2."""
    rho = (Role.DS, Role.LS, Role.MERLIN, Role.ASSASSIN, Role.LS)
    obs = make_obs_with_mission(team=(0, 1), fails=1, assignment=rho)
    lik = uniform_likelihood(obs)
    for i, ρ in enumerate(ASSIGNMENTS):
        k = len({0, 1} & evil_indices(ρ))
        expected = {0: 0.0, 1: 0.5, 2: 0.5}[k]
        assert lik[i] == pytest.approx(expected)


def test_two_fail_likelihood():
    """Mission {0,1} 2 fails: lik = 1/4 if k=2 else 0."""
    rho = (Role.DS, Role.ASSASSIN, Role.MERLIN, Role.LS, Role.LS)
    obs = make_obs_with_mission(team=(0, 1), fails=2, assignment=rho)
    lik = uniform_likelihood(obs)
    for i, ρ in enumerate(ASSIGNMENTS):
        k = len({0, 1} & evil_indices(ρ))
        expected = 0.25 if k == 2 else 0.0
        assert lik[i] == pytest.approx(expected)


def test_likelihood_factorizes_across_missions():
    """Likelihood of two missions should multiply."""
    # Build a 2-round game manually: R1 mission {0,1} passes; R2 mission {0,1,2} fails 1.
    rho = (Role.DS, Role.LS, Role.MERLIN, Role.ASSASSIN, Role.LS)
    g = new_game(assignment=rho)
    g.propose((0, 1))
    g.vote((True, True, True, True, True))
    g.play_mission({0: False, 1: True})  # spy at 0 succeeds (passing mission)
    # Wait — passing mission with spy on team requires spy chose succeed.
    # Let me redo: this builds 0 fails mission.
    # Actually the current setup gives 1 fail (spy chose False). Reset.
    g = new_game(assignment=rho)
    g.propose((1, 2))
    g.vote((True, True, True, True, True))
    g.play_mission({1: True, 2: True})  # both good, 0 fails
    # Now R2: team {0, 1, 2} (spy + good + good) — spy fails
    g.propose((0, 1, 2))
    g.vote((True, True, True, True, True))
    g.play_mission({0: False, 1: True, 2: True})  # 1 fail
    obs = g.observation()
    lik = uniform_likelihood(obs)

    # Compute expected: lik(ρ) = (1/2^k1) * (C(k2,1)/2^k2) where
    #   k1 = |{1,2} ∩ evil(ρ)|, k2 = |{0,1,2} ∩ evil(ρ)|, with f1=0, f2=1
    from math import comb
    for i, ρ in enumerate(ASSIGNMENTS):
        k1 = len({1, 2} & evil_indices(ρ))
        k2 = len({0, 1, 2} & evil_indices(ρ))
        expected = (comb(k1, 0) / 2 ** k1) * (comb(k2, 1) / 2 ** k2)
        assert lik[i] == pytest.approx(expected)
