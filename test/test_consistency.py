"""Tests for consistency.py — deductive zeroing of impossible assignments."""
import numpy as np
import pytest

from src.game import Role, new_game
from src.assignments import ASSIGNMENTS, evil_indices
from src.consistency import consistency_mask
from src.scenarios import make_obs_with_mission


def test_initial_state_all_consistent():
    """No mission observed yet => all 60 assignments consistent."""
    g = new_game(assignment=ASSIGNMENTS[0])
    mask = consistency_mask(g.observation())
    assert mask.all()


def test_zero_fail_is_unconstraining():
    """A passing mission (0 fails) is consistent with any assignment, since
    0 evil on team is always fine and 1 or 2 evil COULD have chosen succeed."""
    rho = (Role.LS, Role.LS, Role.MERLIN, Role.DS, Role.ASSASSIN)
    obs = make_obs_with_mission(team=(0, 1), fails=0, assignment=rho)
    mask = consistency_mask(obs)
    assert mask.all()


def test_one_fail_rules_out_all_good_teams():
    """Mission {0,1} with 1 fail rules out any ρ with 0 evil in {0,1}."""
    rho = (Role.DS, Role.LS, Role.MERLIN, Role.ASSASSIN, Role.LS)
    obs = make_obs_with_mission(team=(0, 1), fails=1, assignment=rho)
    mask = consistency_mask(obs)
    for i, ρ in enumerate(ASSIGNMENTS):
        team_evil = len({0, 1} & evil_indices(ρ))
        assert mask[i] == (team_evil >= 1)


def test_double_fail_pinpoints_evil_pair():
    """Mission {0,1} with 2 fails => only assignments where both 0 and 1 are evil."""
    rho = (Role.DS, Role.ASSASSIN, Role.MERLIN, Role.LS, Role.LS)
    obs = make_obs_with_mission(team=(0, 1), fails=2, assignment=rho)
    mask = consistency_mask(obs)
    for i, ρ in enumerate(ASSIGNMENTS):
        both_evil = (0 in evil_indices(ρ)) and (1 in evil_indices(ρ))
        assert mask[i] == both_evil
    # Exactly 6 such assignments: {0,1} ∈ {(S,A), (A,S)} × 3 placements of M among {2,3,4}
    assert mask.sum() == 6


def test_triple_team_one_fail():
    """Round 2 mission {0,1,2} with 1 fail rules out ρ with 0 evil in {0,1,2}."""
    rho = (Role.DS, Role.LS, Role.MERLIN, Role.ASSASSIN, Role.LS)
    obs = make_obs_with_mission(team=(0, 1, 2), fails=1, assignment=rho, round_idx=1)
    mask = consistency_mask(obs)
    for i, ρ in enumerate(ASSIGNMENTS):
        team_evil = len({0, 1, 2} & evil_indices(ρ))
        assert mask[i] == (team_evil >= 1)
