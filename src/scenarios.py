"""
Shared test scenario builders. NOT a test file — pytest only collects test_*.py.
"""
from __future__ import annotations

from typing import Tuple

from src.game import GameState, Role, TEAM_SIZES, new_game


def _play_one_mission(
    g: GameState,
    team: Tuple[int, ...],
    fails: int,
) -> None:
    """Propose `team`, have everyone approve, and resolve with `fails` fails."""
    g.propose(tuple(team))
    g.vote((True, True, True, True, True))
    evil_on_team = [p for p in team if g.assignment[p].is_evil()]
    assert len(evil_on_team) >= fails, (
        f"team {team} under {g.assignment} has only {len(evil_on_team)} evil; "
        f"can't produce {fails} fails"
    )
    plays = {}
    for p in team:
        if g.assignment[p].is_good():
            plays[p] = True
        else:
            plays[p] = False if evil_on_team.index(p) < fails else True
    g.play_mission(plays)


def make_obs_with_mission(
    team: Tuple[int, ...],
    fails: int,
    assignment: Tuple[Role, ...],
    round_idx: int = 0,
) -> dict:
    """Build an observation that runs a mission with `team` and `fails` on
    round `round_idx`. Earlier rounds are padded with dummy all-good passing
    missions so the game reaches the target round legally.

    Caller's responsibility: ensure `len(team) == TEAM_SIZES[round_idx]` and
    that the team contains enough evil players to produce `fails` fails.
    """
    assert len(team) == TEAM_SIZES[round_idx], (
        f"round {round_idx} requires team size {TEAM_SIZES[round_idx]}, "
        f"got {len(team)}"
    )
    g = new_game(assignment=assignment)

    # Pad with passing missions for earlier rounds, using only good players.
    good_players = [p for p in range(5) if g.assignment[p].is_good()]
    for r in range(round_idx):
        size = TEAM_SIZES[r]
        dummy_team = tuple(good_players[:size])
        _play_one_mission(g, dummy_team, fails=0)

    _play_one_mission(g, team, fails)
    return g.observation()
