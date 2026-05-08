"""Regression tests for paper-faithful 32-vote-profile branching at VoteState."""
import numpy as np
import pytest

from src.cfr import (
    MissionState,
    ProposeState,
    TerminalState,
    VOTE_PROFILES,
    VoteState,
    child_state,
    deduce_actions,
    legal_observations,
    majority_approves,
)
from src.game import MAX_PROPOSALS, NUM_PLAYERS


def test_vote_node_has_32_observations():
    vs = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                   rejected_count=0, team=(0, 1))
    obs = legal_observations(vs)
    assert len(obs) == 32
    assert len(set(obs)) == 32


def test_vote_profiles_canonical_order():
    """Profiles enumerate by integer-bit order (LSB = player 0)."""
    assert VOTE_PROFILES[0] == (False, False, False, False, False)
    assert VOTE_PROFILES[1] == (True, False, False, False, False)   # bit 0 set
    assert VOTE_PROFILES[31] == (True, True, True, True, True)


def test_vote_node_majority_dispatch():
    """Approving profiles → MissionState; rejecting → ProposeState (or terminal)."""
    vs = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                   rejected_count=0, team=(0, 1))
    approving = [p for p in VOTE_PROFILES if majority_approves(p)]
    rejecting = [p for p in VOTE_PROFILES if not majority_approves(p)]
    # 5 players, strict majority is ≥ 3 yes: C(5,3) + C(5,4) + C(5,5) = 16 approving.
    assert len(approving) == 16
    assert len(rejecting) == 16
    for p in approving:
        c = child_state(vs, p)
        assert isinstance(c, MissionState)
    for p in rejecting:
        c = child_state(vs, p)
        assert isinstance(c, ProposeState)
        assert c.rejected_count == 1


def test_vote_node_rejection_collapses_to_one_propose_state():
    """All 16 rejecting profiles map to the SAME ProposeState (state collapses
    on rejection — only the count matters), but the path tokens differ so CFR
    keys regret tables distinctly."""
    vs = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                   rejected_count=0, team=(0, 1))
    rejecting = [p for p in VOTE_PROFILES if not majority_approves(p)]
    children = {child_state(vs, p) for p in rejecting}
    assert len(children) == 1


def test_vote_node_fifth_rejection_terminal():
    """When already at MAX_PROPOSALS - 1 rejections, any rejecting profile ends in spy win."""
    vs = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                   rejected_count=MAX_PROPOSALS - 1, team=(0, 1))
    rejecting = [p for p in VOTE_PROFILES if not majority_approves(p)]
    for p in rejecting:
        c = child_state(vs, p)
        assert isinstance(c, TerminalState)
        assert c.winner == "spy"


def test_deduce_actions_per_player_one_hot():
    """For any profile, player i's action is one-hot at index 0 (yes) or 1 (no)."""
    vs = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                   rejected_count=0, team=(0, 1))
    profile = (True, True, False, True, False)
    actions = deduce_actions(vs, profile)
    assert set(actions.keys()) == set(range(NUM_PLAYERS))
    for i in range(NUM_PLAYERS):
        a = actions[i]
        assert a.shape == (15, 2)
        col = 0 if profile[i] else 1
        np.testing.assert_array_equal(a[:, col], np.ones(15))
        np.testing.assert_array_equal(a[:, 1 - col], np.zeros(15))


def test_deduce_actions_two_distinct_profiles_differ_per_player():
    """Two profiles that differ at player k must produce different actions for k
    (and identical actions for the others)."""
    vs = VoteState(round_idx=0, proposer=0, successes=0, failures=0,
                   rejected_count=0, team=(0, 1))
    p1 = (True, True, True, False, False)
    p2 = (True, True, False, False, False)   # differs at player 2
    a1 = deduce_actions(vs, p1)
    a2 = deduce_actions(vs, p2)
    for i in range(NUM_PLAYERS):
        if i == 2:
            assert not np.array_equal(a1[i], a2[i])
        else:
            np.testing.assert_array_equal(a1[i], a2[i])
