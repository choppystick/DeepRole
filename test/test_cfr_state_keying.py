"""Tests for the state-keyed CFR DAG: structure, topology, and per-iteration semantics.

The state-keyed refactor replaces path-keyed regret tables (one per distinct
public history) with state-keyed tables (one per (public_state, mask)). Multiple
paths reaching the same state share the table; one CFR iteration walks each
state once via forward + backward passes.

These tests verify:
  - DAG construction discovers the expected reachable state set.
  - Topological order: every state appears AFTER all its parents.
  - State count for the smallest demo subgame matches a hand-computed bound.
  - Vote-rejection edges collapse multiple paths onto one successor state.
  - Mission outcomes that produce different masks produce distinct state_keys.
  - Persistent regret tables across iterations are state-keyed (not path-keyed).
"""
import numpy as np
import pytest
import torch

from src.assignments import NUM_ASSIGNMENTS
from src.cfr import (
    CFRSolver,
    MissionState,
    ProposeState,
    TerminalState,
    ValueNetLeafState,
    VoteState,
    _ALL_TRUE_MASK,
    _make_state_key,
)
from src.game import MAX_PROPOSALS, NUM_PLAYERS
from src.value_bank import ValueBank


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------

def _last_attempt_root() -> ProposeState:
    """Smallest demo subgame: rejected_count=4, only one more attempt before forfeit."""
    return ProposeState(round_idx=0, proposer=0, successes=0, failures=0,
                        rejected_count=MAX_PROPOSALS - 1)


def _full_round_root() -> ProposeState:
    """Full round-0 subgame from rejected_count=0."""
    return ProposeState(round_idx=0, proposer=0, successes=0, failures=0,
                        rejected_count=0)


def test_root_key_is_state_plus_all_true_mask():
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    expected = _make_state_key(_last_attempt_root(), _ALL_TRUE_MASK)
    assert solver._root_key == expected


def test_topological_order_parents_before_children():
    """Every state appears in topo order strictly after each of its parents."""
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    position = {k: idx for idx, k in enumerate(solver._topo_order)}
    for parent_key, edges in solver._edges.items():
        for _, child_key in edges:
            assert position[parent_key] < position[child_key], (
                f"parent {parent_key[0]} at pos {position[parent_key]} not before "
                f"child {child_key[0]} at pos {position[child_key]}"
            )


def test_topo_order_includes_every_state_exactly_once():
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    assert len(solver._topo_order) == len(solver._states)
    assert set(solver._topo_order) == set(solver._states.keys())


def test_state_count_last_attempt_subgame():
    """For rejected_count=4 (one shot only), the DAG decomposes as:
      1 ProposeState + 10 VoteStates (one per team) + 10 MissionStates,
      21 ValueNetLeafState entries, and 1 TerminalState.

    Leaf accounting (10 missions × 3 outcomes each = 30 mission-edge endpoints):
      - "0 fails" outcomes: 10 edges, all with mask unchanged → collapse onto
        a single leaf (round=1, proposer=1, s=1, f=0, mask=all_True).
      - "1 fail" outcomes: 10 distinct masks (one per team) → 10 distinct leaves.
      - "2 fails" outcomes: 10 distinct masks (more constrained than 1-fail) → 10.
    Total: 1 + 10 + 10 = 21 leaves.
    """
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    states = list(solver._states.values())
    n_propose = sum(1 for (s, _) in states if isinstance(s, ProposeState))
    n_vote = sum(1 for (s, _) in states if isinstance(s, VoteState))
    n_mission = sum(1 for (s, _) in states if isinstance(s, MissionState))
    n_leaf = sum(1 for (s, _) in states if isinstance(s, ValueNetLeafState))
    n_term = sum(1 for (s, _) in states if isinstance(s, TerminalState))
    assert n_propose == 1
    assert n_vote == 10
    assert n_mission == 10
    assert n_leaf == 21
    assert n_term == 1


def test_rejection_edges_collapse_onto_one_terminal():
    """All 16 rejecting vote profiles from the last-attempt root land on the
    same TerminalState(spy) leaf."""
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    terminals = [k for k, (s, _) in solver._states.items() if isinstance(s, TerminalState)]
    assert len(terminals) == 1
    term_key = terminals[0]
    # Count incoming edges into the terminal.
    incoming = sum(
        1
        for parent_edges in solver._edges.values()
        for (_, child_key) in parent_edges
        if child_key == term_key
    )
    # 10 vote nodes × 16 rejecting profiles each = 160 edges.
    assert incoming == 160


def test_approve_edges_collapse_per_voted_team():
    """All 16 approving vote profiles from one VoteState land on one MissionState."""
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    # Pick any vote state.
    vote_key = next(
        k for k, (s, _) in solver._states.items() if isinstance(s, VoteState)
    )
    edges = solver._edges[vote_key]
    mission_children = [child_key for _, child_key in edges
                        if isinstance(solver._states[child_key][0], MissionState)]
    assert len(mission_children) == 16   # 16 approving profiles per vote node
    assert len(set(mission_children)) == 1   # ... all to the same MissionState


def test_mission_edges_to_distinct_leaf_keys():
    """For a single mission, the (team, fails) outcomes that collapse to the
    same public successor state still produce distinct keys when masks differ."""
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    # All round-0 mission states have team-of-2; observations are 0, 1, 2 fails.
    a_mission_key = next(
        k for k, (s, _) in solver._states.items() if isinstance(s, MissionState)
    )
    edges = solver._edges[a_mission_key]
    assert len(edges) == 3
    children = {child_key for _, child_key in edges}
    assert len(children) == 3   # three distinct (state, mask) keys


def test_full_round_dag_is_finite_and_acyclic():
    """Full round-0 subgame builds without cycles; topological sort succeeds.

    Rejection rotates the proposer deterministically: rejected=0 ⇒ proposer=p0,
    rejected=1 ⇒ proposer=(p0+1)%5, etc. So there is exactly one ProposeState
    per rejected_count, not one per (proposer, rejected) pair.
    """
    bank = ValueBank()
    solver = CFRSolver(_full_round_root(), bank)
    n_propose = sum(
        1 for (s, _) in solver._states.values() if isinstance(s, ProposeState)
    )
    assert n_propose == 5   # one per rejected_count ∈ {0,1,2,3,4}
    n_vote = sum(
        1 for (s, _) in solver._states.values() if isinstance(s, VoteState)
    )
    assert n_vote == 50   # 5 propose × 10 teams
    # Topo sort length matches state count.
    assert len(solver._topo_order) == len(solver._states)


def test_dag_introspection_helpers():
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    assert solver.num_states() == len(solver._states)
    assert solver.num_edges() == sum(len(es) for es in solver._edges.values())


# ---------------------------------------------------------------------------
# State-keyed regret tables
# ---------------------------------------------------------------------------

def test_regret_keys_are_state_keyed_after_iteration():
    """After solving, regret tables are keyed by ((state, mask_bytes), player) —
    not by paths."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    solver.solve(n_iters=2)
    # Every regret-table key has the form ((public_state, mask_bytes), int).
    for key in solver.regrets:
        assert isinstance(key, tuple) and len(key) == 2
        state_key, player = key
        assert isinstance(state_key, tuple) and len(state_key) == 2
        public_state, mask_bytes = state_key
        assert isinstance(mask_bytes, bytes)
        assert isinstance(player, int)


def test_regret_table_count_matches_state_keyed_bound():
    """State-keyed regret tables: one per (state, player) where state is
    a non-leaf decision node. For the full round-0 subgame:
      - 5 propose states × 1 mover (proposer) = 5
      - 50 vote states × 5 movers = 250
      - 50 mission states × 2 movers (team-of-2) = 100
    Total: 355 tables. Path-keyed would have multiplied this by the number of
    distinct paths to each state.
    """
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_full_round_root(), bank)
    solver.solve(n_iters=1)
    expected = 5 * 1 + 50 * 5 + 50 * 2   # = 355
    assert len(solver.regrets) == expected


def test_state_keyed_iteration_strategies_valid():
    """End-to-end: state-keyed CFR produces valid strategy distributions."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    avg = solver.solve(n_iters=3)
    for key, sigma in avg.items():
        assert np.all(sigma >= 0.0), key
        np.testing.assert_allclose(
            sigma.sum(axis=-1), np.ones(sigma.shape[0]), atol=1e-9
        )


def test_state_keyed_regrets_remain_non_negative():
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    solver.solve(n_iters=3)
    for key, r in solver.regrets.items():
        assert np.all(r >= 0.0), key


def test_state_keyed_solve_is_deterministic_with_seed():
    """Two solves with the same seeded value bank produce identical strategies."""
    torch.manual_seed(0)
    bank_a = ValueBank()
    solver_a = CFRSolver(_last_attempt_root(), bank_a)
    avg_a = solver_a.solve(n_iters=3)

    torch.manual_seed(0)
    bank_b = ValueBank()
    solver_b = CFRSolver(_last_attempt_root(), bank_b)
    avg_b = solver_b.solve(n_iters=3)

    assert set(avg_a.keys()) == set(avg_b.keys())
    for k in avg_a:
        np.testing.assert_allclose(avg_a[k], avg_b[k], atol=1e-12)


def test_dag_built_once_not_per_iteration():
    """Solving multiple iterations shouldn't grow the DAG; states/edges are fixed."""
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    n_states_before = solver.num_states()
    n_edges_before = solver.num_edges()
    solver.solve(n_iters=5)
    assert solver.num_states() == n_states_before
    assert solver.num_edges() == n_edges_before


def test_root_strategy_helper_matches_indexed_lookup():
    torch.manual_seed(0)
    bank = ValueBank()
    solver = CFRSolver(_last_attempt_root(), bank)
    solver.solve(n_iters=2)
    sigma_helper = solver.root_strategy(player=0)
    sigma_index = solver.average_strategies()[(solver._root_key, 0)]
    np.testing.assert_array_equal(sigma_helper, sigma_index)
