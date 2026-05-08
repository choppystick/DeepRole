"""
Depth-limited vector-form CFR with deductive logic for DeepRole.

Implements Algorithms 1 and 2 of Serrino et al. (2019):
  SOLVESITUATION → MODIFIEDCFR+ → {TERMINALCFVS, NEURALCFVS, CALCTERMINALBELIEF}.

Design choices:
  Single-subgame solver. Builds and runs CFR over the subgame rooted
    at one ProposeState; the value bank evaluates leaves at the next
    mission outcome.
  Strategy-aware likelihood lives in strategy_likelihood.py; the CFR
    loop uses it via CALCTERMINALBELIEF.
  On-demand recursion. No materialized PublicNode tree. Regret /
    cumulative-strategy tables are keyed by (path, player), where
    path is a tuple of (state_kind, observation) tokens.
  Depth limit "until next mission outcome". Subgame leaves are
    either ValueNetLeafState (next round's proposal — handled by the
    value bank) or TerminalState (5-rejection mordred win, or 3-fail
    mordred win, or pending-assassinate placeholder).
  Vote observations are the full 5-tuple of per-player yes/no votes — all
    32 profiles, matching the reference implementation. Per-player reach
    update uses σ_i[my_vote | I] where my_vote is read off the profile.
  Mission deduction is paper-faithful (responsibility-weighted joint
    factorization). For deterministic info-sets the per-info-set action is
    forced. For the ambiguous (k_I=2, f=1) case, deduce_actions emits a
    sentinel and the CFR loop computes:
      - a per-player CONDITIONAL action distribution given the observed
        outcome (used for m[i] and the σ·a sum), and
      - a responsibility-weighted REACH factor whose product across the
        two evil partners reconstructs outcome_prob exactly. See
        _responsibility_factor.

Action enumeration conventions:
  ProposeState: action_idx = combinations(range(5), team_size).index(team)
  VoteState:    action 0 = approve, action 1 = reject. Observation is a
                VoteProfile = (bool, bool, bool, bool, bool); all 32 profiles
                are distinct children. Rejection profiles all transition to
                the same successor ProposeState (rejection only depends on
                the count) but the path token preserves the profile, so
                regret tables remain distinctly keyed.
  MissionState: action 0 = succeed, action 1 = fail

Counterfactual value tensor shape: (NUM_PLAYERS, 15) — per-player, per-info-set.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.assignments import (
    ASSIGNMENTS,
    INFO_SETS_PER_PLAYER,
    INFO_SET_INDEX,
    NUM_ASSIGNMENTS,
    evil_indices,
)
from src.game import (
    MAX_PROPOSALS,
    NUM_PLAYERS,
    Role,
    TEAM_SIZES,
    WIN_MISSIONS,
)
from src.regret_matching import regret_matching_plus
from src.strategy_likelihood import strategy_aware_likelihood
from src.value_bank import ValueBank
from src.value_net import win_prob_readout


# ---------------------------------------------------------------------------
# Public-state types (frozen, hashable — used as dict keys)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProposeState:
    round_idx: int
    proposer: int
    successes: int
    failures: int
    rejected_count: int


@dataclass(frozen=True)
class VoteState:
    round_idx: int
    proposer: int
    successes: int
    failures: int
    rejected_count: int
    team: Tuple[int, ...]


@dataclass(frozen=True)
class MissionState:
    round_idx: int
    proposer: int
    successes: int
    failures: int
    team: Tuple[int, ...]


@dataclass(frozen=True)
class ValueNetLeafState:
    round_idx: int
    proposer: int
    successes: int
    failures: int


@dataclass(frozen=True)
class TerminalState:
    """winner ∈ {"resistance", "spy", "assassinate_pending"}.

    "assassinate_pending" is a placeholder for subgames whose mission outcome
    triggers the assassination phase. The full DeepRole agent solves this
    separately;  We treat it as a null-utility leaf for now.
    """
    winner: str


# ---------------------------------------------------------------------------
# State helpers — moving players, action spaces, observations, transitions
# ---------------------------------------------------------------------------

def teams_for_size(team_size: int) -> Tuple[Tuple[int, ...], ...]:
    """Canonical ordered enumeration of teams of given size."""
    return tuple(combinations(range(NUM_PLAYERS), team_size))


# Vote profiles: 32 = 2^5 yes/no patterns, in canonical bit order
# (profile[i] is True iff player i voted yes).
VoteProfile = Tuple[bool, bool, bool, bool, bool]


def _enumerate_vote_profiles() -> Tuple[VoteProfile, ...]:
    profiles = []
    for bits in range(1 << NUM_PLAYERS):
        profile = tuple(bool((bits >> i) & 1) for i in range(NUM_PLAYERS))
        profiles.append(profile)
    return tuple(profiles)


VOTE_PROFILES: Tuple[VoteProfile, ...] = _enumerate_vote_profiles()
assert len(VOTE_PROFILES) == 32


def majority_approves(profile: VoteProfile) -> bool:
    """True iff strict majority (≥ 3 of 5) approve."""
    return sum(profile) > NUM_PLAYERS // 2


def moving_players(state) -> Tuple[int, ...]:
    if isinstance(state, ProposeState):
        return (state.proposer,)
    if isinstance(state, VoteState):
        return tuple(range(NUM_PLAYERS))
    if isinstance(state, MissionState):
        return state.team
    raise TypeError(f"No moving players for {type(state).__name__}")


def num_actions(state, _player: int) -> int:
    """Action count for a given player at this state. (Currently uniform across
    moving players at any single state in 5p Avalon.)"""
    if isinstance(state, ProposeState):
        return len(teams_for_size(TEAM_SIZES[state.round_idx]))
    if isinstance(state, VoteState):
        return 2
    if isinstance(state, MissionState):
        return 2
    raise TypeError(f"No actions for {type(state).__name__}")


def legal_observations(state):
    """Public observations available at this state — what the tree branches on."""
    if isinstance(state, ProposeState):
        return teams_for_size(TEAM_SIZES[state.round_idx])
    if isinstance(state, VoteState):
        return VOTE_PROFILES
    if isinstance(state, MissionState):
        return tuple(range(len(state.team) + 1))
    raise TypeError(f"No observations for {type(state).__name__}")


def child_state(state, observation):
    """Public state after observation."""
    if isinstance(state, ProposeState):
        team = observation
        return VoteState(
            round_idx=state.round_idx,
            proposer=state.proposer,
            successes=state.successes,
            failures=state.failures,
            rejected_count=state.rejected_count,
            team=team,
        )
    if isinstance(state, VoteState):
        if majority_approves(observation):
            return MissionState(
                round_idx=state.round_idx,
                proposer=state.proposer,
                successes=state.successes,
                failures=state.failures,
                team=state.team,
            )
        # Rejection — successor depends on count alone, not on the profile.
        new_rejected = state.rejected_count + 1
        if new_rejected >= MAX_PROPOSALS:
            return TerminalState(winner="spy")
        return ProposeState(
            round_idx=state.round_idx,
            proposer=(state.proposer + 1) % NUM_PLAYERS,
            successes=state.successes,
            failures=state.failures,
            rejected_count=new_rejected,
        )
    if isinstance(state, MissionState):
        f = observation
        new_s = state.successes + (1 if f == 0 else 0)
        new_f = state.failures + (1 if f > 0 else 0)
        if new_f >= WIN_MISSIONS:
            return TerminalState(winner="spy")
        if new_s >= WIN_MISSIONS:
            return TerminalState(winner="assassinate_pending")
        return ValueNetLeafState(
            round_idx=state.round_idx + 1,
            proposer=(state.proposer + 1) % NUM_PLAYERS,
            successes=new_s,
            failures=new_f,
        )
    raise TypeError(f"No transition for {type(state).__name__}")


def _path_token(state, observation):
    """Hashable token identifying (state-kind, observation) for path tuples."""
    return (type(state).__name__, observation)


# ---------------------------------------------------------------------------
# Deductive action recovery (deduceActions in the paper)
# ---------------------------------------------------------------------------

def _evil_action_dist(f: int, k_I: int) -> Optional[Tuple[float, float]]:
    """Action distribution (P(succeed), P(fail)) for an evil-on-team player
    given f fails observed and k_I evils on team in their info-set.

    Returns:
      (1.0, 0.0)  forced succeed (f == 0).
      (0.0, 1.0)  forced fail   (f == k_I).
      (0.0, 0.0)  impossible info-set (k_I == 0 or k_I < f).
      None        ambiguous (5p Avalon: only the (k_I=2, f=1) case).
                  Caller in `_cfr` resolves via the joint factorization.
    """
    if k_I == 0 or k_I < f:
        return (0.0, 0.0)
    if f == 0:
        return (1.0, 0.0)
    if f == k_I:
        return (0.0, 1.0)
    if k_I == 2 and f == 1:
        return None
    raise ValueError(
        f"Unexpected (f={f}, k_I={k_I}) in 5p Avalon — at most 2 evils, "
        f"so k_I ≤ 2 and f ≤ 2."
    )


# Sentinel value placed into action arrays at ambiguous mission rows.
# Negative is unambiguously not a valid probability; -1.0 is fine.
_AMBIGUOUS_SENTINEL = -1.0


def _responsibility_factor(my_pass: float, partner_pass: float) -> float:
    """Responsibility-weighted per-player reach factor for the (k_I=2, f=1) case.

    Returns f_me such that f_me * f_partner == P(exactly 1 fail | σ_me, σ_partner)
    where f_partner is _responsibility_factor(partner_pass, my_pass). The split
    is weighted by each player's strategy variance, so a near-deterministic
    player gets most of the responsibility.

    Reference: cfr_plus.cpp::my_single_pass_responsibility (Detry322/DeepRole).
    """
    outcome_prob = my_pass * (1.0 - partner_pass) + (1.0 - my_pass) * partner_pass
    if outcome_prob <= 0.0:
        return 0.0
    my_var = my_pass * my_pass + (1.0 - my_pass) * (1.0 - my_pass)
    partner_var = partner_pass * partner_pass + (1.0 - partner_pass) * (1.0 - partner_pass)
    denom = my_var + partner_var
    if denom <= 0.0:
        # Defensive: each *_var is in [0.5, 1.0], so denom is always ≥ 1.0.
        return outcome_prob ** 0.5
    my_exponent = my_var / denom
    return outcome_prob ** my_exponent


def _partner_viewpoint(player: int, viewpoint: int, partner: int) -> int:
    """For an evil player in `viewpoint` whose key identifies their fellow evil
    at seat `partner`, return the index of the symmetric info-set in the
    partner's INFO_SETS_PER_PLAYER list.

    The partner's role is the dual: if `player` is Spy (DS), partner is Assassin
    (and vice versa). Partner's matching info-set is keyed by (partner_role,
    player) — partner sees `player` at "their" evil seat.
    """
    player_key = INFO_SETS_PER_PLAYER[player][viewpoint]
    if player_key[0] is Role.DS:
        partner_role = Role.ASSASSIN
    elif player_key[0] is Role.ASSASSIN:
        partner_role = Role.DS
    else:
        raise ValueError(
            f"_partner_viewpoint called on non-evil viewpoint: {player_key!r}"
        )
    target_key = (partner_role, player)
    return INFO_SETS_PER_PLAYER[partner].index(target_key)


def deduce_actions(state, observation) -> Dict[int, np.ndarray]:
    """Return {player_idx: (15, |A|)} per-info-set action distribution per moving player.

    For each moving player i and each of i's 15 info-sets I, returns a
    distribution over actions consistent with the observation. The reach-prob
    update at this node is then ⃗π_i[I] *= Σ_a σ_i[I, a] · ⃗a_i[I][a].

    For ProposeState the action is deterministic per info-set (action = team).
    For VoteState the observation is a 5-tuple of yes/no votes, so each
    player's action is determined by their bit in the profile.
    For MissionState the action is deterministic except in the
    (k_I=2, f=1) case, where the marginal-uniform approximation kicks in.
    """
    n_info_sets = 15

    if isinstance(state, ProposeState):
        teams = teams_for_size(TEAM_SIZES[state.round_idx])
        team_idx = teams.index(observation)
        a = np.zeros((n_info_sets, len(teams)), dtype=np.float64)
        a[:, team_idx] = 1.0
        return {state.proposer: a}

    if isinstance(state, VoteState):
        # observation is a VoteProfile (bool, bool, bool, bool, bool).
        actions: Dict[int, np.ndarray] = {}
        for i in range(NUM_PLAYERS):
            action_idx = 0 if observation[i] else 1   # 0 = approve, 1 = reject
            a = np.zeros((n_info_sets, 2), dtype=np.float64)
            a[:, action_idx] = 1.0
            actions[i] = a
        return actions

    if isinstance(state, MissionState):
        team = state.team
        f = observation
        actions: Dict[int, np.ndarray] = {}
        for player_idx in team:
            a = np.zeros((n_info_sets, 2), dtype=np.float64)  # 0=S, 1=F
            for j, key in enumerate(INFO_SETS_PER_PLAYER[player_idx]):
                role = key[0]
                if role is Role.LS or role is Role.MERLIN:
                    # Good player on team — forced to S.
                    a[j, 0] = 1.0
                elif role is Role.DS or role is Role.ASSASSIN:
                    # Evil. Partner is at key[1] (Assassin's or Spy's seat).
                    partner_seat = key[1]
                    k_I = 1 + (1 if partner_seat in team else 0)
                    dist = _evil_action_dist(f, k_I)
                    if dist is None:
                        # Ambiguous — _cfr resolves via joint factorization.
                        a[j, :] = _AMBIGUOUS_SENTINEL
                    else:
                        a[j, 0], a[j, 1] = dist
                else:
                    raise ValueError(f"Unknown role in info-set key: {role!r}")
            actions[player_idx] = a
        return actions

    raise TypeError(f"No deduce_actions for {type(state).__name__}")


# ---------------------------------------------------------------------------
# Consistency mask — incremental, based on mission outcomes seen on path
# ---------------------------------------------------------------------------

def update_consistency_mask(mask: np.ndarray, mission_team: Tuple[int, ...], fails: int) -> np.ndarray:
    """Return a new mask AND-ing in: ρ inconsistent if |evil(ρ) ∩ team| < fails."""
    if fails == 0:
        return mask
    new_mask = mask.copy()
    team_set = set(mission_team)
    for a_idx, rho in enumerate(ASSIGNMENTS):
        if not new_mask[a_idx]:
            continue
        if len(evil_indices(rho) & team_set) < fails:
            new_mask[a_idx] = False
    return new_mask


# ---------------------------------------------------------------------------
# CFR solver
# ---------------------------------------------------------------------------

def _safe_divide(v: np.ndarray, reach_probs: np.ndarray) -> np.ndarray:
    """Divide elementwise; entries with zero reach become zero."""
    out = np.zeros_like(v)
    nonzero = reach_probs > 0
    out[nonzero] = v[nonzero] / reach_probs[nonzero]
    return out


def _utility_for_terminal(winner: str) -> np.ndarray:
    """Per-(player, assignment) utility at a terminal. Shape (5, 60).
    "spy" / "resistance" / "assassinate_pending" supported."""
    util = np.zeros((NUM_PLAYERS, NUM_ASSIGNMENTS), dtype=np.float64)
    if winner == "assassinate_pending":
        return util  # placeholder zero — assassinate solver handles this elsewhere
    if winner == "spy":
        evil_wins = +1.0
        good_wins = -1.0
    elif winner == "resistance":
        evil_wins = -1.0
        good_wins = +1.0
    else:
        raise ValueError(f"Unknown winner: {winner!r}")
    for a_idx, rho in enumerate(ASSIGNMENTS):
        for i in range(NUM_PLAYERS):
            util[i, a_idx] = evil_wins if rho[i].is_evil() else good_wins
    return util


# Cached terminal utilities (small, role-dependent only).
_TERMINAL_UTIL_CACHE: Dict[str, np.ndarray] = {
    "spy": _utility_for_terminal("spy"),
    "resistance": _utility_for_terminal("resistance"),
    "assassinate_pending": _utility_for_terminal("assassinate_pending"),
}


class CFRSolver:
    """Depth-limited vector-form CFR with deductive logic.

    Persistent state across iterations:
      regrets[(path, player)]      — (15, |A|) cumulative regret table.
      strategy_sum[(path, player)] — (15, |A|) reach-weighted strategy sum.

    Iteration loop:
      for it in 1..n_iters:
        w = max(it - averaging_delay, 0)
        run MODIFIEDCFR+ from root with reach=1, mask=all-true.

    After all iterations, average_strategies() returns the empirical average
    σ̄ at every visited (path, player), normalized per info-set.
    """

    def __init__(
        self,
        root_state,
        value_bank: ValueBank,
        prior_belief: np.ndarray = None,
    ) -> None:
        self.root_state = root_state
        self.value_bank = value_bank
        if prior_belief is None:
            prior_belief = np.full(NUM_ASSIGNMENTS, 1.0 / NUM_ASSIGNMENTS, dtype=np.float64)
        prior_belief = np.asarray(prior_belief, dtype=np.float64)
        assert prior_belief.shape == (NUM_ASSIGNMENTS,)
        assert prior_belief.sum() > 0
        self.prior_belief = prior_belief / prior_belief.sum()

        self.regrets: Dict[Tuple, np.ndarray] = {}
        self.strategy_sum: Dict[Tuple, np.ndarray] = {}

    # ---- top-level entry ------------------------------------------------

    def solve(self, n_iters: int, averaging_delay: int = 0) -> Dict[Tuple, np.ndarray]:
        """Algorithm 1 SOLVESITUATION. Returns the average strategies."""
        assert n_iters >= 1
        for it in range(1, n_iters + 1):
            w = max(it - averaging_delay, 0)
            reach = np.ones((NUM_PLAYERS, 15), dtype=np.float64)
            mask = np.ones(NUM_ASSIGNMENTS, dtype=bool)
            self._cfr(self.root_state, (), self.prior_belief, w, reach, mask)
        return self.average_strategies()

    def average_strategies(self) -> Dict[Tuple, np.ndarray]:
        """Per-(path, player) normalized average strategy across iterations."""
        avg: Dict[Tuple, np.ndarray] = {}
        for key, ssum in self.strategy_sum.items():
            total = ssum.sum(axis=-1, keepdims=True)
            n_actions = ssum.shape[-1]
            safe = np.where(total > 0, total, 1.0)
            avg_sigma = np.where(total > 0, ssum / safe, 1.0 / n_actions)
            avg[key] = avg_sigma
        return avg

    def root_strategy(self, player: int) -> np.ndarray:
        """Convenience: return the average strategy at the root for `player`."""
        return self.average_strategies()[((), player)]

    # ---- recursion ------------------------------------------------------

    def _cfr(self, state, path, b, w, reach_probs, mask):
        """MODIFIEDCFR+ (Algorithm 1 lines 10-46). Returns u (5, 15)."""
        if isinstance(state, TerminalState):
            return self._terminal_cfvs(state, b, reach_probs, mask)
        if isinstance(state, ValueNetLeafState):
            return self._neural_cfvs(state, b, reach_probs, mask)

        movers = moving_players(state)

        # Strategy via regret matching+ (lines 17-21).
        sigma: Dict[int, np.ndarray] = {}
        for i in movers:
            key = (path, i)
            if key not in self.regrets:
                shape = (15, num_actions(state, i))
                self.regrets[key] = np.zeros(shape, dtype=np.float64)
                self.strategy_sum[key] = np.zeros(shape, dtype=np.float64)
            sigma[i] = regret_matching_plus(self.regrets[key])

        u = np.zeros((NUM_PLAYERS, 15), dtype=np.float64)
        m: Dict[int, np.ndarray] = {i: np.zeros_like(sigma[i]) for i in movers}

        # Iterate observations (line 22).
        for o in legal_observations(state):
            actions_o = deduce_actions(state, o)

            # MissionState ambiguous rows: replace sentinel with the conditional
            # action distribution (used for m[i] and σ·a sums), and remember
            # the responsibility-weighted reach factor for those rows.
            sentinel_reach: Dict[Tuple[int, int], float] = {}
            if isinstance(state, MissionState):
                for i in movers:
                    a_i = actions_o[i]
                    for j in range(15):
                        if a_i[j, 0] != _AMBIGUOUS_SENTINEL:
                            continue
                        key = INFO_SETS_PER_PLAYER[i][j]
                        partner_seat = key[1]
                        partner_view = _partner_viewpoint(i, j, partner_seat)
                        my_pass = sigma[i][j, 0]
                        partner_pass = sigma[partner_seat][partner_view, 0]
                        outcome_prob = (
                            my_pass * (1.0 - partner_pass)
                            + (1.0 - my_pass) * partner_pass
                        )
                        if outcome_prob > 0.0:
                            a_i[j, 0] = my_pass * (1.0 - partner_pass) / outcome_prob
                            a_i[j, 1] = (1.0 - my_pass) * partner_pass / outcome_prob
                        else:
                            a_i[j, 0] = 0.0
                            a_i[j, 1] = 0.0
                        sentinel_reach[(i, j)] = _responsibility_factor(my_pass, partner_pass)

            # Reach-prob update (lines 23-26). Branch-local copy.
            # For sentinel rows, override σ·a with the responsibility factor.
            new_reach = reach_probs.copy()
            for i in movers:
                factor = (sigma[i] * actions_o[i]).sum(axis=-1)        # (15,)
                for (mover, j), rf in sentinel_reach.items():
                    if mover == i:
                        factor[j] = rf
                new_reach[i] = reach_probs[i] * factor

            child_mask = mask
            if isinstance(state, MissionState):
                child_mask = update_consistency_mask(mask, state.team, o)

            child = child_state(state, o)
            child_path = path + (_path_token(state, o),)
            u_child = self._cfr(child, child_path, b, w, new_reach, child_mask)

            # Aggregate u and m (lines 28-35). u uses the conditional action
            # distribution (σ·a sum); the reach already used the responsibility
            # factor for sentinel rows above.
            for i in range(NUM_PLAYERS):
                if i in movers:
                    factor = (sigma[i] * actions_o[i]).sum(axis=-1)    # (15,)
                    u[i] += factor * u_child[i]
                    # Scatter-add: m_i[I, a] += a_i[I, a] * u_child_i[I].
                    m[i] += actions_o[i] * u_child[i][:, None]
                else:
                    u[i] += u_child[i]

        # Regret + strategy-sum update (lines 37-43). CFR+ floors regrets.
        for i in movers:
            key = (path, i)
            self.regrets[key] = np.maximum(
                self.regrets[key] + m[i] - u[i, :, None], 0.0
            )
            self.strategy_sum[key] += reach_probs[i, :, None] * sigma[i] * w

        return u

    # ---- terminal value calculation (Algorithm 2) -----------------------

    def _calc_terminal_belief(self, b, reach_probs, mask):
        """CALCTERMINALBELIEF — Equation 1 with strategy-aware likelihood
        and consistency mask. Returns (60,) unnormalized terminal belief."""
        lik = strategy_aware_likelihood(reach_probs)
        return b * lik * mask.astype(np.float64)

    def _terminal_cfvs(self, state: TerminalState, b, reach_probs, mask):
        """TERMINALCFVS — exact factual values from u_i(ρ), divided by reach."""
        bterm = self._calc_terminal_belief(b, reach_probs, mask)
        util = _TERMINAL_UTIL_CACHE[state.winner]                  # (5, 60)
        # v[i, j] = sum_a 1{INFO_SET_INDEX[i, a] = j} * bterm[a] * util[i, a]
        v = np.zeros((NUM_PLAYERS, 15), dtype=np.float64)
        for i in range(NUM_PLAYERS):
            np.add.at(v[i], INFO_SET_INDEX[i], bterm * util[i])
        return _safe_divide(v, reach_probs)

    def _neural_cfvs(self, state: ValueNetLeafState, b, reach_probs, mask):
        """NEURALCFVS — value-net leaf with belief renormalization."""
        bterm = self._calc_terminal_belief(b, reach_probs, mask)
        w_sum = float(bterm.sum())
        if w_sum <= 0.0:
            return np.zeros((NUM_PLAYERS, 15), dtype=np.float64)
        bterm_norm = bterm / w_sum
        net = self.value_bank.get(state.proposer, state.successes, state.failures)
        belief_t = torch.from_numpy(bterm_norm)
        with torch.no_grad():
            w_pred = net.forward(state.proposer, belief_t)         # (60,) f32
            v_iset = win_prob_readout(w_pred, belief_t)            # (5, 15) f64
        v_factual = w_sum * v_iset.numpy()
        return _safe_divide(v_factual, reach_probs)
