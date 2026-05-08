"""Tests for the Section 5 training pipeline.

Smaller and quicker than the laptop-default scale: tests run with a few
samples and a few iterations to keep the suite fast.
"""
import numpy as np
import pytest
import torch

from src.assignments import NUM_ASSIGNMENTS
from src.cfr import ProposeState
from src.game import NUM_PLAYERS
from src.training import (
    Datapoint,
    _game_parts_in_dependency_order,
    _root_for_part,
    _v_target_from_cfr,
    endtoend_train,
    generate_datapoints,
    train_value_network,
)
from src.value_bank import ValueBank
from src.value_net import ValueNetwork


# ---------------------------------------------------------------------------
# _v_target_from_cfr
# ---------------------------------------------------------------------------

def test_v_target_shape_and_range():
    rng = np.random.default_rng(0)
    cf = rng.uniform(-1, 1, size=(NUM_PLAYERS, 15))
    belief = np.full(NUM_ASSIGNMENTS, 1.0 / NUM_ASSIGNMENTS)
    target = _v_target_from_cfr(cf, belief)
    assert target.shape == (NUM_ASSIGNMENTS,)
    assert (target >= 0).all() and (target <= 1).all()


def test_v_target_constant_one_corresponds_to_good_winning():
    """If V[i, j] = +1 for every i good in ρ, target ≈ 1 (good always wins)."""
    cf = np.ones((NUM_PLAYERS, 15))   # +1 everywhere
    # For good i in ρ: prob = 1. For evil i: prob = (1 - 1)/2 = 0.
    # Average = 3/5 = 0.6 (3 good, 2 evil per ρ).
    target = _v_target_from_cfr(cf, np.full(NUM_ASSIGNMENTS, 1.0 / NUM_ASSIGNMENTS))
    np.testing.assert_allclose(target, np.full(NUM_ASSIGNMENTS, 0.6))


def test_v_target_constant_negative_one_corresponds_to_evil_winning():
    """If V[i, j] = -1 everywhere: good i prob = 0, evil i prob = 1.
    Average = 2/5 = 0.4 per ρ."""
    cf = -np.ones((NUM_PLAYERS, 15))
    target = _v_target_from_cfr(cf, np.full(NUM_ASSIGNMENTS, 1.0 / NUM_ASSIGNMENTS))
    np.testing.assert_allclose(target, np.full(NUM_ASSIGNMENTS, 0.4))


# ---------------------------------------------------------------------------
# Subgame root construction
# ---------------------------------------------------------------------------

def test_root_for_part_round_idx_matches_s_plus_f():
    for s in range(3):
        for f in range(3):
            root = _root_for_part(proposer=2, s=s, f=f)
            assert isinstance(root, ProposeState)
            assert root.round_idx == s + f
            assert root.successes == s
            assert root.failures == f
            assert root.rejected_count == 0
            assert root.proposer == 2


# ---------------------------------------------------------------------------
# generate_datapoints
# ---------------------------------------------------------------------------

def test_generate_datapoints_returns_correct_shape():
    """Smoke test: generate a tiny batch with minimal CFR iterations."""
    torch.manual_seed(0)
    bank = ValueBank()
    rng = np.random.default_rng(0)
    # Use s=2, f=2 — the smallest game part, fastest to solve.
    data = generate_datapoints(
        s=2, f=2, value_bank=bank,
        n_samples=2, n_cfr_iters=2, rng=rng,
    )
    assert len(data) == 2
    for d in data:
        assert isinstance(d, Datapoint)
        assert 0 <= d.proposer < NUM_PLAYERS
        assert d.belief.shape == (NUM_ASSIGNMENTS,)
        assert d.target.shape == (NUM_ASSIGNMENTS,)
        assert d.belief.sum() == pytest.approx(1.0)
        assert (d.target >= 0).all() and (d.target <= 1).all()


def test_generate_datapoints_deterministic_with_seed():
    """Reproducibility: same seed + same bank → same data."""
    torch.manual_seed(0)
    bank_a = ValueBank()
    rng_a = np.random.default_rng(123)
    data_a = generate_datapoints(s=2, f=2, value_bank=bank_a,
                                  n_samples=2, n_cfr_iters=2, rng=rng_a)

    torch.manual_seed(0)
    bank_b = ValueBank()
    rng_b = np.random.default_rng(123)
    data_b = generate_datapoints(s=2, f=2, value_bank=bank_b,
                                  n_samples=2, n_cfr_iters=2, rng=rng_b)

    assert len(data_a) == len(data_b)
    for da, db in zip(data_a, data_b):
        assert da.proposer == db.proposer
        np.testing.assert_array_equal(da.belief, db.belief)
        np.testing.assert_allclose(da.target, db.target, atol=1e-12)


# ---------------------------------------------------------------------------
# train_value_network
# ---------------------------------------------------------------------------

def test_train_value_network_reduces_loss_on_synthetic_data():
    """If we hand-construct a tiny dataset where target == net output for some
    fixed input, training should drive train loss toward zero."""
    torch.manual_seed(0)
    net = ValueNetwork()
    rng = np.random.default_rng(0)
    # Generate 16 random (proposer, belief, target) datapoints. Target chosen
    # arbitrarily; we only check that train loss DECREASES across epochs.
    data = []
    for _ in range(16):
        p = int(rng.integers(0, NUM_PLAYERS))
        b = rng.dirichlet(np.ones(NUM_ASSIGNMENTS))
        t = rng.uniform(0, 1, size=NUM_ASSIGNMENTS)
        data.append(Datapoint(proposer=p, belief=b, target=t))

    history = train_value_network(
        net, data, epochs=30, batch_size=8, lr=1e-2, val_split=0.0, seed=0,
    )
    assert len(history["train_losses"]) == 30
    # Loss should drop substantially over 30 epochs on this tiny problem.
    assert history["train_losses"][-1] < 0.5 * history["train_losses"][0], (
        history["train_losses"][0], history["train_losses"][-1]
    )


def test_train_value_network_handles_empty_dataset():
    net = ValueNetwork()
    history = train_value_network(net, [], epochs=5)
    assert history["train_losses"] == []
    assert history["val_losses"] == []


def test_train_value_network_validation_split_runs():
    torch.manual_seed(0)
    net = ValueNetwork()
    rng = np.random.default_rng(0)
    data = [
        Datapoint(
            proposer=int(rng.integers(0, NUM_PLAYERS)),
            belief=rng.dirichlet(np.ones(NUM_ASSIGNMENTS)),
            target=rng.uniform(0, 1, size=NUM_ASSIGNMENTS),
        )
        for _ in range(20)
    ]
    history = train_value_network(
        net, data, epochs=3, batch_size=4, lr=1e-3, val_split=0.2, seed=0,
    )
    # Both train and val have one entry per epoch.
    assert len(history["train_losses"]) == 3
    assert len(history["val_losses"]) == 3
    for v in history["val_losses"]:
        assert not np.isnan(v)


# ---------------------------------------------------------------------------
# endtoend_train (Algorithm 3) — smoke test only, scale is tiny
# ---------------------------------------------------------------------------

def test_game_parts_in_dependency_order():
    """Deepest game parts (s+f largest) come first; (0,0) comes last."""
    parts = _game_parts_in_dependency_order()
    assert len(parts) == 9
    assert parts[0] == (2, 2)
    assert parts[-1] == (0, 0)
    # Monotonic non-increasing in s+f.
    sums = [s + f for s, f in parts]
    assert sums == sorted(sums, reverse=True)


def test_endtoend_train_smoke():
    """Smoke test: run endtoend_train at TEST scale (tiny). Verifies the
    pipeline runs through all 45 networks without crashing and returns history.

    With s=2, f=2 having no AssassinateState reach (in this tiny rejected_count
    case), and per the assassinate solver added in 5.1, all game parts should
    successfully solve and train."""
    torch.manual_seed(0)
    bank = ValueBank()
    history = endtoend_train(
        bank,
        n_samples_per_part=2,
        n_cfr_iters=2,
        epochs=2,
        batch_size=4,
        val_split=0.0,
        seed=0,
    )
    # 9 (s, f) parts × 5 proposers = 45 trained nets.
    assert len(history) == 45
    for key, h in history.items():
        proposer, s, f = key
        assert 0 <= proposer < NUM_PLAYERS
        assert 0 <= s < 3 and 0 <= f < 3
        assert len(h["train_losses"]) == 2
