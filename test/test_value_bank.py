"""Tests for value_bank.py — 45-key dispatch and parameter independence."""
import pytest
import torch

from src.assignments import NUM_ASSIGNMENTS
from src.value_bank import VALID_KEYS, ValueBank
from src.value_net import ValueNetwork


def test_45_distinct_valid_keys():
    assert len(VALID_KEYS) == 45
    assert len(set(VALID_KEYS)) == 45
    for (p, s, f) in VALID_KEYS:
        assert 0 <= p < 5
        assert 0 <= s < 3
        assert 0 <= f < 3


def test_bank_get_returns_a_value_network_for_each_valid_key():
    bank = ValueBank()
    for (p, s, f) in VALID_KEYS:
        net = bank.get(p, s, f)
        assert isinstance(net, ValueNetwork)


def test_bank_get_invalid_key_raises():
    bank = ValueBank()
    for bad in [(0, 3, 0), (0, 0, 3), (5, 0, 0), (-1, 0, 0), (0, -1, 0)]:
        with pytest.raises(KeyError):
            bank.get(*bad)


def test_bank_contains():
    bank = ValueBank()
    assert (0, 0, 0) in bank
    assert (4, 2, 2) in bank
    assert (0, 3, 0) not in bank
    assert (5, 0, 0) not in bank


def test_bank_keys_matches_module_constant():
    bank = ValueBank()
    assert bank.keys() == VALID_KEYS


def test_bank_total_parameter_count_equals_45_times_one_net():
    """Bank holds 45 independent networks; total params = 45 * per-net count."""
    bank = ValueBank()
    one_net = ValueNetwork()
    per_net = sum(p.numel() for p in one_net.parameters())
    total = sum(p.numel() for p in bank.parameters())
    assert total == 45 * per_net


def test_bank_parameter_independence():
    """Backprop through one network must NOT touch any other network's gradients."""
    torch.manual_seed(0)
    bank = ValueBank()
    n_a = bank.get(0, 0, 0)
    n_b = bank.get(1, 0, 0)

    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    out = n_a.forward(0, belief)
    out.sum().backward()

    a_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in n_a.parameters()
    )
    b_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in n_b.parameters()
    )
    assert a_has_grad
    assert not b_has_grad


def test_bank_distinct_initial_weights_across_keys():
    """Each ValueNetwork is constructed independently; the first-layer weights
    of two different keys should not be byte-identical."""
    torch.manual_seed(0)
    bank = ValueBank()
    w_a = bank.get(0, 0, 0).net[0].weight
    w_b = bank.get(1, 0, 0).net[0].weight
    assert not torch.allclose(w_a, w_b)


def test_bank_dispatch_returns_same_instance_each_call():
    """get() returns the same module each time (not a fresh copy)."""
    bank = ValueBank()
    assert bank.get(2, 1, 1) is bank.get(2, 1, 1)
