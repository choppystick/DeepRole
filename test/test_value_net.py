"""Tests for value_net.py — forward shapes/range, readout math, gradient flow."""
import numpy as np
import pytest
import torch

from src.assignments import NUM_ASSIGNMENTS
from src.game import NUM_PLAYERS, Role
from src.value_net import (
    GOOD_MASK,
    GOOD_MASK_T,
    M_MATRIX_T,
    ValueNetwork,
    win_prob_readout,
)


# ---------------------------------------------------------------------------
# GOOD_MASK
# ---------------------------------------------------------------------------

def test_good_mask_shape():
    assert GOOD_MASK.shape == (NUM_PLAYERS, NUM_ASSIGNMENTS)


def test_good_mask_three_good_per_assignment():
    """Every assignment has exactly 3 good players (Merlin + 2 LS)."""
    np.testing.assert_array_equal(
        GOOD_MASK.sum(axis=0),
        np.full(NUM_ASSIGNMENTS, 3.0, dtype=np.float32),
    )


def test_good_mask_role_consistency():
    """GOOD_MASK[i, a] agrees with ASSIGNMENTS[a][i].is_good()."""
    from src.assignments import ASSIGNMENTS
    for i in range(NUM_PLAYERS):
        for a, rho in enumerate(ASSIGNMENTS):
            expected = 1.0 if rho[i].is_good() else 0.0
            assert GOOD_MASK[i, a] == expected


# ---------------------------------------------------------------------------
# ValueNetwork forward
# ---------------------------------------------------------------------------

def test_forward_shape():
    torch.manual_seed(0)
    net = ValueNetwork()
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    out = net.forward(proposer=0, belief=belief)
    assert out.shape == (NUM_ASSIGNMENTS,)


def test_forward_output_in_unit_interval():
    torch.manual_seed(0)
    net = ValueNetwork()
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    out = net.forward(proposer=0, belief=belief).detach()
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_forward_dtype_float32():
    """Net runs in float32 internally even when belief is float64."""
    torch.manual_seed(0)
    net = ValueNetwork()
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    out = net.forward(proposer=0, belief=belief)
    assert out.dtype == torch.float32


def test_forward_proposer_changes_output():
    """Different proposer one-hots should produce different outputs in a fresh net."""
    torch.manual_seed(0)
    net = ValueNetwork()
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float32)
    outs = [net.forward(p, belief).detach() for p in range(NUM_PLAYERS)]
    # Probability of all-equal under random init is 0; require at least one diff.
    differ = any(
        not torch.allclose(outs[i], outs[j])
        for i in range(NUM_PLAYERS) for j in range(i + 1, NUM_PLAYERS)
    )
    assert differ


def test_forward_rejects_wrong_belief_shape():
    net = ValueNetwork()
    with pytest.raises(AssertionError):
        net.forward(0, torch.zeros(59))


def test_forward_rejects_invalid_proposer():
    net = ValueNetwork()
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float32)
    with pytest.raises(AssertionError):
        net.forward(-1, belief)
    with pytest.raises(AssertionError):
        net.forward(5, belief)


# ---------------------------------------------------------------------------
# win_prob_readout
# ---------------------------------------------------------------------------

def test_readout_shape_and_dtype():
    w = torch.rand(NUM_ASSIGNMENTS)
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    V = win_prob_readout(w, belief)
    assert V.shape == (NUM_PLAYERS, 15)
    assert V.dtype == torch.float64


def test_readout_constant_w_half():
    """w=0.5 everywhere => payoff=0.5 for every (i,a) =>
       V_i[j] = 0.5 * sum_a M_i[j,a] * b[a]."""
    w = torch.full((NUM_ASSIGNMENTS,), 0.5)
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    V = win_prob_readout(w, belief)
    expected = 0.5 * torch.einsum("ija,a->ij", M_MATRIX_T.to(torch.float64), belief)
    torch.testing.assert_close(V, expected)


def test_readout_w_one_good_wins():
    """w=1 => good payoff=1, evil payoff=0 =>
       V_i[j] = sum_a M_i[j,a] * b[a] * good_mask[i,a]."""
    w = torch.ones(NUM_ASSIGNMENTS)
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    V = win_prob_readout(w, belief)
    expected = torch.einsum(
        "ija,ia->ij",
        M_MATRIX_T.to(torch.float64),
        GOOD_MASK_T.to(torch.float64) * belief.unsqueeze(0),
    )
    torch.testing.assert_close(V, expected)


def test_readout_w_zero_evil_wins():
    """w=0 => good payoff=0, evil payoff=1 =>
       V_i[j] = sum_a M_i[j,a] * b[a] * (1 - good_mask[i,a])."""
    w = torch.zeros(NUM_ASSIGNMENTS)
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    V = win_prob_readout(w, belief)
    evil_mask = 1.0 - GOOD_MASK_T.to(torch.float64)
    expected = torch.einsum(
        "ija,ia->ij",
        M_MATRIX_T.to(torch.float64),
        evil_mask * belief.unsqueeze(0),
    )
    torch.testing.assert_close(V, expected)


def test_readout_handcomputed_LS_info_set():
    """Player 0's LS info set covers exactly the 24 assignments where
    player 0 is LS. With uniform belief b=1/60 and w=1 (good always wins):
       V_0[LS-index] = sum_{a: rho[0]=LS} (1/60) * 1 = 24/60 = 0.4
    """
    from src.assignments import INFO_SETS_PER_PLAYER
    ls_idx = next(
        j for j, key in enumerate(INFO_SETS_PER_PLAYER[0]) if key[0] is Role.LS
    )

    w = torch.ones(NUM_ASSIGNMENTS)
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    V = win_prob_readout(w, belief)
    assert V[0, ls_idx].item() == pytest.approx(24.0 / 60.0)


# ---------------------------------------------------------------------------
# Gradient flow through the readout into MLP weights
# ---------------------------------------------------------------------------

def test_gradient_flows_through_readout():
    """Backprop a scalar loss V.sum() into every parameter of the MLP."""
    torch.manual_seed(0)
    net = ValueNetwork()
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    w = net.forward(proposer=0, belief=belief)
    V = win_prob_readout(w, belief)
    V.sum().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name}: grad is None"
        assert p.grad.abs().sum().item() > 0.0, f"{name}: zero gradient"


def test_readout_no_learnable_parameters():
    """win_prob_readout is a free function; it must not introduce parameters.
    Sanity: the only parameters in a (net, readout) computation belong to the net."""
    net = ValueNetwork()
    belief = torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=torch.float64)
    w = net.forward(0, belief)
    V = win_prob_readout(w, belief)
    # All tensors used by readout (M_MATRIX_T, GOOD_MASK_T) have requires_grad=False.
    assert M_MATRIX_T.requires_grad is False
    assert GOOD_MASK_T.requires_grad is False
    # Output participates in autograd via w.
    assert V.requires_grad
