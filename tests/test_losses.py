"""
Unit tests for losses.py — the masked, depth-aware loss functions used to
train the DL_bathy models.

These tests only require torch (no GDAL / rasterio / GEE), so they run
quickly in CI on every push and pull request.
"""

import math

import pytest
import torch

from losses import (
    masked_sum_sse_and_count,
    masked_sum_abs_and_count,
    masked_sum_rpe_and_count,
    masked_sum_swf_and_count,
    masked_sum_huber_and_count,
)


def test_sse_matches_manual_computation():
    pred   = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.5, 2.0, 3.5, 10.0])
    mask   = torch.tensor([1, 1, 1, 0])  # last pixel masked out

    total, n_valid = masked_sum_sse_and_count(pred, target, mask)

    expected_total = (0.5**2) + (0.0**2) + (0.5**2)  # pixel 4 excluded
    assert n_valid.item() == 3
    assert math.isclose(total.item(), expected_total, rel_tol=1e-6)


def test_sse_all_masked_returns_zero():
    pred   = torch.tensor([1.0, 2.0])
    target = torch.tensor([5.0, 5.0])
    mask   = torch.tensor([0, 0])

    total, n_valid = masked_sum_sse_and_count(pred, target, mask)

    assert n_valid.item() == 0
    assert total.item() == 0.0


def test_mae_matches_manual_computation():
    pred   = torch.tensor([1.0, 2.0, 10.0])
    target = torch.tensor([1.0, 5.0, 10.0])
    mask   = torch.tensor([1, 1, 1])

    total, n_valid = masked_sum_abs_and_count(pred, target, mask)

    assert n_valid.item() == 3
    assert math.isclose(total.item(), 0.0 + 3.0 + 0.0, rel_tol=1e-6)


def test_rpe_matches_manual_computation():
    pred   = torch.tensor([1.1, 4.0])
    target = torch.tensor([1.0, 5.0])
    mask   = torch.tensor([1, 1])

    total, n_valid = masked_sum_rpe_and_count(pred, target, mask)

    expected = abs(1.1 - 1.0) / 1.0 + abs(4.0 - 5.0) / 5.0
    assert n_valid.item() == 2
    assert math.isclose(total.item(), expected, rel_tol=1e-5)


def test_swf_weights_shallow_pixels_more_than_deep():
    # Same absolute error (=1) at a shallow depth vs. a deep depth: SWF should
    # produce a larger weighted loss for the shallow pixel because the
    # weighting function w = 1 + beta * exp(-|Z|/Z0) is larger near the
    # surface than at depth.
    mask = torch.tensor([1])

    # error = |pred - target| = 1 in both cases; only depth differs.
    shallow_loss, _ = masked_sum_swf_and_count(
        torch.tensor([1.0]), torch.tensor([2.0]), mask, beta=5.0, Z0=10.0  # depth = 2 m
    )
    deep_loss, _ = masked_sum_swf_and_count(
        torch.tensor([18.0]), torch.tensor([19.0]), mask, beta=5.0, Z0=10.0  # depth = 19 m
    )

    assert shallow_loss.item() > deep_loss.item()


def test_swf_matches_manual_computation():
    pred   = torch.tensor([0.0])
    target = torch.tensor([5.0])
    mask   = torch.tensor([1])
    beta, Z0 = 5.0, 10.0

    total, n_valid = masked_sum_swf_and_count(pred, target, mask, beta=beta, Z0=Z0)

    weight = 1.0 + beta * math.exp(-abs(5.0) / Z0)
    expected = weight * (0.0 - 5.0) ** 2
    assert n_valid.item() == 1
    assert math.isclose(total.item(), expected, rel_tol=1e-5)


def test_huber_behaves_as_l2_below_delta_and_l1_above():
    delta = 1.0
    # small error -> quadratic (L2) branch
    pred_small   = torch.tensor([0.0])
    target_small = torch.tensor([0.5])
    # large error -> linear (L1) branch
    pred_large   = torch.tensor([0.0])
    target_large = torch.tensor([5.0])
    mask = torch.tensor([1])

    small_total, _ = masked_sum_huber_and_count(pred_small, target_small, mask, delta=delta)
    large_total, _ = masked_sum_huber_and_count(pred_large, target_large, mask, delta=delta)

    assert math.isclose(small_total.item(), 0.5 * 0.5**2, rel_tol=1e-5)
    assert math.isclose(large_total.item(), delta * 5.0 - 0.5 * delta**2, rel_tol=1e-5)


@pytest.mark.parametrize(
    "loss_fn",
    [
        masked_sum_sse_and_count,
        masked_sum_abs_and_count,
        masked_sum_rpe_and_count,
        masked_sum_swf_and_count,
        masked_sum_huber_and_count,
    ],
)
def test_all_losses_are_non_negative(loss_fn):
    torch.manual_seed(0)
    pred   = torch.randn(100)
    target = torch.randn(100) * 5  # depths on a plausible scale
    mask   = (torch.rand(100) > 0.3).int()

    total, n_valid = loss_fn(pred, target, mask)
    assert total.item() >= 0.0
    assert n_valid.item() == mask.sum().item()
