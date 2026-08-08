#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
losses.py
=========
Masked, depth-aware loss functions for satellite-derived bathymetry (SDB)
training. Extracted from step2_train_test.py so they can be unit-tested
without importing the heavier GDAL / rasterio / GEE dependencies used
elsewhere in the pipeline.

Each function takes a prediction tensor, a target tensor, and a boolean
(or 0/1) validity mask of the same shape, and returns a tuple of
(summed loss over valid pixels, count of valid pixels). Dividing the two
gives the mean loss; keeping them separate lets callers accumulate sums
and counts across multiple batches before computing a single epoch-level
mean.
"""

import torch


def masked_sum_sse_and_count(pred, target, mask):
    """Sum of squared errors over valid pixels, plus valid-pixel count."""
    mask_bool = mask.bool()
    n_valid   = mask_bool.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
    diff = pred[mask_bool] - target[mask_bool]
    return (diff * diff).sum(), n_valid


def masked_sum_abs_and_count(pred, target, mask):
    """Sum of absolute errors (MAE numerator) over valid pixels."""
    mask_bool = mask.bool()
    n_valid   = mask_bool.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
    mae = torch.abs(pred[mask_bool] - target[mask_bool]).sum()
    return mae, n_valid


def masked_sum_rpe_and_count(pred, target, mask, eps=1e-6):
    """Sum of relative percentage errors over valid pixels."""
    mask_bool = mask.bool()
    n_valid   = mask_bool.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)
    pred_v = pred[mask_bool]
    targ_v = target[mask_bool]
    denom  = torch.abs(targ_v).clamp(min=eps)
    return (torch.abs(pred_v - targ_v) / denom).sum(), n_valid


def masked_sum_swf_and_count(pred, target, mask, beta=5.0, Z0=20.0):
    """
    Smooth Weight Function (SWF) loss - depth-weighted squared error.

    Weight per pixel: w_i = 1 + beta * exp(-|Z_i| / Z0)
    Shallow pixels receive a higher weight, encouraging accuracy near the surface.

    Returns the sum of weighted squared errors and the valid-pixel count.
    """
    mask_bool = mask.bool()
    n_valid   = mask_bool.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)

    pred_v = pred[mask_bool]
    targ_v = target[mask_bool]
    diff   = pred_v - targ_v
    weight = 1.0 + beta * torch.exp(-torch.abs(targ_v) / Z0)
    return (weight * diff * diff).sum(), n_valid


def masked_sum_huber_and_count(pred, target, mask, delta=1.0):
    """
    Huber loss (smooth L1) over valid pixels.

    Behaves like L2 for small errors (|err| < delta) and L1 for large errors,
    making it robust to depth outliers.
    """
    mask_bool = mask.bool()
    n_valid   = mask_bool.sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=pred.device), torch.tensor(0.0, device=pred.device)

    diff         = torch.abs(pred[mask_bool] - target[mask_bool])
    is_l2        = diff < delta
    l2_loss      = 0.5 * diff[is_l2] ** 2
    l1_loss      = delta * diff[~is_l2] - 0.5 * delta ** 2
    return l2_loss.sum() + l1_loss.sum(), n_valid
