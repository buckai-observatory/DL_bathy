#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_train_test.py
===================
Trains a DeepLabV3+ segmentation model for satellite-derived bathymetry (SDB)
using GeoTIFF patches produced by step1_splitting.py.

Key features:
  - Lazy GeoTIFF loading to prevent out-of-memory errors on large datasets
  - Configurable encoder backbone: ConvNeXt-Large, EfficientNet-B4, ResNet-50/101
  - Validity-mask-aware loss and metrics (ignores NaN/cloud pixels)
  - Multiple loss options: RMSE (SSE), MAE, RPE, SWF (depth-weighted RMSE), Huber
  - Early stopping and ReduceLROnPlateau learning-rate scheduling
  - Multi-GPU training via DataParallel
  - Full test inference with per-patch GeoTIFF predictions and visualisation plots
"""

from osgeo import gdal
import rasterio
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import datetime
import os, json, csv, random, glob, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
import segmentation_models_pytorch as smp
import sys

startime = timer()

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Path to the patch dataset produced by step1_splitting.py
TrainingDataPath = "./Data"

# Output directory for this training run (models, logs, plots)
LiDAR_Model = "Model_output"
os.makedirs(LiDAR_Model, exist_ok=True)

# Redirect all console output to a log file
log_path    = os.path.join(LiDAR_Model, "console_output_step2.log")
sys.stdout  = open(log_path, "w", buffering=1)   # line-buffered
sys.stderr  = sys.stdout

# Worker count: defaults to SLURM_CPUS_PER_TASK if set, otherwise 16
num_workers_default = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
available_cpus      = os.cpu_count() or 1
print(f"CPU cores available: {available_cpus}, DataLoader workers: {num_workers_default}")

# --- Hyperparameters and run settings ---
params = {
    # Encoder backbone: "tu-convnext_large" | "efficientnet-b4" | "resnet101" | "resnet50"
    "Model_mode":              "tu-convnext_large",

    "depth_scale":             1,       # Depth scale factor (1 = metres, no rescaling)
    "min_valid_ratio":         0.1,     # Minimum valid-pixel fraction for training patches
    "min_valid_ratio_for_val": 0.1,     # Minimum valid-pixel fraction for validation patches
    "grad_clip_norm":          5.0,     # L2 gradient clipping threshold

    # Output activation of the segmentation head.
    # Use "none" (i.e., linear) for most backbones; "softplus" or "relu" if desired.
    "output_activation":       "none",
    "clamp_output":            False,   # If True, clamp predictions to (-100, 0) at inference

    "num_epochs":              1000,    # Max epochs (early stopping will terminate sooner)
    "batch_size":              16,
    "lr":                      1e-4,
    "earlystop_patience":      10,      # Epochs without improvement before stopping
    "reduce_lr_patience":      5,       # ReduceLROnPlateau patience

    "depth_threshold":         20,      # Depth boundary (metres) for deep/shallow classification
    "depth_type":              "deep",  # "deep" or "shallow" – which side to keep for training

    # Loss function: "sse" (RMSE) | "mae" | "rpe" | "swf" (depth-weighted RMSE) | "huber"
    "loss_type":               "swf",
    "SWF_beta":                5.0,     # SWF weight amplitude  (w = 1 + beta * exp(-|Z|/Z0))
    "SWF_Z0":                  10.0,    # SWF decay depth (metres)

    # Input scaling: "Log_band" (pre-applied in step1) | "divide_10000" | "minmax_manual" | "none"
    "scaling_method":          "Log_band",

    "lambda_l1":               0,       # L1 regularisation weight (0 = disabled)
    "lambda_l2":               0,       # L2 regularisation weight (AdamW handles this via weight_decay)
    "lambda_custom":           0,       # Total-variation regularisation weight (0 = disabled)

    # Sub-folder names inside TrainingDataPath
    "train_source":            "train_aug",
    "valid_source":            "valid_aug",
    "test_source":             "test",

    "LiDAR_Model":             LiDAR_Model,
    "num_workers":             num_workers_default,
}

dpi = 300  # Resolution for saved figures

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_reproducible(seed=42):
    """Set all relevant random seeds for reproducible training."""
    os.environ["PYTHONHASHSEED"]          = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True

seed = 42
set_reproducible(seed)

# =============================================================================
# LOGGING
# =============================================================================

def log_params(params, log_dir):
    """Write hyperparameter dict to a timestamped text file and print it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"hyperparams_{timestamp}.txt")
    lines     = ["=== Training Parameters ==="] + [f"{k:25}: {v}" for k, v in params.items()]
    text      = "\n".join(lines)
    print(text)
    with open(log_file, "w") as f:
        f.write(text + "\n")
    return log_file

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
# Extracted to losses.py so they can be unit-tested without the GDAL /
# rasterio / GEE dependencies used elsewhere in this script.
from losses import (
    masked_sum_sse_and_count,
    masked_sum_abs_and_count,
    masked_sum_rpe_and_count,
    masked_sum_swf_and_count,
    masked_sum_huber_and_count,
)


def regularization(model, pred=None, lambda_l1=0.0, lambda_l2=0.0, lambda_custom=0.0):
    """
    Compute optional regularisation terms.

    Supports L1 weight decay, L2 weight decay, and total-variation (TV)
    spatial smoothness on predictions. AdamW's built-in weight_decay already
    handles L2, so lambda_l2 should typically remain 0.
    """
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    regu = 0.0

    if lambda_l1 != 0.0:
        regu += lambda_l1 * sum(p.abs().sum() for p in base_model.parameters())

    if lambda_l2 != 0.0:
        regu += lambda_l2 * sum((p ** 2).sum() for p in base_model.parameters())

    if lambda_custom != 0.0:
        if pred is None:
            raise ValueError("pred must be provided for TV regularisation.")
        diff_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        diff_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        regu += lambda_custom * (diff_x.mean() + diff_y.mean())

    return regu

# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

def init_classifier_head(module):
    """Kaiming-initialise Conv2d layers and zero-initialise BN in a module."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias,   0)


def count_parameters(model):
    """Print the total number of trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")


def replace_bn_with_gn(module, num_groups=32):
    """Recursively replace BatchNorm2d with GroupNorm (useful for small batch sizes)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = min(num_groups, child.num_features)
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            replace_bn_with_gn(child, num_groups)


def build_deeplab_head(n_in=256, n_out=1, activation="linear"):
    """Build a single-conv segmentation head with optional output activation."""
    layers = [nn.Conv2d(n_in, n_out, kernel_size=1)]
    if activation == "softplus":
        layers.append(nn.Softplus(beta=1.0))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation != "linear":
        raise ValueError(f"Unknown activation: '{activation}'")
    return nn.Sequential(*layers)


def load_model(Model_mode, device, lr=1e-4, reduce_lr_patience=5, output_activation=None):
    """
    Initialise a DeepLabV3+ model with the specified encoder backbone.

    All backbones use the unified smp.DeepLabV3Plus interface with ImageNet
    pre-trained encoder weights. The segmentation head is re-initialised with
    Kaiming normal weights for better convergence on bathymetry data.

    Parameters
    ----------
    Model_mode         : str   – One of 'tu-convnext_large', 'efficientnet-b4',
                                 'resnet50', 'resnet101'.
    device             : torch.device
    lr                 : float – Initial learning rate.
    reduce_lr_patience : int   – Patience for ReduceLROnPlateau.
    output_activation  : str or None – Passed to smp as the head activation.

    Returns
    -------
    model, optimizer, scheduler, scaler
    """
    mode_map = {
        "efficientnet-b4":       "efficientnet-b4",
        "convnext":              "tu-convnext_large",
        "convnext_large":        "tu-convnext_large",
        "tu-convnext_large":     "tu-convnext_large",
        "resnet50":              "resnet50",
        "deeplabv3_resnet50":    "resnet50",
        "resnet101":             "resnet101",
        "deeplabv3_resnet101":   "resnet101",
    }
    encoder = mode_map.get(Model_mode.lower())
    if encoder is None:
        raise ValueError(f"Unknown Model_mode: '{Model_mode}'")

    print(f"Initialising DeepLabV3+ with encoder: {encoder}")
    model = smp.DeepLabV3Plus(
        encoder_name         = encoder,
        encoder_weights      = "imagenet",
        in_channels          = 12,
        classes              = 1,
        activation           = output_activation,
        encoder_output_stride = 16,
    ).to(device)

    init_classifier_head(model.segmentation_head)
    count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=reduce_lr_patience)
    scaler    = torch.cuda.amp.GradScaler(enabled=True)
    return model, optimizer, scheduler, scaler

# =============================================================================
# DATA HELPERS
# =============================================================================

def read_patch_tif(path):
    """
    Read a 14-band patch GeoTIFF into numpy arrays.

    Band layout (set by step1_splitting.py):
      Bands 1–12 : Sentinel-2 reflectance (float32)
      Band 13    : LiDAR depth in metres   (float32)
      Band 14    : Validity mask, 1=valid  (uint8)

    Returns
    -------
    X       : np.ndarray (12, H, W)
    y       : np.ndarray (H, W)
    mask    : np.ndarray (H, W), uint8
    profile : dict
    """
    with rasterio.open(path) as src:
        arr     = src.read().astype(np.float32)
        profile = src.profile

    if arr.shape[0] < 13:
        raise ValueError(f"{path}: fewer than 13 bands (got {arr.shape[0]}).")

    nbands = min(12, arr.shape[0] - 1)
    X      = arr[:nbands]
    if nbands < 12:
        # Pad missing bands with NaN to maintain shape (12, H, W)
        X_pad           = np.full((12, arr.shape[1], arr.shape[2]), np.nan, dtype=np.float32)
        X_pad[:nbands]  = X
        X               = X_pad

    y    = arr[12]  # Band 13 (0-indexed as 12): LiDAR depth
    mask = arr[13].astype(np.uint8) if arr.shape[0] >= 14 else (~np.isnan(y)).astype(np.uint8)
    return X, y, mask, profile


def save_prediction_geotiff(pred_arr, profile, out_path, tags=None):
    """Write a single-band float32 prediction array to a GeoTIFF."""
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(pred_arr.astype(np.float32), 1)
        if tags:
            dst.update_tags(**tags)


def filter_valid_patches(file_list, min_valid_ratio=0.1):
    """
    Return only those patch files whose valid-pixel fraction meets the threshold.

    Reads band 14 (validity mask) to compute the ratio. Falls back to band 13
    (depth) for files without a mask band.
    """
    valid_files = []
    print(f"Filtering {len(file_list)} patches (min_valid_ratio={min_valid_ratio})...")
    for f in file_list:
        try:
            with rasterio.open(f) as src:
                if src.count >= 14:
                    mask         = src.read(14)
                    valid_pixels = np.count_nonzero(mask)
                else:
                    depth        = src.read(13)
                    valid_pixels = np.count_nonzero(~np.isnan(depth))
                if valid_pixels / mask.size >= min_valid_ratio:
                    valid_files.append(f)
        except Exception:
            pass  # Skip corrupt/unreadable files silently
    print(f"Retained {len(valid_files)} / {len(file_list)} patches.")
    return valid_files


def compute_global_stats_from_files(file_list):
    """Compute global depth min/max and NaN fraction across all training patches."""
    y_min, y_max     = float('inf'), float('-inf')
    total_pix, nan_pix = 0, 0
    for f in file_list:
        try:
            with rasterio.open(f) as src:
                if src.count < 13:
                    continue
        except Exception:
            continue
        _, y, _, _ = read_patch_tif(f)
        total_pix += y.size
        nan_pix   += np.isnan(y).sum()
        if np.any(~np.isnan(y)):
            y_min = min(y_min, float(np.nanmin(y)))
            y_max = max(y_max, float(np.nanmax(y)))

    nan_pct = nan_pix / total_pix * 100 if total_pix > 0 else 0
    return (np.nan if y_min == float('inf') else y_min,
            np.nan if y_max == float('-inf') else y_max,
            nan_pct, nan_pix, total_pix)


def calculate_binary_metrics(y_true, y_pred, threshold):
    """
    Compute binary classification metrics using depth threshold to define 'deep' (positive).

    Pixels with |depth| > threshold are classified as Positive.
    NaN values in y_true are excluded.

    Returns
    -------
    dict with keys: TP, TN, FP, FN, Precision, Recall, F1_Score, Accuracy
    """
    valid = ~np.isnan(y_true)
    y_t   = y_true[valid]
    y_p   = y_pred[valid]

    pos_true = np.abs(y_t) > threshold
    pos_pred = np.abs(y_p) > threshold

    TP = np.sum( pos_true &  pos_pred)
    TN = np.sum(~pos_true & ~pos_pred)
    FP = np.sum(~pos_true &  pos_pred)
    FN = np.sum( pos_true & ~pos_pred)

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    return {
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
        'Precision': precision, 'Recall': recall, 'F1_Score': f1, 'Accuracy': accuracy
    }

# =============================================================================
# DATASET
# =============================================================================

class BathyDataset(Dataset):
    """
    PyTorch Dataset for bathymetry patch GeoTIFFs.

    Loads patches on-the-fly (lazy loading) to avoid loading the entire dataset
    into memory. Applies depth thresholding and optional band scaling.

    Each item returned: (X_tensor, y_tensor, mask_tensor)
      X_tensor    : (12, H, W) float32
      y_tensor    : (1, H, W)  float32 – NaN depths replaced with 0 (mask handles exclusion)
      mask_tensor : (1, H, W)  float32 – 1 for pixels used in loss, 0 otherwise
    """

    def __init__(self, file_list, params, is_train=False):
        self.file_list       = file_list
        self.params          = params
        self.depth_threshold = abs(params["depth_threshold"])
        self.depth_type      = params["depth_type"]
        self.scaling_method  = params["scaling_method"]
        self.bmins = self.bmaxs = None

        if self.scaling_method == "minmax_manual":
            bmin_path = os.path.join(params["LiDAR_Model"], "bmin.npy")
            bmax_path = os.path.join(params["LiDAR_Model"], "bmax.npy")
            if os.path.exists(bmin_path) and os.path.exists(bmax_path):
                self.bmins = np.load(bmin_path)
                self.bmaxs = np.load(bmax_path)
            else:
                print(f"[WARN] bmin/bmax not found in {params['LiDAR_Model']}; MinMax scaling skipped.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            X, y, mask, _ = read_patch_tif(file_path)
        except Exception as e:
            print(f"[ERROR] Cannot read {file_path}: {e}. Returning zeros.")
            H, W = 256, 256
            return torch.zeros(12, H, W), torch.zeros(1, H, W), torch.zeros(1, H, W)

        # Exclude pixels outside the target depth range
        if   self.depth_type == 'shallow': condition = np.abs(y) <  self.depth_threshold
        elif self.depth_type == 'deep':    condition = np.abs(y) >  self.depth_threshold
        else:                              condition = np.zeros_like(y, dtype=bool)
        mask[condition] = 0

        # Band scaling
        if self.scaling_method == "divide_10000":
            X = X / 10000.0
        elif self.scaling_method == "minmax_manual" and self.bmins is not None:
            for b in range(X.shape[0]):
                rng    = self.bmaxs[b] - self.bmins[b] + 1e-8
                X[b]  = (X[b] - self.bmins[b]) / rng
        # Note: "Log_band" is pre-applied in step1_splitting.py; no action needed here.

        # Replace NaN/Inf with 0 before tensor conversion (mask ensures they are ignored in loss)
        X      = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_fill = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(y_fill).unsqueeze(0).float(),
            torch.from_numpy(mask).unsqueeze(0).float()
        )

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    log_params(params, LiDAR_Model)

    # --- Device setup ---
    if torch.cuda.is_available():
        device      = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        gpu_count   = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device      = torch.device("mps")
        device_name = "Apple MPS"
        gpu_count   = 0
    else:
        device      = torch.device("cpu")
        device_name = "CPU"
        gpu_count   = 0

    print(f"Device: {device} ({device_name}) | GPUs: {gpu_count}")
    if gpu_count > 1:
        print(f"Using DataParallel across {gpu_count} GPUs.")

    # -------------------------------------------------------------------------
    # 1. Discover and filter patch files
    # -------------------------------------------------------------------------
    train_dir = os.path.join(TrainingDataPath, params["train_source"])
    val_dir   = os.path.join(TrainingDataPath, params["valid_source"])
    test_dir  = os.path.join(TrainingDataPath, params["test_source"])

    train_files_all = sorted(glob.glob(os.path.join(train_dir, "*.tif")))
    val_files_all   = sorted(glob.glob(os.path.join(val_dir,   "*.tif")))
    test_files_all  = sorted(glob.glob(os.path.join(test_dir,  "*.tif")))

    if not train_files_all:
        raise SystemExit(f"[ERROR] No training patches found in: {train_dir}")

    train_files = filter_valid_patches(train_files_all, params["min_valid_ratio"])         if params["min_valid_ratio"]         else train_files_all
    val_files   = filter_valid_patches(val_files_all,   params["min_valid_ratio_for_val"]) if params["min_valid_ratio_for_val"] else val_files_all
    test_files  = test_files_all

    # -------------------------------------------------------------------------
    # 2. Per-band MinMax scaling initialisation (if requested)
    # -------------------------------------------------------------------------
    if params["scaling_method"] == "minmax_manual":
        bmin_path = os.path.join(LiDAR_Model, "bmin.npy")
        bmax_path = os.path.join(LiDAR_Model, "bmax.npy")
        if not (os.path.exists(bmin_path) and os.path.exists(bmax_path)):
            print("Computing per-band Min/Max over training set...")
            X0, _, _, _ = read_patch_tif(train_files[0])
            n_bands     = X0.shape[0]
            bmins       = np.full(n_bands, float('inf'))
            bmaxs       = np.full(n_bands, float('-inf'))
            for f in train_files:
                X, _, _, _ = read_patch_tif(f)
                for b in range(n_bands):
                    bv = X[b]
                    if np.any(~np.isnan(bv)):
                        bmins[b] = min(bmins[b], float(np.nanmin(bv)))
                        bmaxs[b] = max(bmaxs[b], float(np.nanmax(bv)))
            bmins[bmins == float('inf')]  = 0.0
            bmaxs[bmaxs == float('-inf')] = 1.0
            np.save(bmin_path, bmins)
            np.save(bmax_path, bmaxs)
            print(f"Saved bmin/bmax to {LiDAR_Model}.")
        else:
            print(f"Re-using existing bmin/bmax from {LiDAR_Model}.")

    # -------------------------------------------------------------------------
    # 3. DataLoaders
    # -------------------------------------------------------------------------
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds     = BathyDataset(train_files, params, is_train=True)
    val_ds       = BathyDataset(val_files,   params, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True,  num_workers=params["num_workers"], generator=g, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"], pin_memory=True)
    print("DataLoaders ready.")

    # -------------------------------------------------------------------------
    # 4. Model, optimiser, scheduler
    # -------------------------------------------------------------------------
    model, optimizer, scheduler, scaler = load_model(
        params["Model_mode"], device,
        lr=params["lr"],
        reduce_lr_patience=params["reduce_lr_patience"],
        output_activation=(None if params["output_activation"] == "none" else params["output_activation"])
    )
    if gpu_count > 1:
        model = nn.DataParallel(model)

    # -------------------------------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------------------------------
    csv_log_file = os.path.join(LiDAR_Model, "training_metrics.csv")
    fieldnames   = [
        "epoch",
        "train_RMSE_m", "train_MAE_m", "train_SWF_m", "train_RPE_%",
        "val_RMSE_m",   "val_MAE_m",   "val_SWF_m",   "val_RPE_%",
        "skipped_train", "skipped_val", "LR"
    ]

    best_val_loss      = float('inf')
    early_stop_counter = 0

    with open(csv_log_file, "w", newline="") as f_metrics:
        writer = csv.DictWriter(f_metrics, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, params["num_epochs"] + 1):
            model.train()
            sse_total = mae_total = rpe_total = swf_wse_total = 0.0
            valid_pixels          = 0
            train_skipped_batches = 0

            for xb, yb, mb in train_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                optimizer.zero_grad()

                out = model(xb)
                # Unwrap dict output (torchvision DeepLabV3 returns {'out': tensor})
                if isinstance(out, dict):
                    out = out['out']

                # --- Primary loss for back-propagation ---
                loss_fns = {
                    "rpe":   lambda: masked_sum_rpe_and_count(out, yb, mb),
                    "mae":   lambda: masked_sum_abs_and_count(out, yb, mb),
                    "huber": lambda: masked_sum_huber_and_count(out, yb, mb, delta=1.0),
                    "swf":   lambda: masked_sum_swf_and_count(out, yb, mb, params["SWF_beta"], params["SWF_Z0"]),
                }
                loss_sum, n_val = loss_fns.get(params["loss_type"],
                                               lambda: masked_sum_sse_and_count(out, yb, mb))()

                if n_val < 1e-6:
                    train_skipped_batches += 1
                    continue

                reg_term  = regularization(model, out, params["lambda_l1"], params["lambda_l2"], params["lambda_custom"])
                loss      = loss_sum / (n_val + 1e-8) + reg_term
                loss.backward()

                # Gradient clipping
                params_to_clip = model.module.parameters() if gpu_count > 1 else model.parameters()
                torch.nn.utils.clip_grad_norm_(params_to_clip, params["grad_clip_norm"])
                optimizer.step()

                # --- Accumulate all metrics for logging ---
                sse_b,     n_b = masked_sum_sse_and_count(out, yb, mb)
                mae_b,     _   = masked_sum_abs_and_count(out, yb, mb)
                rpe_b,     _   = masked_sum_rpe_and_count(out, yb, mb)
                swf_wse_b, _   = masked_sum_swf_and_count(out, yb, mb, params["SWF_beta"], params["SWF_Z0"])

                sse_total     += sse_b.item()
                mae_total     += mae_b.item()
                rpe_total     += rpe_b.item()
                swf_wse_total += swf_wse_b.item()
                valid_pixels  += n_b.item()

            # Train statistics for this epoch
            vp = valid_pixels
            train_rmse     = (sse_total     / vp) ** 0.5 if vp > 0 else float("nan")
            train_mae      = (mae_total     / vp)        if vp > 0 else float("nan")
            train_rpe_avg  = (rpe_total     / vp) * 100  if vp > 0 else float("nan")
            train_swf_rmse = (swf_wse_total / vp) ** 0.5 if vp > 0 else float("nan")

            # --- Validation ---
            model.eval()
            val_sse = val_mae = val_rpe = val_swf_wse = 0.0
            val_pix = val_skipped_batches = 0

            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                with torch.no_grad():
                    out = model(xb)
                    if isinstance(out, dict):
                        out = out['out']

                if params["clamp_output"]:
                    out = torch.clamp(out, -100.0, 0.0)

                s, n   = masked_sum_sse_and_count(out, yb, mb)
                m, _   = masked_sum_abs_and_count(out, yb, mb)
                r, _   = masked_sum_rpe_and_count(out, yb, mb)
                sw, _  = masked_sum_swf_and_count(out, yb, mb, params["SWF_beta"], params["SWF_Z0"])

                if n.item() < 1e-6:
                    val_skipped_batches += 1
                    continue

                val_sse     += s.item()
                val_mae     += m.item()
                val_rpe     += r.item()
                val_swf_wse += sw.item()
                val_pix     += n.item()

            vp2 = val_pix
            val_rmse     = (val_sse     / vp2) ** 0.5 if vp2 > 0 else float("nan")
            val_mae_avg  = (val_mae     / vp2)        if vp2 > 0 else float("nan")
            val_rpe_avg  = (val_rpe     / vp2) * 100  if vp2 > 0 else float("nan")
            val_swf_rmse = (val_swf_wse / vp2) ** 0.5 if vp2 > 0 else float("nan")

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:03d}: "
                f"Train RMSE={train_rmse:.2f}m MAE={train_mae:.2f}m SWF={train_swf_rmse:.2f}m RPE={train_rpe_avg:.2f}% | "
                f"Val   RMSE={val_rmse:.2f}m MAE={val_mae_avg:.2f}m SWF={val_swf_rmse:.2f}m RPE={val_rpe_avg:.2f}% | "
                f"LR={current_lr:.2e} | Skipped train={train_skipped_batches} val={val_skipped_batches}"
            )

            writer.writerow({
                "epoch": epoch,
                "train_RMSE_m": train_rmse, "train_MAE_m": train_mae,
                "train_SWF_m":  train_swf_rmse, "train_RPE_%": train_rpe_avg,
                "val_RMSE_m":   val_rmse,   "val_MAE_m":   val_mae_avg,
                "val_SWF_m":    val_swf_rmse,   "val_RPE_%":   val_rpe_avg,
                "skipped_train": train_skipped_batches,
                "skipped_val":   val_skipped_batches,
                "LR": current_lr
            })
            f_metrics.flush()

            # --- Scheduler step & early stopping ---
            monitor_map = {
                "swf": (val_swf_rmse, "SWF RMSE"),
                "rpe": (val_rpe_avg,  "RPE %"),
                "mae": (val_mae_avg,  "MAE"),
            }
            current_val_metric, metric_name = monitor_map.get(
                params.get("loss_type", "sse").lower(), (val_rmse, "RMSE")
            )
            scheduler.step(current_val_metric)

            if current_val_metric < best_val_loss:
                best_val_loss      = current_val_metric
                early_stop_counter = 0
                state = model.module.state_dict() if gpu_count > 1 else model.state_dict()
                torch.save(state, os.path.join(LiDAR_Model, "best_model.pth"))
                print(f"  --> New best {metric_name}: {current_val_metric:.4f} — model saved.")
            else:
                early_stop_counter += 1
                print(f"  --> No improvement ({metric_name}) for {early_stop_counter}/{params['earlystop_patience']} epochs.")

            if early_stop_counter >= params["earlystop_patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    final_epoch = epoch

    # -------------------------------------------------------------------------
    # 6. Save final model and metadata
    # -------------------------------------------------------------------------
    final_model_path = os.path.join(LiDAR_Model, "deeplabv3_bathy_final.pth")
    state = model.module.state_dict() if gpu_count > 1 else model.state_dict()
    torch.save(state, final_model_path)

    meta = {
        "Model_mode":                    params["Model_mode"],
        "depth_scale":                   params["depth_scale"],
        "min_valid_ratio":               params["min_valid_ratio"],
        "grad_clip_norm":                params["grad_clip_norm"],
        "output_activation":             params["output_activation"],
        "clamp_output":                  params["clamp_output"],
        "num_epochs_trained":            final_epoch,
        "batch_size":                    params["batch_size"],
        "lr":                            params["lr"],
        "depth_threshold":               params["depth_threshold"],
        "depth_type":                    params["depth_type"],
        "loss_type":                     params["loss_type"],
        "SWF_beta":                      params["SWF_beta"],
        "SWF_Z0":                        params["SWF_Z0"],
        "scaling_method":                params["scaling_method"],
        "train_skipped_final_epoch":     int(train_skipped_batches),
        "val_skipped_final_epoch":       int(val_skipped_batches),
        "best_val_loss":                 float(best_val_loss) if best_val_loss != float('inf') else "inf",
        "saved_best_model":              os.path.join(LiDAR_Model, "best_model.pth"),
        "gpu_count":                     gpu_count,
    }
    with open(os.path.join(LiDAR_Model, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete. Final model: {final_model_path}")

    # -------------------------------------------------------------------------
    # 7. Training curves
    # -------------------------------------------------------------------------
    log = pd.read_csv(csv_log_file).replace("nan", np.nan)
    for col in ["train_RMSE_m", "train_MAE_m", "train_SWF_m", "train_RPE_%",
                "val_RMSE_m",   "val_MAE_m",   "val_SWF_m",   "val_RPE_%"]:
        log[col] = pd.to_numeric(log[col], errors='coerce')

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    def _plot(ax, col_train, col_val, ylabel, title, color):
        ax.plot(log["epoch"], log[col_train], label=f"Train {ylabel}", color=color, marker='o')
        if col_val in log and not log[col_val].isna().all():
            ax.plot(log["epoch"], log[col_val], label=f"Val {ylabel}", color=color, linestyle='--', marker='x')
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True)

    _plot(axes[0, 0], "train_RMSE_m", "val_RMSE_m", "RMSE [m]",     "RMSE",         'b')
    _plot(axes[0, 1], "train_MAE_m",  "val_MAE_m",  "MAE [m]",      "MAE",          'g')
    _plot(axes[1, 0], "train_SWF_m",  "val_SWF_m",  "SWF RMSE [m]", "SWF RMSE",  'purple')
    _plot(axes[1, 1], "train_RPE_%",  "val_RPE_%",  "RPE [%]",      "RPE",          'r')
    axes[2, 0].plot(log["epoch"], log["LR"], color='k', marker='o')
    axes[2, 0].set_xlabel("Epoch"); axes[2, 0].set_ylabel("LR"); axes[2, 0].set_title("Learning Rate"); axes[2, 0].grid(True)
    axes[2, 1].axis('off')

    plt.tight_layout()
    curves_path = os.path.join(LiDAR_Model, "training_curves.png")
    plt.savefig(curves_path, dpi=dpi)
    plt.close()
    print(f"Training curves saved: {curves_path}")

    # -------------------------------------------------------------------------
    # 8. Inference on test set
    # -------------------------------------------------------------------------
    print("\nRunning inference on test set...")

    if not test_files:
        print("[WARN] No test files found.")
    else:
        # Load best model (single-GPU wrapper for inference)
        state_dict = torch.load(os.path.join(LiDAR_Model, "best_model.pth"))
        if gpu_count > 1:
            test_model, _, _, _ = load_model(
                params["Model_mode"], torch.device("cuda:0"),
                lr=params["lr"], reduce_lr_patience=params["reduce_lr_patience"],
                output_activation=(None if params["output_activation"] == "none" else params["output_activation"])
            )
        else:
            test_model = model
        test_model.load_state_dict(state_dict)
        test_model = test_model.to(device)
        test_model.eval()

        preds_folder = os.path.join(LiDAR_Model, "predictions")
        os.makedirs(preds_folder, exist_ok=True)

        results          = []    # Full-range per-patch results
        within3m_results = []    # Within-3m per-patch results
        all_depths_gt, all_depths_pred       = [], []
        all_3m_depths_gt, all_3m_depths_pred = [], []
        all_swf_wse, all_3m_swf_wse          = [], []

        if params["scaling_method"] == "minmax_manual":
            bmins = np.load(os.path.join(LiDAR_Model, "bmin.npy"))
            bmaxs = np.load(os.path.join(LiDAR_Model, "bmax.npy"))

        def _make_viz(gt_arr, pred_arr, v_mask, base, suffix, title_str):
            """Create and save a 3-panel depth / prediction / error plot."""
            gt_m   = np.where(v_mask, gt_arr,   np.nan)
            pr_m   = np.where(v_mask, pred_arr, np.nan)
            err_m  = np.where(v_mask, pred_arr - gt_arr, np.nan)
            v_min  = np.nanmin([np.nanmin(gt_m),  np.nanmin(pr_m)])
            v_max  = np.nanmax([np.nanmax(gt_m),  np.nanmax(pr_m)])
            e_max  = max(abs(np.nanmax(err_m)) if not np.all(np.isnan(err_m)) else 1.0, 1.0)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.subplots_adjust(left=0.1, right=0.9, wspace=0.1)
            im0 = axes[0].imshow(gt_m,  cmap='jet', vmin=v_min, vmax=v_max); axes[0].set_title("Ground Truth Depth (m)");       axes[0].axis('off')
            im1 = axes[1].imshow(pr_m,  cmap='jet', vmin=v_min, vmax=v_max); axes[1].set_title("Predicted Depth (m)");          axes[1].axis('off')
            im2 = axes[2].imshow(err_m, cmap='coolwarm', vmin=-e_max, vmax=e_max); axes[2].set_title("Prediction − GT (m)");    axes[2].axis('off')
            fig.add_axes([0.05, 0.1, 0.02, 0.8]); plt.colorbar(im0, cax=fig.axes[-1], orientation='vertical')
            fig.add_axes([0.93, 0.1, 0.02, 0.8]); plt.colorbar(im2, cax=fig.axes[-1], orientation='vertical')
            fig.suptitle(title_str, fontsize=14)
            plt.savefig(os.path.join(preds_folder, f"{base}{suffix}.png"), dpi=dpi, bbox_inches='tight')
            plt.close()

        for tfile in test_files:
            X_in, y_patch, mask_patch, profile = read_patch_tif(tfile)
            base = os.path.splitext(os.path.basename(tfile))[0]

            # Apply depth threshold
            if   params["depth_type"] == 'shallow': cond = np.abs(y_patch) <  params["depth_threshold"]
            elif params["depth_type"] == 'deep':    cond = np.abs(y_patch) >  params["depth_threshold"]
            else:                                    cond = np.zeros_like(y_patch, dtype=bool)
            cond         |= (y_patch < -20) | (y_patch > -0.01)
            outside_3m    = (y_patch < -3.0) | (y_patch > -0.01) | np.isnan(y_patch)
            mask_patch[cond] = 0
            y_patch[cond]    = np.nan

            y_patch3m    = y_patch.copy()
            mask_patch3m = mask_patch.copy()
            mask_patch3m[outside_3m] = 0
            y_patch3m[outside_3m]    = np.nan

            # Input scaling
            if   params["scaling_method"] == "divide_10000":    X_in = X_in / 10000.0
            elif params["scaling_method"] == "minmax_manual":
                for b in range(12):
                    X_in[b] = (X_in[b] - bmins[b]) / (bmaxs[b] - bmins[b] + 1e-8)

            X_t = torch.from_numpy(np.nan_to_num(X_in)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                pred_out = test_model(X_t)
                if isinstance(pred_out, dict):
                    pred_out = pred_out['out']

            pred_np = pred_out.cpu().numpy()[0, 0]
            if params["clamp_output"]:
                pred_np = np.clip(pred_np, -100.0, 0.0)

            save_prediction_geotiff(pred_np, profile, os.path.join(preds_folder, f"{base}_pred.tif"))

            def _record_metrics(y_arr, mask_arr, label, results_list, gt_list, pred_list, swf_list, y_t, m_t):
                """Compute and record per-patch metrics."""
                vm = mask_arr.astype(bool) & np.isfinite(pred_np) & np.isfinite(y_arr)
                if not vm.any():
                    print(f"{base}: 0 valid pixels ({label}) – skipped.")
                    return
                p_val, g_val = pred_np[vm], y_arr[vm]
                rmse  = np.sqrt(np.nanmean((p_val - g_val) ** 2))
                mae   = np.nanmean(np.abs(p_val - g_val))
                rpe   = np.nanmean(np.abs(p_val - g_val) / (np.abs(g_val) + 1e-6)) * 100
                swf_w, _ = masked_sum_swf_and_count(pred_out, y_t, m_t, params["SWF_beta"], params["SWF_Z0"])
                swf_r    = (swf_w.item() / vm.sum()) ** 0.5
                print(f"{base} [{label}]: RMSE={rmse:.3f}m MAE={mae:.3f}m SWF={swf_r:.3f}m RPE={rpe:.2f}%")
                results_list.append({"File": base, "RMSE_m": rmse, "MAE_m": mae, "SWF_m": swf_r, "RPE_%": rpe,
                                     "Valid_pixels": vm.sum(), "GT_min_depth_m": g_val.min(), "GT_max_depth_m": g_val.max(),
                                     "Pred_min_depth_m": p_val.min(), "Pred_max_depth_m": p_val.max()})
                gt_list.append(g_val); pred_list.append(p_val); swf_list.append(swf_w.item())
                _make_viz(y_arr, pred_np, vm, base, f"_{label}",
                          f"{base} | RMSE={rmse:.2f}m MAE={mae:.2f}m RPE={rpe:.2f}%")

            y_t_full = torch.from_numpy(np.nan_to_num(y_patch)).unsqueeze(0).unsqueeze(0).float().to(device)
            m_t_full = torch.from_numpy(mask_patch).unsqueeze(0).unsqueeze(0).float().to(device)
            y_t_3m   = torch.from_numpy(np.nan_to_num(y_patch3m)).unsqueeze(0).unsqueeze(0).float().to(device)
            m_t_3m   = torch.from_numpy(mask_patch3m).unsqueeze(0).unsqueeze(0).float().to(device)

            _record_metrics(y_patch3m, mask_patch3m, "within3m", within3m_results, all_3m_depths_gt, all_3m_depths_pred, all_3m_swf_wse, y_t_3m, m_t_3m)
            _record_metrics(y_patch,   mask_patch,   "full",     results,           all_depths_gt,    all_depths_pred,    all_swf_wse,    y_t_full, m_t_full)

        def _summarise_and_save(results_list, gt_all, pred_all, swf_all, csv_name, label):
            """Print overall test statistics and save results CSV."""
            if not results_list:
                print(f"No valid test results for [{label}].")
                return
            gt_all   = np.concatenate(gt_all)
            pred_all = np.concatenate(pred_all)
            n_pix    = len(gt_all)
            ov_rmse  = np.sqrt(np.nanmean((pred_all - gt_all) ** 2))
            ov_mae   = np.nanmean(np.abs(pred_all - gt_all))
            ov_rpe   = np.nanmean(np.abs(pred_all - gt_all) / (np.abs(gt_all) + 1e-6)) * 100
            ov_swf   = np.sqrt(sum(swf_all) / n_pix)
            cm       = calculate_binary_metrics(gt_all, pred_all, params["depth_threshold"])
            print(f"\n===== Overall [{label}] =====")
            print(f"RMSE={ov_rmse:.3f}m  MAE={ov_mae:.3f}m  SWF={ov_swf:.3f}m  RPE={ov_rpe:.2f}%")
            print(f"Precision={cm['Precision']:.3f}  Recall={cm['Recall']:.3f}  F1={cm['F1_Score']:.3f}  Acc={cm['Accuracy']:.3f}")
            results_list.append({
                "File": "OVERALL", "RMSE_m": ov_rmse, "MAE_m": ov_mae, "SWF_m": ov_swf, "RPE_%": ov_rpe,
                "Valid_pixels": n_pix, "GT_min_depth_m": gt_all.min(), "GT_max_depth_m": gt_all.max(),
                "Pred_min_depth_m": pred_all.min(), "Pred_max_depth_m": pred_all.max(),
                "Classification_Precision": cm['Precision'], "Classification_Recall": cm['Recall'],
                "Classification_F1_Score": cm['F1_Score'],  "Classification_Accuracy": cm['Accuracy'],
                "Classification_TP": cm['TP'], "Classification_TN": cm['TN'],
                "Classification_FP": cm['FP'], "Classification_FN": cm['FN'],
            })
            pd.DataFrame(results_list).to_csv(os.path.join(preds_folder, csv_name), index=False, float_format="%.4f")

            # Scatter plot: predicted vs ground truth
            sample = np.random.choice(n_pix, min(n_pix, 100_000), replace=False)
            lims   = [min(gt_all.min(), pred_all.min()) - 1, max(gt_all.max(), pred_all.max()) + 1]
            plt.figure(figsize=(8, 8))
            plt.scatter(gt_all[sample], pred_all[sample], s=5, c="royalblue", alpha=0.4)
            plt.plot(lims, lims, "k--", lw=2)
            plt.xlim(lims); plt.ylim(lims)
            plt.xlabel("Ground Truth Depth (m)"); plt.ylabel("Predicted Depth (m)")
            plt.title(f"[{label}] Pred vs GT — RMSE={ov_rmse:.2f}m  MAE={ov_mae:.2f}m")
            plt.grid(True)
            plt.savefig(os.path.join(preds_folder, f"Scatter_{label}.png"), dpi=dpi)
            plt.close()

        _summarise_and_save(within3m_results, all_3m_depths_gt, all_3m_depths_pred, all_3m_swf_wse,
                            "within3m_prediction_metrics.csv", "within3m")
        _summarise_and_save(results, all_depths_gt, all_depths_pred, all_swf_wse,
                            "prediction_metrics.csv", "full")

    # -------------------------------------------------------------------------
    # 9. Elapsed time
    # -------------------------------------------------------------------------
    elapsed          = timer() - startime
    days, rem        = divmod(elapsed, 86400)
    hours, rem       = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal elapsed time: {int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("Done.")
