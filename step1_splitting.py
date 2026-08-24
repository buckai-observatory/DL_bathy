# -*- coding: utf-8 -*-
"""
step1_patch_extraction.py
=========================
Prepares satellite-derived bathymetry training data by:
  - Fetching Sentinel-2 L2A scenes (with SCL cloud/water mask) over the
    AOI + time window via the geoai-datacubes package -- OR reading
    pre-downloaded scenes from disk (`DATA_SOURCE = 'local_scenes'`)
  - Aligning Sentinel-2 imagery with LiDAR ground truth
  - Deriving cloud/water masks from the SCL band (class 6 = water)
  - Generating tide correction grids using the DTU23 tidal model
  - Extracting, splitting, (optionally) balancing, and augmenting image patches
  - Saving patches as GeoTIFFs with 14 bands:
      Bands 1-12 : Sentinel-2 reflectance
      Band 13    : LiDAR depth (metres)
      Band 14    : Validity mask (1=valid, 0=invalid)

Earlier versions of this script fetched the cloud/water mask via Google
Earth Engine and required users to pre-download the Sentinel-2 imagery
separately. The current default drops both dependencies; the historical
GEE code path lives in git history if needed.

Splitting behaviour
-------------------
Set train_ratio and val_ratio to control the split.
Setting both to 0 puts ALL patches into the test set (useful for inference-only
datasets where no model training is required from this site).

Output directory structure
--------------------------
LiDAR_Model/
  train/           val/           test/
  train_balanced/  val_balanced/  test_balanced/   (only if apply_balance=True)
  train_aug/       val_aug/       test_aug/        (only if apply_augmentation=True)
  *_Patches_Mask_split.jpg        split overlay visualisation
  cloud_removal_counts.csv
  console_output_step1.log        (optional; enable in USER CONFIGURATION)
"""

from timeit import default_timer as timer
import os
import glob
import re
import random
import platform
import subprocess
from datetime import datetime, timedelta
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from osgeo import gdal
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from skimage.transform import resize
import sys

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# --- Sentinel-2 processing level ---
Sentinel2_level = 'L2A'   # 'L1C' or 'L2A' (geoai-datacubes path is L2A-only)

# --- Data acquisition source -------------------------------------------------
# 'geoai_datacubes' (default): auto-fetch Sentinel-2 L2A scenes over the AOI
#                              and time window using the geoai-datacubes
#                              package. No credentials, no manual downloads.
#                              Requires: pip install geoai-datacubes
#
# 'local_scenes':              bring-your-own pre-downloaded scenes. Fast-
#                              start path for reviewers who already have a
#                              set of matched S2 L2A scenes on disk (e.g.
#                              from an earlier geoai-datacubes run or a
#                              custom downloader). Each scene must be a
#                              13-band GeoTIFF with:
#                                Bands 1-12 = B01, B02, B03, B04, B05, B06,
#                                             B07, B08, B8A, B09, B11, B12
#                                Band 13    = SCL (L2A Scene Classification)
#                              This matches what geoai-datacubes writes by
#                              default; other download tools that omit SCL
#                              are not supported by local_scenes mode.
DATA_SOURCE = 'geoai_datacubes'

# --- AOI + time range (used only when DATA_SOURCE == 'geoai_datacubes') -----
# WGS84 bounding box: (lon_min, lat_min, lon_max, lat_max).
# Default: Great Barrier Reef northern site (S2 tile T55LCE) matching the
# ground-truth GeoTIFFs shipped in `Ground_Truth.zip`.
AOI                = [145.14, -14.56, 145.66, -14.26]
TIME_RANGE         = ('2020-06-01', '2020-09-30')   # austral dry season 2020
MAX_CLOUD_COVERAGE = 0.20                            # scene-level cloud filter
FETCH_RESOLUTION_M = 10                              # matches S2 native 10 m

# --- Pre-downloaded scenes (used only when DATA_SOURCE == 'local_scenes') ---
# Directory or glob pointing at 13-band L2A GeoTIFFs (see contract above).
LOCAL_SCENES_DIR = './sentinel_images_L2A/'

# --- LiDAR reference file ---
# Path to the co-registered LiDAR GeoTIFF used as ground truth.
# Update this to point to your own LiDAR file.
script_dir = os.path.dirname(os.path.abspath(__file__))
lidar_filename = 'your_lidar_reference.tif'   # <-- update to your LiDAR filename

# --- Output directory ---
LiDAR_Model = os.path.join(script_dir, "Data")
os.makedirs(LiDAR_Model, exist_ok=True)

# --- Optional console logging ---
# Uncomment the block below to redirect all console output to a log file.
# log_path = os.path.join(LiDAR_Model, "console_output_step1.log")
# sys.stdout = open(log_path, "w", buffering=1)
# sys.stderr = sys.stdout

# --- DTU23 tidal model paths ---
# Update these to match your DTU23 installation.
DTU23_WIN_PATH   = r"C:\path\to\DTU23"          # Windows: directory containing run_perth_new
DTU23_LINUX_PATH = "/path/to/DTU23/SOFTWARE"    # Linux / HPC equivalent

# --- Train / Validation / Test split ratios ---
# Set both to 0 to assign ALL patches to the test set (inference-only mode).
train_ratio = 1.0   # Fraction for training   (0.0–1.0)
val_ratio   = 0.0   # Fraction for validation  (0.0–1.0)
# Test fraction is implicit: 1.0 - train_ratio - val_ratio

# --- Patch settings ---
patch_size   = 512   # Square patch side length (pixels)
stride_value = 256   # Sliding-window stride (pixels)

# --- Depth / scene settings ---
rough_mean_sea_surface = 0     # Mean sea surface offset to subtract from LiDAR (metres)
deepestDepth2Train     = -30   # Deepest depth to include in training (metres)

# --- Patch validity filtering ---
valid_pixel_threshold = 0.1    # Min fraction of valid pixels required to keep a patch
strict_nan_filter     = False  # If True, discard patches with any NaN in the LiDAR target

# --- Splitting strategy ---
# 'consistent_spatial' : deterministic spatial hash → prevents data leakage (recommended)
# 'per_image'          : random split per scene
split_strategy = 'consistent_spatial'

# --- Balancing and augmentation ---
apply_balance      = False  # Depth-bin balancing (hybrid over/under-sampling)
apply_augmentation = True   # Rotation and flip augmentation

augmentation_rotations = [90, 180, 270]
augmentation_flip      = True

# --- Reflectance normalisation ---
# Logcube=True  → X = log(X / 10000 + ε)   (recommended for Sentinel-2)
# Logcube=False → X = X (no scaling applied)
Logcube           = True
band_normalization = 10000
Epsilon           = 1e-6   # Prevents log(0)

# =============================================================================
# INITIALISATION
# =============================================================================

print(f"Patch size    : {patch_size}")
print(f"Stride        : {stride_value}")
print(f"Train ratio   : {train_ratio}")
print(f"Val ratio     : {val_ratio}")
print(f"Scaling       : {'log(band/10000)' if Logcube else 'none (original values)'}")

# Output sub-directories
train_dir = os.path.join(LiDAR_Model, "train")
val_dir   = os.path.join(LiDAR_Model, "valid")
test_dir  = os.path.join(LiDAR_Model, "test")
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

train_balanced_dir = os.path.join(LiDAR_Model, "train_balanced")
val_balanced_dir   = os.path.join(LiDAR_Model, "valid_balanced")
test_balanced_dir  = os.path.join(LiDAR_Model, "test_balanced")
train_aug_dir      = os.path.join(LiDAR_Model, "train_aug")
val_aug_dir        = os.path.join(LiDAR_Model, "valid_aug")
test_aug_dir       = os.path.join(LiDAR_Model, "test_aug")

dpi = 300
random.seed(42)
startime = timer()

# --- Load LiDAR reference ---
lidar_file = os.path.join(script_dir, lidar_filename)
with rasterio.open(lidar_file) as lidar_data:
    lidar_grid      = lidar_data.read(1).astype(np.float32)
    nodata_value    = lidar_data.nodata
    lidar_transform = lidar_data.transform
    lidar_crs       = lidar_data.crs

if nodata_value is not None:
    lidar_grid[lidar_grid == nodata_value] = np.nan
lidar_grid -= rough_mean_sea_surface

# --- Sentinel-2 acquisition helpers ------------------------------------------
# Two helpers replace the old GEE fetch_water_mask() flow:
#   * _fetch_scenes_via_geoai_datacubes: pulls every S2 L2A scene over the
#     AOI + time_range through the geoai-datacubes package. Each returned
#     file is a 13-band GeoTIFF: bands 1-12 = spectral, band 13 = SCL.
#   * _scl_water_mask_from_scene: reads the SCL band (13) from a scene and
#     returns a binary non-water mask (0 = water pixel, 1 = everything
#     else). Deliberately mirrors the semantics of the old GEE call so the
#     downstream masking code did not need to change.

_S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
              "B8A", "B09", "B11", "B12", "SCL"]


def _fetch_scenes_via_geoai_datacubes(aoi, time_range, max_cloud, res, out_dir):
    """Fetch every Sentinel-2 L2A scene over AOI + time_range via
    geoai-datacubes. Returns a list of paths to 13-band GeoTIFFs
    (B01-B12 + SCL) on the shared local-UTM grid."""
    try:
        from geoai_datacubes.fetch import fetch_sentinel_data
    except ImportError as exc:
        raise ImportError(
            "DATA_SOURCE='geoai_datacubes' requires the geoai-datacubes "
            "package. Install with `pip install geoai-datacubes` (see "
            "https://github.com/buckai-observatory/geoai-datacubes). "
            "Alternatively set DATA_SOURCE='local_scenes' and point "
            "LOCAL_SCENES_DIR at a folder of 13-band L2A GeoTIFFs.") from exc

    os.makedirs(out_dir, exist_ok=True)
    fetch_sentinel_data(
        "Sentinel-2",
        bands=_S2_BANDS,
        time_range=time_range,
        roi=aoi,
        resolution=res,
        save_folder=out_dir,
        max_cloud_coverage=max_cloud,
        provider="auto",
    )
    scenes = sorted(glob.glob(
        os.path.join(out_dir, "Sentinel-2_*", "Sentinel-2_full_size.tiff")))
    if not scenes:
        raise RuntimeError(
            f"geoai-datacubes returned no scenes for AOI={aoi}, "
            f"time_range={time_range}, max_cloud={max_cloud}. Widen the "
            "time range or raise MAX_CLOUD_COVERAGE.")
    return scenes


def _scl_water_mask_from_scene(scene_path):
    """Read SCL from a fetched (or pre-downloaded) 13-band S2 scene and
    return a binary non-water mask: 0 = water pixel, 1 = everything else.
    Replaces the old GEE fetch_water_mask() call."""
    with rasterio.open(scene_path) as src:
        if src.count < 13:
            raise RuntimeError(
                f"{scene_path}: expected 13-band L2A scene (bands 1-12 = "
                f"spectral + band 13 = SCL), got {src.count} bands. This "
                "usually means the scene was downloaded by a tool that "
                "did not include the SCL band -- either re-fetch via "
                "geoai-datacubes (DATA_SOURCE='geoai_datacubes') or "
                "regenerate the scene with SCL included.")
        scl = src.read(13)
    # SCL code 6 = water. Non-water becomes 1; water becomes 0.
    return (scl != 6).astype(np.uint8)


# --- Discover Sentinel-2 files -----------------------------------------------
if Sentinel2_level == 'L1C':
    # Legacy L1C branch: unchanged from earlier revisions.
    sentinel_files = glob.glob(os.path.join(script_dir, 'Dongsha_s2_img', '*B02.tif'))
elif DATA_SOURCE == 'geoai_datacubes':
    sentinel_files = _fetch_scenes_via_geoai_datacubes(
        aoi=AOI, time_range=TIME_RANGE,
        max_cloud=MAX_CLOUD_COVERAGE, res=FETCH_RESOLUTION_M,
        out_dir=os.path.join(script_dir, "sentinel_images_L2A_fetched"),
    )
elif DATA_SOURCE == 'local_scenes':
    local_dir = os.path.expanduser(LOCAL_SCENES_DIR)
    if any(c in local_dir for c in "*?["):
        sentinel_files = sorted(glob.glob(local_dir))
    else:
        sentinel_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.tif*"),
                                          recursive=True))
    if not sentinel_files:
        raise FileNotFoundError(
            f"local_scenes mode: no .tif/.tiff files found under {local_dir!r}. "
            "Either point LOCAL_SCENES_DIR at a real folder of 13-band L2A "
            "scenes, or switch to DATA_SOURCE = 'geoai_datacubes'.")
else:
    raise ValueError(
        f"DATA_SOURCE={DATA_SOURCE!r} is not supported. Use "
        "'geoai_datacubes' (fetch from AOI+time_range) or 'local_scenes' "
        "(bring-your-own 13-band L2A GeoTIFFs).")

print(f"Found {len(sentinel_files)} Sentinel-2 file(s) for processing.")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def date_to_seconds_since_1985(date):
    """Convert a datetime to integer seconds elapsed since 1985-01-01."""
    return int((date - datetime(1985, 1, 1)).total_seconds())


def to_wsl_path(win_path):
    """Convert a Windows absolute path to its WSL /mnt/... equivalent."""
    abs_path = os.path.abspath(win_path)
    return "/mnt/" + abs_path[0].lower() + abs_path[2:].replace("\\", "/")


def extract_acquisition_time(filename, sentinel2_level):
    """
    Parse the acquisition timestamp from a Sentinel-2 filename.

    Returns a string formatted as:
      L1C      → 'YYYYMMDDTHHMMSS'
      L2A      → 'YYYY-MM-DD_HH-MM-SS'
      Fallback → 'YYYYMMDD'
    """
    if sentinel2_level == 'L1C':
        m = re.search(r'_(\d{8}T\d{6})_', filename)
        if m:
            return m.group(1)
    else:
        m = re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_', filename)
        if m:
            return f"{m.group(1)}_{m.group(2)}"

    m = re.search(r'(\d{8})', os.path.basename(filename))
    if m:
        return m.group(1)
    raise ValueError(f"No valid acquisition time found in filename: {filename}")


def get_bbox_from_transform(transform, rows, cols):
    """Return the (minx, miny, maxx, maxy) bounding box of a raster."""
    left, top  = transform.c, transform.f
    resx, resy = transform.a, -transform.e
    return (left, top - resy * rows, left + resx * cols, top)

# =============================================================================
# DTU23 TIDAL MODEL
# =============================================================================

def run_dtu23(start_date, end_date, latitude, longitude):
    """
    Call the DTU23 Fortran tidal model and return the predicted tide height (m).

    The executable path and temp-file location are chosen automatically based
    on the detected OS (Windows/WSL or Linux).

    Parameters
    ----------
    start_date, end_date : datetime   1-second window around the acquisition time.
    latitude, longitude  : float      Geographic coordinates of the query point.

    Returns
    -------
    float  Tide height in metres.
    """
    start_s   = date_to_seconds_since_1985(start_date)
    end_s     = date_to_seconds_since_1985(end_date)

    if platform.system() == "Windows":
        temp_real    = os.path.join(DTU23_WIN_PATH, "temp.txt")
        temp_fortran = to_wsl_path(temp_real)
        cmd          = f"cd {to_wsl_path(DTU23_WIN_PATH)} && ./run_perth_new"
        run_cmd      = ["wsl", "bash", "-c", cmd]
    else:
        temp_real    = "/tmp/temp.txt"
        temp_fortran = temp_real
        cmd          = f"cd {DTU23_LINUX_PATH} && ./run_perth_new"
        run_cmd      = ["bash", "-c", cmd]

    inputs  = f"{temp_fortran}\n{start_s}\n{end_s}\n1\n{latitude}\n{longitude}\n"
    process = subprocess.run(run_cmd, input=inputs, text=True, capture_output=True)

    if process.returncode != 0:
        raise RuntimeError(f"DTU23 failed:\n{process.stdout}\n{process.stderr}")
    if not os.path.exists(temp_real):
        raise FileNotFoundError(f"DTU23 output not found: {temp_real}")

    tide_value = None
    with open(temp_real) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tide_value = float(parts[-1])
    return tide_value


def get_tide(lat, lon, acq_datetime):
    """Query DTU23 for the tide height at a single point and moment."""
    return run_dtu23(acq_datetime - timedelta(seconds=1), acq_datetime, lat, lon)


def generate_tide_grid(sentinel_file, acq_datetime, tide_folder):
    """
    Build a spatially interpolated tide-height GeoTIFF for a Sentinel-2 scene.

    Samples DTU23 on a coarse 1/16° grid (≈ 6 km resolution), then bilinearly
    interpolates to the full Sentinel-2 pixel grid.

    Parameters
    ----------
    sentinel_file : str       Path to Sentinel-2 GeoTIFF (for extent / CRS).
    acq_datetime  : datetime  Image acquisition time.
    tide_folder   : str       Output directory for the tide GeoTIFF.

    Returns
    -------
    out_name    : str         Path to the saved GeoTIFF.
    tide_interp : np.ndarray  Tide heights in metres, shape (H, W).
    """
    with rasterio.open(sentinel_file) as src:
        transform     = src.transform
        crs           = src.crs
        height, width = src.height, src.width
        r_idx, c_idx  = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs, ys        = rasterio.transform.xy(transform, r_idx, c_idx)
        lons, lats    = np.array(xs), np.array(ys)

    step        = 1 / 16.0
    coarse_lons = np.arange(np.floor(lons.min() / step) * step, np.ceil(lons.max() / step) * step + step, step)
    coarse_lats = np.arange(np.floor(lats.min() / step) * step, np.ceil(lats.max() / step) * step + step, step)

    tide_samples, coords = [], []
    for lat in coarse_lats:
        for lon in coarse_lons:
            tide_samples.append(get_tide(lat, lon, acq_datetime))
            coords.append((lon, lat))

    tide_interp = griddata(np.array(coords), np.array(tide_samples),
                           np.column_stack([lons.ravel(), lats.ravel()]), method="linear")
    tide_interp = tide_interp.reshape(height, width).astype("float32")

    os.makedirs(tide_folder, exist_ok=True)
    out_name = os.path.join(tide_folder, f"tide_{acq_datetime:%Y%m%d_%H%M%S}.tif")
    with rasterio.open(out_name, "w", driver="GTiff", height=height, width=width,
                       count=1, dtype="float32", crs=crs, transform=transform, compress="lzw") as dst:
        dst.write(tide_interp, 1)
    print(f"Tide grid saved: {out_name}")
    return out_name, tide_interp

# =============================================================================
# PATCH EXTRACTION
# =============================================================================

def extract_patches_no_nans_gen(cube, lidar, valid_mask, patch_size=128, stride=1, min_valid_ratio=0.1):
    """
    Memory-efficient generator that yields valid image patches one at a time.

    A patch is yielded only if it passes both filtering criteria:
      1. Flexible filter : fraction of valid pixels >= min_valid_ratio.
      2. Strict filter   : no NaN in the LiDAR target (only when strict_nan_filter=True).

    Parameters
    ----------
    cube            : np.ndarray (bands, H, W)  Normalised Sentinel-2 cube.
    lidar           : np.ndarray (H, W)         Co-registered LiDAR depth (m), NaN where invalid.
    valid_mask      : np.ndarray (H, W) bool    True for cloud-free, in-extent pixels.
    patch_size      : int                       Square patch side length.
    stride          : int                       Sliding-window stride.
    min_valid_ratio : float                     Minimum valid-pixel fraction [0, 1].

    Yields
    ------
    patch_cube  : float32 (bands, P, P)
    patch_lidar : float32 (P, P)
    patch_mask  : float32 (P, P)   1=valid, 0=invalid
    coords      : tuple (r0, c0, r1, c1)
    """
    global strict_nan_filter

    _, H, W = cube.shape
    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch_mask = valid_mask[r:r + patch_size, c:c + patch_size]

            # Filter 1: minimum valid-pixel fraction
            if np.count_nonzero(patch_mask) / patch_mask.size < min_valid_ratio:
                continue

            patch_cube  = cube[:, r:r + patch_size, c:c + patch_size]
            patch_lidar = lidar[r:r + patch_size, c:c + patch_size]

            # Filter 2 (optional): no NaN in LiDAR target
            if strict_nan_filter and np.isnan(patch_lidar).any():
                continue

            yield (patch_cube.astype(np.float32),
                   patch_lidar.astype(np.float32),
                   patch_mask.astype(np.float32),
                   (r, c, r + patch_size, c + patch_size))

# =============================================================================
# GEOTIFF I/O
# =============================================================================

def save_patches_from_metadata(metadata_list, out_dir, base_transform, crs,
                                date_str, source_file, set_name, prefix="patch"):
    """
    Read patch data from temporary .npz files and write them as 14-band GeoTIFFs.

    Band layout:
      1–12 : Sentinel-2 reflectance
      13   : LiDAR depth (metres)
      14   : Validity mask (1=valid, 0=invalid)
    """
    os.makedirs(out_dir, exist_ok=True)
    for idx, meta in enumerate(metadata_list):
        with np.load(meta['path']) as data:
            X_patch = data['X']
            y_patch = data['y']
            M_patch = data['M']

        r0, c0, r1, c1  = meta['coords']
        patch_transform = base_transform * Affine.translation(c0, r0)
        nbands          = X_patch.shape[0]

        out_path = os.path.join(out_dir, f"{prefix}_{idx:06d}.tif")
        profile  = {
            "driver": "GTiff",
            "height": X_patch.shape[1], "width": X_patch.shape[2],
            "count":  nbands + 2,        # 12 + LiDAR + mask = 14
            "dtype":  "float32",
            "crs":    crs, "transform": patch_transform, "compress": "lzw"
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            for b in range(nbands):
                dst.write(X_patch[b], b + 1)
            dst.write(y_patch, nbands + 1)   # Band 13: LiDAR depth
            dst.write(M_patch, nbands + 2)   # Band 14: validity mask
            dst.update_tags(DATE=date_str, SOURCE_FILE=os.path.basename(source_file),
                            SET=set_name, BAND_14="Validity_Mask")

# =============================================================================
# DEPTH-BALANCED SAMPLING
# =============================================================================

def balance_metadata_by_depth(metadata_list, bins=np.arange(-70, 1, 1)):
    """
    Equalise the 1 m depth-bin distribution using hybrid over/under-sampling.

    Each non-empty bin is brought to the median bin count: bins above the
    median are under-sampled; bins below are over-sampled with replacement.

    Parameters
    ----------
    metadata_list : list of dict  Must contain a 'mean_depth' key.
    bins          : np.ndarray    Bin edges in metres.

    Returns
    -------
    list of dict  Shuffled, balanced metadata.
    """
    if not metadata_list:
        return []

    means   = np.array([m['mean_depth'] for m in metadata_list])
    bin_idx = np.clip(np.digitize(means, bins) - 1, 0, len(bins) - 1)

    bin_to_idx = {}
    for i, b in enumerate(bin_idx):
        bin_to_idx.setdefault(b, []).append(i)

    counts = np.array([len(v) for v in bin_to_idx.values() if v])
    if not len(counts):
        return metadata_list

    target = max(1, int(np.median(counts)))

    balanced = []
    for b in range(len(bins)):
        idxs = bin_to_idx.get(b, [])
        if not idxs:
            continue
        if len(idxs) > target:
            chosen = list(np.random.choice(idxs, target, replace=False))
        elif len(idxs) < target:
            chosen = list(idxs) + list(np.random.choice(idxs, target - len(idxs), replace=True))
        else:
            chosen = list(idxs)
        balanced.extend(chosen)

    np.random.shuffle(balanced)
    return [metadata_list[i] for i in balanced]

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_folder_tifs(folder_path, rotations=(90, 180, 270), flip=True):
    """
    Augment all GeoTIFFs in a folder with rotations and/or a horizontal flip.

    New files are written to the same folder with '_rot{angle}' or '_flip'
    suffixes appended to the base filename.

    Parameters
    ----------
    folder_path : str           Directory containing *.tif files.
    rotations   : list of int   Rotation angles in degrees (multiples of 90).
    flip        : bool          If True, also produce a horizontally-flipped copy.
    """
    if not os.path.exists(folder_path):
        print(f"Augmentation skipped – folder not found: {folder_path}")
        return

    for f in sorted(glob.glob(os.path.join(folder_path, "*.tif"))):
        try:
            with rasterio.open(f) as src:
                arr  = src.read()
                meta = src.meta.copy()
                tags = src.tags()
            base = os.path.splitext(os.path.basename(f))[0]

            for angle in rotations:
                rotated = np.rot90(arr, k=(angle // 90) % 4, axes=(1, 2))
                outp    = os.path.join(folder_path, f"{base}_rot{angle}.tif")
                with rasterio.open(outp, "w", **meta) as dst:
                    dst.write(rotated); dst.update_tags(**tags)

            if flip:
                flipped = np.flip(arr, axis=2)  # horizontal flip
                outp    = os.path.join(folder_path, f"{base}_flip.tif")
                with rasterio.open(outp, "w", **meta) as dst:
                    dst.write(flipped); dst.update_tags(**tags)

        except Exception as e:
            print(f"Augmentation error for {f}: {e}")

    print(f"Augmentation complete: {folder_path}")

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

cloud_removal_count = []

for i, sentinel_file in enumerate(sentinel_files):
    try:
        print("=" * 80)
        print(f"[{i + 1}/{len(sentinel_files)}] {sentinel_file}")

        # --- Parse acquisition date for tagging / filename output ---
        # For geoai-datacubes-fetched scenes, the ISO date lives in the
        # parent directory name (e.g. .../Sentinel-2_2020-06-15_.../
        # Sentinel-2_full_size.tiff). For legacy local scenes, the L2A
        # convention embeds the date in the file basename itself. We
        # search both so either layout works.
        base_name = os.path.basename(sentinel_file)
        parent    = os.path.basename(os.path.dirname(sentinel_file))
        if Sentinel2_level == 'L1C':
            date_str = base_name.split('_')[1][:8]
        else:
            m = re.search(r'(\d{4}-\d{2}-\d{2})', parent) \
                or re.search(r'(\d{4}-\d{2}-\d{2})', base_name) \
                or re.search(r'(\d{8})', parent) \
                or re.search(r'(\d{8})', base_name)
            if not m:
                raise ValueError(
                    f"Cannot parse acquisition date from either the "
                    f"filename {base_name!r} or the parent folder "
                    f"{parent!r}. Rename the scene folder to include a "
                    "YYYY-MM-DD or YYYYMMDD token.")
            date_str = m.group(1).replace('-', '')

        # --- Read Sentinel-2 cube (first 12 bands) + derive water mask ---
        # geoai-datacubes writes 13-band L2A scenes (bands 1-12 = B01-B12,
        # band 13 = SCL). Local pre-downloaded scenes must follow the same
        # convention -- see the DATA_SOURCE docstring in USER CONFIGURATION.
        with rasterio.open(sentinel_file) as src:
            if src.count < 4:
                raise ValueError("Insufficient bands in Sentinel-2 file (expected >= 4).")
            sentinel_transform = src.transform
            sentinel_crs       = src.crs
            cube               = src.read(indexes=list(range(1, 13))).astype(float)

        # --- Cloud / water mask (derived from the SCL band of the same
        # scene we just opened). Cache to Cloud_Masks/rev_YYYYMMDD.tif so
        # a rerun does not re-derive on every pass.
        cloud_mask_glob = glob.glob(os.path.join(script_dir, "Cloud_Masks", f"rev_{date_str}*.tif"))
        if not cloud_mask_glob:
            rows, cols = cube.shape[1], cube.shape[2]
            cloud_mask = _scl_water_mask_from_scene(sentinel_file)
            if cloud_mask.shape != (rows, cols):
                cloud_mask = resize(cloud_mask, (rows, cols),
                                     order=0, preserve_range=True,
                                     anti_aliasing=False)
            cloud_mask = (cloud_mask > 0).astype(np.uint8)

            os.makedirs(os.path.join(script_dir, "Cloud_Masks"), exist_ok=True)
            rev_path = os.path.join(script_dir, "Cloud_Masks", f"rev_{date_str}.tif")
            with rasterio.open(rev_path, "w", driver="GTiff",
                               height=cloud_mask.shape[0], width=cloud_mask.shape[1],
                               count=1, dtype="uint8", crs=str(sentinel_crs),
                               transform=sentinel_transform) as dst:
                dst.write(cloud_mask, 1)
            cloud_mask_file = rev_path
        else:
            cloud_mask_file = cloud_mask_glob[0]

        # Reproject cloud mask to Sentinel-2 grid if dimensions differ
        with rasterio.open(cloud_mask_file) as src:
            cld             = src.read(1)
            cloud_transform = src.transform
            cloud_crs       = src.crs

        if cld.shape != (cube.shape[1], cube.shape[2]):
            dest = np.empty((cube.shape[1], cube.shape[2]), dtype=cld.dtype)
            reproject(source=cld, destination=dest,
                      src_transform=cloud_transform, src_crs=cloud_crs,
                      dst_transform=sentinel_transform, dst_crs=sentinel_crs,
                      resampling=Resampling.nearest,
                      src_nodata=np.nan, dst_nodata=np.nan)
            cloud_mask = dest
        else:
            cloud_mask = cld

        valid_pixel_mask = (cloud_mask == 0)
        cloud_removal_count.append(int(np.sum(cloud_mask > 0)))

        # --- Reproject LiDAR to Sentinel-2 grid ---
        resampled_lidar = np.empty((cube.shape[1], cube.shape[2]), dtype=lidar_grid.dtype)
        reproject(lidar_grid, resampled_lidar,
                  src_transform=lidar_transform, src_crs=lidar_crs,
                  dst_transform=sentinel_transform, dst_crs=sentinel_crs,
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        resampled_lidar[resampled_lidar < deepestDepth2Train] = np.nan

        # --- Tidal correction ---
        acquisition_time_str = extract_acquisition_time(sentinel_file, Sentinel2_level)
        if Sentinel2_level == 'L1C':
            acq_dt = datetime.strptime(acquisition_time_str, '%Y%m%dT%H%M%S')
        else:
            try:
                acq_dt = datetime.strptime(acquisition_time_str, '%Y-%m-%d_%H-%M-%S')
            except ValueError:
                acq_dt = datetime.strptime(acquisition_time_str, '%Y%m%d')

        tide_folder = os.path.join(script_dir, "DTU23_tide")
        tide_path   = os.path.join(tide_folder, f"tide_{acq_dt:%Y%m%d_%H%M%S}.tif")
        if not os.path.exists(tide_path):
            _, tide_grid = generate_tide_grid(sentinel_file, acq_dt, tide_folder)
        else:
            with rasterio.open(tide_path) as src:
                tide_grid = src.read(1)
        resampled_lidar += tide_grid

        # --- Normalise Sentinel-2 reflectance ---
        if Logcube:
            cube = np.log(cube / band_normalization + Epsilon)
        else:
            cube = cube / band_normalization

        # --- Clip to smallest valid overlapping extent ---
        combined_valid = (~np.isnan(cube).any(axis=0)) & (~np.isnan(resampled_lidar)) & valid_pixel_mask
        if not np.any(combined_valid):
            print("No overlapping valid pixels – skipping scene.")
            continue

        rows_idx, cols_idx = np.where(combined_valid)
        min_r, max_r = rows_idx.min(), rows_idx.max()
        min_c, max_c = cols_idx.min(), cols_idx.max()

        cube_clipped       = cube[:, min_r:max_r + 1, min_c:max_c + 1]
        lidar_clipped      = resampled_lidar[min_r:max_r + 1, min_c:max_c + 1]
        valid_mask_clipped = valid_pixel_mask[min_r:max_r + 1, min_c:max_c + 1]
        clipped_transform  = sentinel_transform * Affine.translation(min_c, min_r)
        clipped_crs        = sentinel_crs

        # -----------------------------------------------------------------------
        # PATCH EXTRACTION, SPLITTING, AND SAVING
        # -----------------------------------------------------------------------
        print(f"Extracting patches  [strategy='{split_strategy}', min_valid={valid_pixel_threshold}]")

        with tempfile.TemporaryDirectory(prefix="lidar_patches_") as temp_dir:
            train_meta, val_meta, test_meta = [], [], []
            patch_count = 0

            patch_gen = extract_patches_no_nans_gen(
                cube_clipped, lidar_clipped, valid_mask_clipped,
                patch_size=patch_size, stride=stride_value,
                min_valid_ratio=valid_pixel_threshold
            )

            train_thresh = train_ratio
            val_thresh   = train_ratio + val_ratio

            for X, y, M, coord in patch_gen:
                patch_count += 1

                # Assign to train / val / test
                if split_strategy == 'per_image':
                    rv = random.random()
                    set_name = 'train' if rv < train_thresh else ('val' if rv < val_thresh else 'test')

                elif split_strategy == 'consistent_spatial':
                    r, c, _, _ = coord
                    ph = (r * 31 + c) % 100
                    set_name = ('train' if ph < int(train_thresh * 100)
                                else ('val' if ph < int(val_thresh * 100) else 'test'))
                else:
                    raise ValueError(f"Unknown split_strategy: '{split_strategy}'")

                # Save patch to temp storage
                temp_path = os.path.join(temp_dir, f"patch_{patch_count:07d}.npz")
                np.savez_compressed(temp_path, X=X, y=y, M=M)

                meta_entry = {
                    'path':       temp_path,
                    'mean_depth': float(np.nanmean(y[M > 0])),
                    'coords':     coord
                }
                if   set_name == 'train': train_meta.append(meta_entry)
                elif set_name == 'val':   val_meta.append(meta_entry)
                else:                     test_meta.append(meta_entry)

            print(f"Extracted {patch_count} valid patches.")
            if patch_count == 0:
                print("No valid patches – skipping scene.")
                continue

            print(f"Split: Train={len(train_meta)}, Val={len(val_meta)}, Test={len(test_meta)}")

            # --- Save unbalanced patches ---
            save_patches_from_metadata(train_meta, train_dir, clipped_transform, clipped_crs,
                                       date_str, sentinel_file, "train", prefix=f"{date_str}_train")
            save_patches_from_metadata(val_meta,   val_dir,   clipped_transform, clipped_crs,
                                       date_str, sentinel_file, "valid", prefix=f"{date_str}_val")
            save_patches_from_metadata(test_meta,  test_dir,  clipped_transform, clipped_crs,
                                       date_str, sentinel_file, "test",  prefix=f"{date_str}_test")

            # --- Optional: depth-balanced saves ---
            if apply_balance:
                depth_bins     = np.arange(deepestDepth2Train, 1, 1)
                train_meta_bal = balance_metadata_by_depth(train_meta, bins=depth_bins)
                val_meta_bal   = balance_metadata_by_depth(val_meta,   bins=depth_bins)
                test_meta_bal  = balance_metadata_by_depth(test_meta,  bins=depth_bins)
                print(f"Balanced: Train={len(train_meta_bal)}, Val={len(val_meta_bal)}, Test={len(test_meta_bal)}")

                for d in [train_balanced_dir, val_balanced_dir, test_balanced_dir]:
                    os.makedirs(d, exist_ok=True)
                save_patches_from_metadata(train_meta_bal, train_balanced_dir, clipped_transform, clipped_crs,
                                           date_str, sentinel_file, "train", prefix=f"{date_str}_train")
                save_patches_from_metadata(val_meta_bal,   val_balanced_dir,   clipped_transform, clipped_crs,
                                           date_str, sentinel_file, "valid", prefix=f"{date_str}_val")
                save_patches_from_metadata(test_meta_bal,  test_balanced_dir,  clipped_transform, clipped_crs,
                                           date_str, sentinel_file, "test",  prefix=f"{date_str}_test")

                # Balanced split overlay plot
                b_masks = {k: np.zeros(lidar_clipped.shape, dtype=np.uint8)
                           for k in ('train', 'val', 'test')}
                for meta in train_meta_bal: r0,c0,r1,c1 = meta['coords']; b_masks['train'][r0:r1, c0:c1] = 1
                for meta in val_meta_bal:   r0,c0,r1,c1 = meta['coords']; b_masks['val'][r0:r1, c0:c1]   = 1
                for meta in test_meta_bal:  r0,c0,r1,c1 = meta['coords']; b_masks['test'][r0:r1, c0:c1]  = 1
                invalid_mask = (valid_pixel_mask == 0)
                plt.figure(figsize=(12, 10))
                plt.imshow(lidar_clipped, cmap='gray')
                plt.imshow(b_masks['train'], cmap='Blues',   alpha=0.4)
                plt.imshow(b_masks['val'],   cmap='Greens',  alpha=0.4)
                plt.imshow(b_masks['test'],  cmap='Reds',    alpha=0.4)
                plt.imshow(np.ma.masked_where(~invalid_mask, invalid_mask), cmap='Oranges', alpha=0.5)
                plt.title(f"{date_str} (balanced) – Train (blue), Val (green), Test (red), NaN (orange)")
                plt.axis('off')
                plt.savefig(os.path.join(LiDAR_Model, f"{date_str}_Patches_Mask_balanced.jpg"),
                            dpi=dpi, bbox_inches='tight')
                plt.close()

            # --- Unbalanced split overlay plot ---
            u_masks = {k: np.zeros(lidar_clipped.shape, dtype=np.uint8)
                       for k in ('train', 'val', 'test')}
            for meta in train_meta: r0,c0,r1,c1 = meta['coords']; u_masks['train'][r0:r1, c0:c1] = 1
            for meta in val_meta:   r0,c0,r1,c1 = meta['coords']; u_masks['val'][r0:r1, c0:c1]   = 1
            for meta in test_meta:  r0,c0,r1,c1 = meta['coords']; u_masks['test'][r0:r1, c0:c1]  = 1
            invalid_mask = (valid_pixel_mask == 0)

            plt.figure(figsize=(12, 10))
            plt.imshow(lidar_clipped, cmap='gray')
            plt.imshow(u_masks['train'], cmap='Greens', alpha=0.4)
            plt.imshow(u_masks['val'],   cmap='Reds',   alpha=0.4)
            plt.imshow(np.ma.masked_where(~invalid_mask, invalid_mask), cmap='Oranges', alpha=0.5)
            plt.title(f"Patch split ({date_str}) – Train (green), Val (red), NaN (orange)")
            plt.axis('off')
            plt.savefig(os.path.join(LiDAR_Model, f"{date_str}_Patches_Mask_split.jpg"),
                        dpi=dpi, bbox_inches='tight')
            plt.close()

    except Exception as e:
        print(f"Error processing {sentinel_file}: {e}")

# =============================================================================
# AUGMENTATION
# =============================================================================

if apply_augmentation:
    splits = [
        (train_dir, train_balanced_dir, train_aug_dir, train_ratio),
        (val_dir,   val_balanced_dir,   val_aug_dir,   val_ratio),
        (test_dir,  test_balanced_dir,  test_aug_dir,  1.0 - train_ratio - val_ratio),
    ]
    for raw_dir, bal_dir, aug_dir, ratio in splits:
        os.makedirs(aug_dir, exist_ok=True)
        # Prefer raw patches; fall back to balanced if raw folder is empty or ratio is 0
        source = raw_dir if (ratio > 0 and os.path.exists(raw_dir) and os.listdir(raw_dir)) \
                 else (bal_dir if apply_balance else None)
        if source is None:
            continue
        for f in glob.glob(os.path.join(source, "*.tif")):
            dst = os.path.join(aug_dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy(f, dst)
        augment_folder_tifs(aug_dir, rotations=augmentation_rotations, flip=augmentation_flip)
else:
    print("apply_augmentation=False – augmentation skipped.")

# =============================================================================
# CLOUD-REMOVAL STATISTICS
# =============================================================================

if cloud_removal_count:
    n     = len(cloud_removal_count)
    dates = [
        os.path.basename(f).split('_')[1].replace('-', '') if '_' in os.path.basename(f)
        else os.path.basename(f)
        for f in sentinel_files[:n]
    ]
    pd.DataFrame({'Date': dates, 'Cloud_Pixels_Removed': cloud_removal_count}) \
      .to_csv(os.path.join(LiDAR_Model, 'cloud_removal_counts.csv'), index=False)
    print("Cloud removal statistics saved.")

# =============================================================================
# ELAPSED TIME
# =============================================================================

elapsed          = timer() - startime
days, rem        = divmod(elapsed, 86400)
hours, rem       = divmod(rem, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total elapsed time: {int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")
