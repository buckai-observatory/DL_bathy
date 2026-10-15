"""
Patch preparation (`sdb split`, formerly step1_splitting.py).

Prepares satellite-derived bathymetry training data by:
  - Fetching Sentinel-2 L2A scenes (with SCL cloud/water mask) over the
    AOI + time window via the geoai-datacubes package -- OR reading
    pre-downloaded scenes from disk (DATA_SOURCE = 'local_scenes')
  - Aligning Sentinel-2 imagery with LiDAR ground truth
  - Deriving cloud/water masks from the SCL band (class 6 = water)
  - Generating tide correction grids using pyTMD (EOT20 by default), a
    user-supplied tide grid, or skipping tidal correction entirely
  - Extracting, splitting, (optionally) balancing, and augmenting image patches
  - Saving patches as 14-band GeoTIFFs (see dl_bathy.patch_io)

Setting train_ratio and val_ratio both to 0 puts ALL patches into the test set
(useful for inference-only datasets).

Output directory structure (inside ``output_dir``):
  train/           valid/          test/
  train_balanced/  valid_balanced/ test_balanced/   (only if apply_balance=True)
  train_aug/       valid_aug/      test_aug/        (only if apply_augmentation=True)
  *_Patches_Mask_split.jpg        split overlay visualisation
  cloud_removal_counts.csv
"""

import glob
import os
import random
import re
import shutil
import tempfile
from datetime import datetime
from timeit import default_timer as timer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy.interpolate import griddata
from skimage.transform import resize

from .patch_io import save_patches_from_metadata

# =============================================================================
# SENTINEL-2 ACQUISITION
# =============================================================================
# Two helpers replace the old GEE fetch_water_mask() flow:
#   * _fetch_scenes_via_geoai_datacubes: pulls every S2 L2A scene over the
#     AOI + time_range through the geoai-datacubes package. Each returned
#     file is a 13-band GeoTIFF: bands 1-12 = spectral, band 13 = SCL.
#   * _scl_water_mask_from_scene: reads the SCL band (13) from a scene and
#     returns a binary non-water mask (0 = water pixel, 1 = everything else).

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
            "package. Install with `pip install 'dl-bathy[fetch]'` (see "
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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
# TIDAL CORRECTION
# =============================================================================

def sample_tide_pytmd(lons, lats, acq_datetime, model_name="EOT20", directory=None):
    """
    Predict ocean tide height (m) at a set of points and a single moment in
    time, using the pyTMD package.

    Parameters
    ----------
    lons, lats   : np.ndarray  1-D arrays of query longitudes/latitudes (deg).
    acq_datetime : datetime    Acquisition time shared by all points.
    model_name   : str         Ocean tide model known to pyTMD (default
                                'EOT20'; see pyTMD.io.model for others).
    directory    : str or None Directory holding the model's constituent
                                files. None uses pyTMD's default cache dir.

    Returns
    -------
    np.ndarray  Tide heights in metres, same shape as lons/lats.
    """
    try:
        import pyTMD
    except ImportError as exc:
        raise ImportError(
            "TIDE_SOURCE='pytmd' requires the pyTMD package. Install with "
            "`pip install 'dl-bathy[tide]'`. Alternatively set TIDE_SOURCE='custom' "
            "(bring-your-own tide grids) or 'none' (skip tidal correction).") from exc

    delta_time = np.full(lons.shape, np.datetime64(acq_datetime), dtype="datetime64[ns]")
    tide = pyTMD.compute.tide_elevations(
        lons, lats, delta_time,
        directory=directory, model=model_name,
        crs=4326, type="drift", standard="datetime",
    )
    return np.asarray(tide, dtype="float32")


def generate_tide_grid(sentinel_file, acq_datetime, tide_folder, model_name="EOT20", directory=None):
    """
    Build a spatially interpolated tide-height GeoTIFF for a Sentinel-2 scene
    using pyTMD (TIDE_SOURCE == 'pytmd').

    Samples the tide model on a coarse 1/16° grid (≈ 6 km resolution) in a
    single batched pyTMD call, then bilinearly interpolates to the full
    Sentinel-2 pixel grid.

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
    grid_lon, grid_lat = np.meshgrid(coarse_lons, coarse_lats, indexing="xy")
    coords = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])

    tide_samples = sample_tide_pytmd(coords[:, 0], coords[:, 1], acq_datetime, model_name, directory)

    tide_interp = griddata(coords, tide_samples,
                           np.column_stack([lons.ravel(), lats.ravel()]), method="linear")
    tide_interp = tide_interp.reshape(height, width).astype("float32")

    os.makedirs(tide_folder, exist_ok=True)
    out_name = os.path.join(tide_folder, f"tide_{acq_datetime:%Y%m%d_%H%M%S}.tif")
    with rasterio.open(out_name, "w", driver="GTiff", height=height, width=width,
                       count=1, dtype="float32", crs=crs, transform=transform, compress="lzw") as dst:
        dst.write(tide_interp, 1)
    print(f"Tide grid saved: {out_name}")
    return out_name, tide_interp


def get_tide_grid_for_scene(sentinel_file, acq_datetime, lidar_shape, cfg):
    """
    Dispatch tidal correction according to TIDE_SOURCE and return a tide-
    height grid (metres) matching the Sentinel-2 scene's pixel grid.

    'pytmd'  -- compute (and cache) a tide grid with pyTMD/EOT20.
    'custom' -- read a user-supplied GeoTIFF from CUSTOM_TIDE_DIR.
    'none'   -- return an all-zero grid (no tidal correction applied).
    """
    TIDE_SOURCE = cfg["TIDE_SOURCE"]
    if TIDE_SOURCE == 'none':
        return np.zeros(lidar_shape, dtype="float32")

    if TIDE_SOURCE == 'pytmd':
        tide_folder = cfg["tide_cache_dir"]
        tide_path   = os.path.join(tide_folder, f"tide_{acq_datetime:%Y%m%d_%H%M%S}.tif")
        if not os.path.exists(tide_path):
            _, tide_grid = generate_tide_grid(sentinel_file, acq_datetime, tide_folder,
                                              cfg["TIDE_MODEL_NAME"], cfg["TIDE_MODEL_DIRECTORY"])
            return tide_grid
        with rasterio.open(tide_path) as src:
            return src.read(1)

    if TIDE_SOURCE == 'custom':
        custom_dir  = cfg["CUSTOM_TIDE_DIR"]
        tide_path   = os.path.join(custom_dir, f"tide_{acq_datetime:%Y%m%d_%H%M%S}.tif")
        if not os.path.exists(tide_path):
            raise FileNotFoundError(
                f"TIDE_SOURCE='custom' but no tide grid found at {tide_path!r}. "
                "Place one single-band GeoTIFF per Sentinel-2 scene in "
                "CUSTOM_TIDE_DIR, named tide_<acquisition:%Y%m%d_%H%M%S>.tif, "
                "covering the same extent/CRS as that scene (metres, "
                "positive = higher sea level).")
        with rasterio.open(tide_path) as src:
            tide_grid = src.read(1)
        if tide_grid.shape != lidar_shape:
            raise ValueError(
                f"{tide_path}: custom tide grid shape {tide_grid.shape} does "
                f"not match the Sentinel-2 scene grid {lidar_shape}. Generate "
                "it on the same grid as the Sentinel-2 scene it corrects.")
        return tide_grid

    raise ValueError(
        f"TIDE_SOURCE={TIDE_SOURCE!r} is not supported. Use 'pytmd' "
        "(EOT20 via pyTMD), 'custom' (bring-your-own tide grids), or "
        "'none' (skip tidal correction).")

# =============================================================================
# PATCH EXTRACTION AND SPLITTING
# =============================================================================

def extract_patches_no_nans_gen(cube, lidar, valid_mask, patch_size=128, stride=1,
                                min_valid_ratio=0.1, strict_nan_filter=False):
    """
    Memory-efficient generator that yields valid image patches one at a time.

    A patch is yielded only if it passes both filtering criteria:
      1. Flexible filter : fraction of valid pixels >= min_valid_ratio.
      2. Strict filter   : no NaN in the LiDAR target (only when strict_nan_filter=True).

    Yields
    ------
    patch_cube  : float32 (bands, P, P)
    patch_lidar : float32 (P, P)
    patch_mask  : float32 (P, P)   1=valid, 0=invalid
    coords      : tuple (r0, c0, r1, c1)
    """
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


def assign_split(coord, split_strategy, train_ratio, val_ratio, rng=random):
    """
    Assign a patch to 'train', 'val' or 'test'.

    'consistent_spatial' hashes the patch's top-left corner, so the same
    location always gets the same split across scenes; 'per_image' draws at
    random.
    """
    train_thresh = train_ratio
    val_thresh   = train_ratio + val_ratio
    if split_strategy == 'per_image':
        rv = rng.random()
        return 'train' if rv < train_thresh else ('val' if rv < val_thresh else 'test')
    if split_strategy == 'consistent_spatial':
        r, c, _, _ = coord
        ph = (r * 31 + c) % 100
        return ('train' if ph < int(train_thresh * 100)
                else ('val' if ph < int(val_thresh * 100) else 'test'))
    raise ValueError(f"Unknown split_strategy: '{split_strategy}'")

# =============================================================================
# DEPTH-BALANCED SAMPLING
# =============================================================================

def balance_metadata_by_depth(metadata_list, bins=np.arange(-70, 1, 1)):
    """
    Equalise the 1 m depth-bin distribution using hybrid over/under-sampling.

    Each non-empty bin is brought to the median bin count: bins above the
    median are under-sampled; bins below are over-sampled with replacement.
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


def _split_overlay(lidar_clipped, metas, valid_pixel_mask, cmaps, title, out_path, dpi):
    masks = {k: np.zeros(lidar_clipped.shape, dtype=np.uint8) for k in metas}
    for k, meta_list in metas.items():
        for meta in meta_list:
            r0, c0, r1, c1 = meta['coords']
            masks[k][r0:r1, c0:c1] = 1
    invalid_mask = (valid_pixel_mask == 0)
    plt.figure(figsize=(12, 10))
    plt.imshow(lidar_clipped, cmap='gray')
    for k, cmap in cmaps.items():
        plt.imshow(masks[k], cmap=cmap, alpha=0.4)
    plt.imshow(np.ma.masked_where(~invalid_mask, invalid_mask), cmap='Oranges', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def find_sentinel_files(cfg):
    """List (or fetch) the Sentinel-2 scenes to process."""
    if cfg["Sentinel2_level"] == 'L1C':
        # Legacy L1C branch
        return glob.glob(os.path.join(cfg["l1c_dir"], '*B02.tif'))
    if cfg["DATA_SOURCE"] == 'geoai_datacubes':
        return _fetch_scenes_via_geoai_datacubes(
            aoi=cfg["AOI"], time_range=tuple(cfg["TIME_RANGE"]),
            max_cloud=cfg["MAX_CLOUD_COVERAGE"], res=cfg["FETCH_RESOLUTION_M"],
            out_dir=cfg["fetch_dir"],
        )
    if cfg["DATA_SOURCE"] == 'local_scenes':
        local_dir = cfg["LOCAL_SCENES_DIR"]
        if any(c in local_dir for c in "*?["):
            sentinel_files = sorted(glob.glob(local_dir))
        else:
            sentinel_files = sorted(glob.glob(os.path.join(local_dir, "**", "*.tif*"),
                                              recursive=True))
        if not sentinel_files:
            raise FileNotFoundError(
                f"local_scenes mode: no .tif/.tiff files found under {local_dir!r}. "
                "Either point LOCAL_SCENES_DIR at a real folder of 13-band L2A "
                "scenes, or switch to DATA_SOURCE: geoai_datacubes.")
        return sentinel_files
    raise ValueError(
        f"DATA_SOURCE={cfg['DATA_SOURCE']!r} is not supported. Use "
        "'geoai_datacubes' (fetch from AOI+TIME_RANGE) or 'local_scenes' "
        "(bring-your-own 13-band L2A GeoTIFFs).")


def run(cfg):
    """Run patch preparation with the settings in *cfg* (see dl_bathy.config)."""
    patch_size   = cfg["patch_size"]
    stride_value = cfg["stride_value"]
    train_ratio  = cfg["train_ratio"]
    val_ratio    = cfg["val_ratio"]
    Logcube      = cfg["Logcube"]
    LiDAR_Model  = cfg["output_dir"]
    dpi          = cfg["dpi"]
    os.makedirs(LiDAR_Model, exist_ok=True)

    print(f"Patch size    : {patch_size}")
    print(f"Stride        : {stride_value}")
    print(f"Train ratio   : {train_ratio}")
    print(f"Val ratio     : {val_ratio}")
    print(f"Scaling       : {'log(band/10000)' if Logcube else 'none (original values)'}")

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

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    startime = timer()

    # --- Load LiDAR reference ---
    with rasterio.open(cfg["lidar_file"]) as lidar_data:
        lidar_grid      = lidar_data.read(1).astype(np.float32)
        nodata_value    = lidar_data.nodata
        lidar_transform = lidar_data.transform
        lidar_crs       = lidar_data.crs

    if nodata_value is not None:
        lidar_grid[lidar_grid == nodata_value] = np.nan
    lidar_grid -= cfg["rough_mean_sea_surface"]

    sentinel_files = find_sentinel_files(cfg)
    print(f"Found {len(sentinel_files)} Sentinel-2 file(s) for processing.")

    cloud_mask_dir     = cfg["cloud_mask_dir"]
    cloud_removal_count = []

    for i, sentinel_file in enumerate(sentinel_files):
        try:
            print("=" * 80)
            print(f"[{i + 1}/{len(sentinel_files)}] {sentinel_file}")

            # --- Parse acquisition date ---
            # For geoai-datacubes-fetched scenes, the ISO date lives in the
            # parent directory name (e.g. .../Sentinel-2_2020-06-15_.../
            # Sentinel-2_full_size.tiff). For local scenes, the date is
            # usually in the file name. Both are searched.
            base_name = os.path.basename(sentinel_file)
            parent    = os.path.basename(os.path.dirname(sentinel_file))
            if cfg["Sentinel2_level"] == 'L1C':
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

            # --- Read Sentinel-2 cube (first 12 bands) ---
            with rasterio.open(sentinel_file) as src:
                if src.count < 4:
                    raise ValueError("Insufficient bands in Sentinel-2 file (expected >= 4).")
                sentinel_transform = src.transform
                sentinel_crs       = src.crs
                cube               = src.read(indexes=list(range(1, 13))).astype(float)

            # --- Cloud / water mask, derived from the scene's SCL band and
            # cached as rev_<YYYYMMDD>.tif so reruns skip this step ---
            cloud_mask_glob = glob.glob(os.path.join(cloud_mask_dir, f"rev_{date_str}*.tif"))
            if not cloud_mask_glob:
                rows, cols = cube.shape[1], cube.shape[2]
                cloud_mask = _scl_water_mask_from_scene(sentinel_file)
                if cloud_mask.shape != (rows, cols):
                    cloud_mask = resize(cloud_mask, (rows, cols),
                                        order=0, preserve_range=True,
                                        anti_aliasing=False)
                cloud_mask = (cloud_mask > 0).astype(np.uint8)

                os.makedirs(cloud_mask_dir, exist_ok=True)
                rev_path = os.path.join(cloud_mask_dir, f"rev_{date_str}.tif")
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
            resampled_lidar[resampled_lidar < cfg["deepestDepth2Train"]] = np.nan

            # --- Tidal correction ---
            acquisition_time_str = extract_acquisition_time(sentinel_file, cfg["Sentinel2_level"])
            if cfg["Sentinel2_level"] == 'L1C':
                acq_dt = datetime.strptime(acquisition_time_str, '%Y%m%dT%H%M%S')
            else:
                try:
                    acq_dt = datetime.strptime(acquisition_time_str, '%Y-%m-%d_%H-%M-%S')
                except ValueError:
                    acq_dt = datetime.strptime(acquisition_time_str, '%Y%m%d')

            tide_grid = get_tide_grid_for_scene(sentinel_file, acq_dt, resampled_lidar.shape, cfg)
            resampled_lidar += tide_grid

            # --- Normalise Sentinel-2 reflectance ---
            if Logcube:
                cube = np.log(cube / cfg["band_normalization"] + cfg["Epsilon"])
            else:
                cube = cube / cfg["band_normalization"]

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

            # -------------------------------------------------------------------
            # PATCH EXTRACTION, SPLITTING, AND SAVING
            # -------------------------------------------------------------------
            print(f"Extracting patches  [strategy='{cfg['split_strategy']}', min_valid={cfg['valid_pixel_threshold']}]")

            with tempfile.TemporaryDirectory(prefix="lidar_patches_") as temp_dir:
                train_meta, val_meta, test_meta = [], [], []
                patch_count = 0

                patch_gen = extract_patches_no_nans_gen(
                    cube_clipped, lidar_clipped, valid_mask_clipped,
                    patch_size=patch_size, stride=stride_value,
                    min_valid_ratio=cfg["valid_pixel_threshold"],
                    strict_nan_filter=cfg["strict_nan_filter"],
                )

                for X, y, M, coord in patch_gen:
                    patch_count += 1
                    set_name = assign_split(coord, cfg["split_strategy"], train_ratio, val_ratio)

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
                if cfg["apply_balance"]:
                    depth_bins     = np.arange(cfg["deepestDepth2Train"], 1, 1)
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

                    _split_overlay(lidar_clipped,
                                   {'train': train_meta_bal, 'val': val_meta_bal, 'test': test_meta_bal},
                                   valid_pixel_mask, {'train': 'Blues', 'val': 'Greens', 'test': 'Reds'},
                                   f"{date_str} (balanced) – Train (blue), Val (green), Test (red), NaN (orange)",
                                   os.path.join(LiDAR_Model, f"{date_str}_Patches_Mask_balanced.jpg"), dpi)

                # --- Unbalanced split overlay plot ---
                _split_overlay(lidar_clipped,
                               {'train': train_meta, 'val': val_meta, 'test': test_meta},
                               valid_pixel_mask, {'train': 'Greens', 'val': 'Reds'},
                               f"Patch split ({date_str}) – Train (green), Val (red), NaN (orange)",
                               os.path.join(LiDAR_Model, f"{date_str}_Patches_Mask_split.jpg"), dpi)

        except Exception as e:
            print(f"Error processing {sentinel_file}: {e}")

    # =========================================================================
    # AUGMENTATION
    # =========================================================================

    if cfg["apply_augmentation"]:
        splits = [
            (train_dir, train_balanced_dir, train_aug_dir, train_ratio),
            (val_dir,   val_balanced_dir,   val_aug_dir,   val_ratio),
            (test_dir,  test_balanced_dir,  test_aug_dir,  1.0 - train_ratio - val_ratio),
        ]
        for raw_dir, bal_dir, aug_dir, ratio in splits:
            os.makedirs(aug_dir, exist_ok=True)
            # Prefer raw patches; fall back to balanced if raw folder is empty or ratio is 0
            source = raw_dir if (ratio > 0 and os.path.exists(raw_dir) and os.listdir(raw_dir)) \
                     else (bal_dir if cfg["apply_balance"] else None)
            if source is None:
                continue
            for f in glob.glob(os.path.join(source, "*.tif")):
                dst = os.path.join(aug_dir, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy(f, dst)
            augment_folder_tifs(aug_dir, rotations=cfg["augmentation_rotations"], flip=cfg["augmentation_flip"])
    else:
        print("apply_augmentation=False – augmentation skipped.")

    # =========================================================================
    # CLOUD-REMOVAL STATISTICS
    # =========================================================================

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

    elapsed          = timer() - startime
    days, rem        = divmod(elapsed, 86400)
    hours, rem       = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total elapsed time: {int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")
