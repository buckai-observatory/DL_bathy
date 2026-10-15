"""
Reading and writing the 14-band patch GeoTIFFs shared by `sdb split` and `sdb train`.

Band layout:
  1–12 : Sentinel-2 reflectance
  13   : LiDAR depth (metres, NaN where invalid)
  14   : Validity mask (1 = valid, 0 = invalid)
"""

import os

import numpy as np
import rasterio
from rasterio.transform import Affine


def save_patches_from_metadata(metadata_list, out_dir, base_transform, crs,
                               date_str, source_file, set_name, prefix="patch"):
    """
    Read patch data from temporary .npz files and write them as 14-band GeoTIFFs.
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


def read_patch_tif(path):
    """
    Read a 14-band patch GeoTIFF into numpy arrays.

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
