"""
inference.py
============
Inference-only entry point for the DL_bathy DeepLabV3+ satellite-derived
bathymetry model. Loads a trained checkpoint from the Hugging Face Hub
(no local weights needed) and runs prediction on Sentinel-2 L2A imagery.

Two typical entry points:

* :func:`predict_from_geoai_datacubes` -- one call, no manual scene download.
  Give it an AOI + time range; it fetches Sentinel-2 L2A via geoai-datacubes,
  reconstructs the model, runs inference, and returns a georeferenced depth
  array. Suitable for the demo notebook and for one-off predictions on
  arbitrary AOIs.

* :func:`load_model` + :func:`predict_depth` -- lower-level path. Load the
  model once, then call :func:`predict_depth` on any (12, H, W) array of
  Sentinel-2 L2A bands 1--12 (B01--B12). Suitable when you already have the
  imagery on disk or when running against many AOIs sequentially.

The model architecture is reconstructed exactly as in ``step2_train_test.py``:
smp.DeepLabV3Plus with a 12-band input head and an ImageNet-pretrained
encoder (default: tu-convnext_large). Only the weights change with each
release published to the Hub.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn

from dl_bathy.bands import normalize_reflectance
from dl_bathy.config import SPLIT_DEFAULTS


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

#: Default HF Hub repository holding published DL_bathy checkpoints.
#: Weights are versioned there; update the ``revision`` argument to pin to a
#: specific release.
#: TEMPORARY: weights are published at Amy5133/dl-bathy-weights pending a
#: transfer to the buckai-observatory org on Hugging Face (needs org access
#: there first). Update back to "buckai-observatory/dl-bathy-weights" once
#: that transfer happens -- HF's repo move preserves all files, so this is
#: the only line that needs to change.
HF_DEFAULT_REPO_ID = "Amy5133/dl-bathy-weights"

#: Default checkpoint filename inside the HF repo. Matches the filename that
#: step2_train_test.py writes at end-of-training.
HF_DEFAULT_FILENAME = "deeplabv3_bathy_final.pth"

#: Encoder backbone used by the reference checkpoint. Same string accepted by
#: ``step2_train_test.py``'s ``Model_mode``. Passed straight through to smp.
DEFAULT_ENCODER = "tu-convnext_large"

# DeepLabV3+ encoder output stride: input height and width must be multiples of it.
_OUTPUT_STRIDE = 16

#: Explicit Sentinel-2 L2A band list to fetch via geoai-datacubes. Must be
#: passed explicitly -- geoai-datacubes' bands=None default for Sentinel-2
#: only resolves to ["B04", "B08", "SCL", "AOT", "WVP"] (5 bands), not the
#: full spectral stack this model expects. Matches step1_splitting.py's
#: _S2_BANDS so fetched scenes have the same 13-band layout (B01-B12 + SCL)
#: the training pipeline uses.
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
            "B8A", "B09", "B11", "B12", "SCL"]


# ----------------------------------------------------------------------------
# Model construction / loading
# ----------------------------------------------------------------------------

def _resolve_encoder(encoder_name: str) -> str:
    """Map aliases used in step2_train_test.py to smp encoder strings."""
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
    key = encoder_name.lower()
    if key not in mode_map:
        raise ValueError(
            f"Unknown encoder_name {encoder_name!r}. "
            f"Choose one of: {sorted(set(mode_map))}."
        )
    return mode_map[key]


def build_model(encoder_name: str = DEFAULT_ENCODER,
                device: Union[str, torch.device, None] = None) -> nn.Module:
    """Reconstruct the DL_bathy DeepLabV3+ architecture (no weights loaded).

    Matches the exact call in ``step2_train_test.py``'s ``load_model``, minus
    the training-only bits (optimiser, scheduler, gradient scaler). Encoder
    weights are set to ``None`` because a checkpoint is expected to overwrite
    them shortly; there is no reason to spend the download budget on ImageNet
    weights that will be discarded on the next line.

    Parameters
    ----------
    encoder_name : str
        Encoder backbone identifier. See :data:`DEFAULT_ENCODER`.
    device : str | torch.device | None
        Where to place the model. ``None`` leaves it on CPU.

    Returns
    -------
    torch.nn.Module
        The model in eval mode. Call :func:`predict_depth` to run inference.
    """
    encoder = _resolve_encoder(encoder_name)
    model = smp.DeepLabV3Plus(
        encoder_name          = encoder,
        encoder_weights       = None,     # will be replaced by checkpoint
        in_channels           = 12,
        classes               = 1,
        activation            = None,
        encoder_output_stride = _OUTPUT_STRIDE,
    )
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


def _default_device() -> torch.device:
    """CUDA if visible, else MPS on Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    repo_id:      str = HF_DEFAULT_REPO_ID,
    filename:     str = HF_DEFAULT_FILENAME,
    revision:     Optional[str] = None,
    encoder_name: str = DEFAULT_ENCODER,
    device:       Union[str, torch.device, None] = None,
    cache_dir:    Optional[str] = None,
) -> nn.Module:
    """Download a checkpoint from the Hugging Face Hub and load it.

    Parameters
    ----------
    repo_id : str
        HF Hub repo holding the checkpoint (``org/repo``). Default:
        :data:`HF_DEFAULT_REPO_ID`.
    filename : str
        Checkpoint filename inside the repo. Default:
        :data:`HF_DEFAULT_FILENAME`.
    revision : str, optional
        Git revision (tag / branch / commit SHA) inside the HF repo. Useful
        to pin a specific model version for reproducibility.
    encoder_name : str
        Encoder backbone; must match the checkpoint's architecture. Default:
        :data:`DEFAULT_ENCODER`.
    device : str | torch.device | None
        Compute device. ``None`` picks CUDA > MPS > CPU automatically.
    cache_dir : str, optional
        Passed to ``huggingface_hub.hf_hub_download``. Defaults to the HF
        cache under ``~/.cache/huggingface``.

    Returns
    -------
    torch.nn.Module
        The model with the checkpoint's weights loaded, in eval mode, on the
        selected device.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for load_model(). "
            "Install with: pip install huggingface_hub"
        ) from e

    ckpt_path = hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir,
    )

    if device is None:
        device = _default_device()

    model = build_model(encoder_name=encoder_name, device=device)
    state = torch.load(ckpt_path, map_location=device)
    # step2 saves either a bare state dict or a {"state_dict": ..., ...} bundle.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Handle checkpoints saved from DataParallel-wrapped models (prefixed
    # with "module."), which is how step2_train_test.py saves them when
    # trained on multiple GPUs.
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------

def predict_depth(
    model:  nn.Module,
    s2_bands: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device, None] = None,
    normalize: bool = True,
    log_scale: bool = SPLIT_DEFAULTS["Logcube"],
    band_normalization: float = SPLIT_DEFAULTS["band_normalization"],
    epsilon: float = SPLIT_DEFAULTS["Epsilon"],
    tile_size: int = SPLIT_DEFAULTS["patch_size"],
    overlap: int = 64,
) -> np.ndarray:
    """Predict per-pixel water depth from a Sentinel-2 L2A tile.

    Scenes of any size work: the model runs on overlapping tiles of
    ``tile_size`` pixels (the training patch size by default), which keeps
    memory bounded and lets the scene's height and width be any value.

    The bands are normalised exactly as ``sdb split`` normalises training
    patches (:func:`dl_bathy.bands.normalize_reflectance`; by default
    ``log(band / 10000 + 1e-6)``), and invalid values are set to 0 as in
    training.

    Parameters
    ----------
    model : torch.nn.Module
        A model returned by :func:`load_model`.
    s2_bands : np.ndarray | torch.Tensor
        Shape ``(12, H, W)`` -- Sentinel-2 L2A bands 1--12 (B01--B12), as
        raw digital numbers (reflectance x 10000), in the band order
        ``sdb split`` uses. NaN pixels (clouds / no-data) are allowed.
    device : str | torch.device | None
        Where to run the forward pass. Defaults to the model's device.
    normalize : bool
        Set to False if ``s2_bands`` is already normalised, for example the
        first 12 bands of a patch written by ``sdb split``.
    log_scale, band_normalization, epsilon : bool, float, float
        The ``Logcube``, ``band_normalization`` and ``Epsilon`` settings the
        model was trained with. The defaults match ``sdb split``'s defaults
        and the released weights.
    tile_size : int
        Tile side in pixels, a multiple of 16. Lower it if memory is short.
    overlap : int
        Overlap between neighbouring tiles in pixels; predictions are
        blended across it.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, float32 -- predicted depth in metres (negative =
        deeper, 0 = sea level), matching the sign convention of the LiDAR
        targets this model was trained on (see ``dl_bathy/split.py``). NaN
        where any input band is NaN or cannot be normalised.
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(s2_bands, torch.Tensor):
        s2_bands = s2_bands.detach().cpu().numpy()
    bands = np.asarray(s2_bands, dtype=np.float64)

    if bands.ndim != 3 or bands.shape[0] != 12:
        raise ValueError(
            f"s2_bands must have shape (12, H, W); got {tuple(bands.shape)}."
        )

    if normalize:
        bands = normalize_reflectance(bands, log_scale, band_normalization, epsilon)
    invalid = ~np.isfinite(bands).all(axis=0)
    bands = np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0)

    depth = _predict_tiled(model, bands.astype(np.float32), device, tile_size, overlap)
    depth[invalid] = np.nan
    return depth


def _tile_starts(n: int, tile: int, stride: int) -> list:
    """Start offsets of tiles of length `tile` covering 0..n, the last one flush with n."""
    if n <= tile:
        return [0]
    starts = list(range(0, n - tile + 1, stride))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def _predict_tiled(model, bands, device, tile_size, overlap):
    """Run `model` on (12, H, W) `bands` in overlapping tiles and blend them.

    DeepLabV3+ needs height and width divisible by its output stride (16), so
    the scene is reflect-padded to a multiple of 16 and cropped back.
    Overlapping tiles are blended with weights that ramp down linearly over
    `overlap` pixels at each tile edge, which hides tile seams.
    """
    if tile_size % _OUTPUT_STRIDE:
        raise ValueError(f"tile_size must be a multiple of {_OUTPUT_STRIDE}; got {tile_size}.")
    _, H, W = bands.shape
    Hp, Wp = (-(-n // _OUTPUT_STRIDE) * _OUTPUT_STRIDE for n in (H, W))
    pad = ((0, 0), (0, Hp - H), (0, Wp - W))
    bands = np.pad(bands, pad, mode="reflect" if min(H, W) > 1 else "edge")

    th, tw = min(tile_size, Hp), min(tile_size, Wp)
    ramp = max(min(overlap, th // 2, tw // 2), 0)

    def weights(n):
        i = np.arange(n, dtype=np.float32)
        return np.minimum(1.0, np.minimum(i + 1, n - i) / (ramp + 1))

    w = np.outer(weights(th), weights(tw))
    acc = np.zeros((Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    with torch.no_grad():
        for r in _tile_starts(Hp, th, max(th - overlap, 1)):
            for c in _tile_starts(Wp, tw, max(tw - overlap, 1)):
                x = torch.from_numpy(np.ascontiguousarray(bands[:, r:r + th, c:c + tw]))
                y = model(x.unsqueeze(0).to(device))[0, 0].cpu().numpy()   # (th, tw)
                acc[r:r + th, c:c + tw] += y * w
                wsum[r:r + th, c:c + tw] += w
    return (acc / wsum)[:H, :W]


# ----------------------------------------------------------------------------
# One-shot convenience: AOI + time range -> depth map
# ----------------------------------------------------------------------------

def predict_from_geoai_datacubes(
    aoi: list,
    time_range: Tuple[str, str],
    model: Optional[nn.Module] = None,
    max_cloud_coverage: float = 0.20,
    resolution: int = 10,
    save_folder: str = "s2_fetched",
    device: Union[str, torch.device, None] = None,
    water_only: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Fetch Sentinel-2 L2A via geoai-datacubes and predict bathymetry.

    Convenience wrapper for the demo notebook: one call, no manual scene
    download. Behaves as ``fetch -> predict_depth -> return``.

    Parameters
    ----------
    aoi : list of 4 floats
        WGS84 bounding box ``[lon_min, lat_min, lon_max, lat_max]``.
    time_range : (str, str)
        ISO date pair ``(start, end)`` used by geoai-datacubes for the STAC
        query.
    model : torch.nn.Module, optional
        A pre-loaded model. ``None`` calls :func:`load_model` with defaults
        (downloads the reference checkpoint from HF Hub).
    max_cloud_coverage : float
        Fraction ``[0..1]`` passed through to geoai-datacubes.
    resolution : int
        Output pixel size in metres. Matches the ``FETCH_RESOLUTION_M``
        parameter in ``step1_splitting.py``.
    save_folder : str
        Where geoai-datacubes writes the fetched scene folder. Local disk
        only; not uploaded anywhere.
    device : str | torch.device | None
        Compute device; defaults to CUDA > MPS > CPU.
    water_only : bool
        Set depths to NaN wherever the scene's Scene Classification Layer
        (band 13) is not water (class 6), i.e. land, cloud, cloud shadow
        and no-data. The model was trained on water pixels only.

    Returns
    -------
    (depths, metadata) : (np.ndarray, dict)
        ``depths`` is a ``(H, W)`` float32 array of predicted depths in
        metres (negative = below sea level). The model was trained on
        0-20 m, so values in optically deep water (open ocean) are not
        meaningful and tend towards 0. ``metadata`` carries scene provenance from geoai-datacubes:
        ``scene_id``, ``date``, ``crs``, ``transform`` (rasterio.Affine),
        and the input ``aoi`` echoed back.
    """
    try:
        from geoai_datacubes.fetch import fetch_sentinel_data
    except ImportError as e:
        raise ImportError(
            "geoai-datacubes is required for predict_from_geoai_datacubes(). "
            "Install with: pip install geoai-datacubes"
        ) from e
    import glob
    import os
    import rasterio

    if model is None:
        model = load_model(device=device)

    fetch_sentinel_data(
        mission="Sentinel-2",
        bands=S2_BANDS,   # explicit 13-band L2A list; see S2_BANDS docstring
        time_range=time_range,
        roi=aoi,
        provider="earthsearch",
        resolution=resolution,
        save_folder=save_folder,
        max_cloud_coverage=max_cloud_coverage,
    )

    # Locate the multi-band GeoTIFF that geoai-datacubes wrote. Its filename
    # convention is <Mission>_full_size.tiff under a scene-id folder.
    matches = sorted(glob.glob(
        os.path.join(save_folder, "*", "Sentinel-2_full_size.tiff")
    ))
    if not matches:
        raise RuntimeError(
            f"No Sentinel-2 scene written under {save_folder!r}; check the "
            "AOI / time range / cloud filter."
        )
    scene_path = matches[-1]

    with rasterio.open(scene_path) as src:
        arr       = src.read().astype(np.float32)
        crs       = src.crs
        transform = src.transform
    # Bands 1..12 are B01..B12; band 13 (if present) is SCL. Drop SCL.
    if arr.shape[0] < 12:
        raise ValueError(
            f"Fetched scene has {arr.shape[0]} bands; expected at least 12."
        )
    s2_bands = arr[:12]

    depths = predict_depth(model, s2_bands, device=device)
    if water_only and arr.shape[0] >= 13:
        depths[arr[12] != 6] = np.nan   # SCL class 6 = water

    scene_id = os.path.basename(os.path.dirname(scene_path))
    return depths, {
        "scene_id":  scene_id,
        "scene_path": scene_path,
        "crs":       crs,
        "transform": transform,
        "aoi":       list(aoi),
        "time_range": list(time_range),
    }


__all__ = [
    "HF_DEFAULT_REPO_ID",
    "HF_DEFAULT_FILENAME",
    "DEFAULT_ENCODER",
    "S2_BANDS",
    "build_model",
    "load_model",
    "predict_depth",
    "predict_from_geoai_datacubes",
]
