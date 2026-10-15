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


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

#: Default HF Hub repository holding published DL_bathy checkpoints.
#: Weights are versioned there; update the ``revision`` argument to pin to a
#: specific release.
HF_DEFAULT_REPO_ID = "buckai-observatory/dl-bathy-weights"

#: Default checkpoint filename inside the HF repo. Matches the filename that
#: step2_train_test.py writes at end-of-training.
HF_DEFAULT_FILENAME = "deeplabv3_bathy_final.pth"

#: Encoder backbone used by the reference checkpoint. Same string accepted by
#: ``step2_train_test.py``'s ``Model_mode``. Passed straight through to smp.
DEFAULT_ENCODER = "tu-convnext_large"

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
        encoder_output_stride = 16,
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
) -> np.ndarray:
    """Predict per-pixel water depth from a Sentinel-2 L2A tile.

    Parameters
    ----------
    model : torch.nn.Module
        A model returned by :func:`load_model`.
    s2_bands : np.ndarray | torch.Tensor
        Shape ``(12, H, W)`` -- Sentinel-2 L2A bands 1--12 (B01--B12) in the
        same order as ``step1_splitting.py`` writes them. dtype float32.
        NaN pixels (clouds / no-data) are allowed; the model will emit
        predictions for them too, so mask post-hoc with the SCL band if
        needed.
    device : str | torch.device | None
        Where to run the forward pass. Defaults to the model's device.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)`` -- predicted depth in metres (positive = deeper).
        Same dtype as ``model``'s output (typically float32).
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(s2_bands, np.ndarray):
        x = torch.from_numpy(s2_bands.astype(np.float32))
    else:
        x = s2_bands.float()

    if x.ndim != 3 or x.shape[0] != 12:
        raise ValueError(
            f"s2_bands must have shape (12, H, W); got {tuple(x.shape)}."
        )

    x = x.unsqueeze(0).to(device)   # (1, 12, H, W)
    with torch.no_grad():
        y = model(x)                # (1, 1, H, W)
    return y.squeeze(0).squeeze(0).cpu().numpy()


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

    Returns
    -------
    (depths, metadata) : (np.ndarray, dict)
        ``depths`` is a ``(H, W)`` float32 array of predicted depths in
        metres. ``metadata`` carries scene provenance from geoai-datacubes:
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
