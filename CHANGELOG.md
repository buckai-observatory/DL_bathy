# Changelog

All notable changes to `DL_bathy` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Anything landed on `main` since the last tagged release goes here until
the next tag is cut.

### Added

- **Installable package and command line**: `pyproject.toml` (`pip install -e ".[all]"`,
  with `train`, `inference`, `fetch`, `tide` and `dev` extras) and an `sdb` command:
  `sdb split`, `sdb train` and `sdb init-config`. The step code moved into
  `dl_bathy/split.py` and `dl_bathy/train.py`; `step1_splitting.py` and
  `step2_train_test.py` remain as thin wrappers.
- **YAML configuration**: settings move from the USER CONFIGURATION blocks into
  one config file with `split` and `train` sections. Defaults reproduce the
  previous scripts; unknown keys are rejected. New `encoder_weights` setting
  (`null` trains without downloading ImageNet weights).
- **Synthetic tests**: config/CLI, patch extraction and splitting, custom tide
  grids, and end-to-end `sdb split` + 1-epoch CPU `sdb train` on a tiny
  synthetic scene; CI now runs them on Linux (Python 3.10–3.12) and Windows,
  plus a job without PyTorch.
- **Tide correction**: `TIDE_SOURCE` switch in `step1_splitting.py` — the
  DTU23 Fortran tide model wrapper is replaced by `pyTMD` + EOT20 by
  default, with `'custom'` (bring-your-own tide grids) and `'none'`
  fallbacks.
- **Inference-only path**: `inference.py`, `MODEL_CARD.md`, and
  `notebooks/dl_bathy_demo.ipynb` — run bathymetry prediction from
  Hugging Face Hub-hosted weights without retraining.
- **Tiered dependency installs**: `requirements-test.txt` and
  `requirements-inference.txt` alongside the existing `requirements.txt`,
  so contributors only touching `losses.py` or `inference.py` skip the
  GDAL / pyTMD / full training-stack install.
- **`smoke-tests/`**: network-dependent, end-to-end scripts (not run in
  CI) covering environment checks and the full inference path against
  the public GBR demo AOI — the scripted equivalent of running
  `notebooks/dl_bathy_demo.ipynb` without Jupyter.

### Fixed

- `filter_valid_patches` no longer drops 13-band patches (without a mask band)
  because of an undefined variable.
- README parameter tables now show the actual defaults (`strict_nan_filter`,
  split ratios, patch size and stride, output folder names).

## `v0.1.0` — 2026-08-08

First public release, archived on Zenodo at
[`10.5281/zenodo.21845182`](https://doi.org/10.5281/zenodo.21845182).

### Added

- **Two-step training pipeline** for satellite-derived bathymetry from
  Sentinel-2 imagery.
  - `step1_splitting.py` — Sentinel-2 acquisition (via
    [`geoai-datacubes`](https://github.com/buckai-observatory/geoai-datacubes)
    by default; local-scenes fallback), tidal correction, patch
    generation, spatially-consistent train / val / test splits, class
    balancing, and rotation / flip augmentation.
  - `step2_train_test.py` — DeepLabV3+ training with a choice of
    encoder backbones (ConvNeXt-Large, EfficientNet-B4, ResNet-50/101),
    mask-aware losses (SSE / MAE / RPE / SWF / Huber), early stopping,
    LR scheduling, multi-GPU training, and full-scene test inference
    with per-patch GeoTIFF predictions and visualisations.
- **Extracted loss module** — `losses.py` isolates the depth-weighted,
  validity-mask-aware loss functions so they can be unit-tested
  without the heavy geospatial dependencies used elsewhere. Unit
  tests live in `tests/test_losses.py`.
- **Public ground-truth subset** — `Ground_Truth.zip` ships the Great
  Barrier Reef bathymetric reference used by the demo path, so the
  training pipeline can be inspected end-to-end without needing
  proprietary LiDAR data.
- **JOSS paper draft** — `paper.md` and `paper.bib`, plus the
  `Draft PDF` GitHub Actions workflow.
- **Community-guideline files** — `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `CITATION.cff`, MIT `LICENSE`.

### Software paper

- Hsu, H.-J., & Moortgat, J. (2026). *From Local Training to Large-Scale
  Mapping: A Comparative Assessment of Machine Learning and Deep
  Learning for Transferable Satellite-Derived Bathymetry*. **Remote
  Sensing** 18(11): 1768. [`10.3390/rs18111768`](https://doi.org/10.3390/rs18111768).

### Timeline

Development history predates the public repository; `v0.1.0` is the
initial snapshot published on 2026-08-08.
