# Changelog

All notable changes to `DL_bathy` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Anything landed on `main` since the last tagged release goes here until
the next tag is cut.

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
