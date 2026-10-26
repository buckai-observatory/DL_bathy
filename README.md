# Satellite-Derived Bathymetry (SDB) — Deep Learning Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-BA0C2F.svg?style=flat-square)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![tests](https://github.com/buckai-observatory/DL_bathy/actions/workflows/tests.yml/badge.svg)](https://github.com/buckai-observatory/DL_bathy/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21845182.svg)](https://doi.org/10.5281/zenodo.21845182)
[![BuckAI Observatory](https://img.shields.io/badge/BuckAI-Observatory-BA0C2F.svg?style=flat-square)](https://ai-observatory.osu.edu)

A two-step pipeline for training a deep learning model to estimate shallow-water bathymetry from Sentinel-2 multispectral imagery using available bathymetric measurements as ground truth.

---

## Data availability

`Ground_Truth.zip` contains open bathymetry ground truth (Great Barrier Reef).
Confidential/proprietary data used elsewhere in the associated paper has been
removed and is not included in this repository.

---

## Citation

This code was developed to support the following paper. If you use this code or build upon it, please cite:

> Hsu, H.-J., & Moortgat, J. (2026). *From Local Training to Large-Scale Mapping: A Comparative Assessment of Machine Learning and Deep Learning for Transferable Satellite-Derived Bathymetry*. Remote Sensing, 18(11), 1768. https://doi.org/10.3390/rs18111768

BibTeX:

```bibtex
@article{hsu_moortgat_sdb,
  author    = {Hsu, Hsiao-Jou and Moortgat, Joachim},
  title     = {From Local Training to Large-Scale Mapping: A Comparative Assessment
               of Machine Learning and Deep Learning for Transferable
               Satellite-Derived Bathymetry},
  journal   = {Remote Sensing},
  volume    = {18},
  number    = {11},
  pages     = {1768},
  year      = {2026},
  doi       = {10.3390/rs18111768},
  publisher = {MDPI}
}
```


---

## Overview

Two ways to use this repository:

- **Inference-only path** (no training, no ground truth needed): use
  the pretrained weights hosted on the Hugging Face Hub to run
  bathymetry prediction on any Sentinel-2 tile via the top-level
  `inference.py` module. The Colab-ready notebook
  [`notebooks/dl_bathy_demo.ipynb`](notebooks/dl_bathy_demo.ipynb) walks
  through fetching a scene, running the model, and comparing the
  prediction to the public ground-truth subset in `Ground_Truth.zip` —
  end-to-end, ~10 minutes on a Colab CPU runtime.
- **Full training path** (bring your own ground truth): the two-step
  training pipeline below prepares patches from a LiDAR reference
  raster and trains a new DeepLabV3+ model against them.

| Step | Command / module | Purpose |
|------|--------|---------|
| Inference | `inference.py` | Load pretrained weights from HF Hub, predict depths on Sentinel-2 imagery |
| 1 | `sdb split` | Prepare, align, split, balance, and (optionally) augment training patches |
| 2 | `sdb train` | Train a DeepLabV3+ model, evaluate on the test set, and save predictions |

### Inference-only quickstart

```python
from inference import predict_from_geoai_datacubes

depths, meta = predict_from_geoai_datacubes(
    aoi        = [145.14, -14.56, 145.66, -14.26],   # any lon/lat bbox
    time_range = ("2020-06-01", "2020-09-30"),
)
# depths is a (H, W) numpy array of predicted depths in metres.
```

See [`MODEL_CARD.md`](MODEL_CARD.md) for a full description of the
pretrained model — architecture, training data, intended use,
out-of-scope cautions, and evaluation metrics.

---

## Requirements

### Python packages

```
numpy pandas matplotlib scipy scikit-image
rasterio gdal
geoai-datacubes
torch torchvision
segmentation-models-pytorch
pyTMD
huggingface_hub
```

Install the package from a clone of this repository, choosing the extras you need:

```bash
pip install -e ".[all]"        # everything: training, inference, geoai-datacubes fetch, pyTMD tides
pip install -e ".[train]"      # `sdb split` + `sdb train` with local scenes and no tide model
pip install -e ".[inference]"  # inference.py only
pip install -e .               # `sdb split` only (local scenes, custom or no tides)
```

Extras: `train` (PyTorch, segmentation-models-pytorch), `inference` (adds
`huggingface_hub`), `fetch` (`geoai-datacubes`), `tide` (`pyTMD`), `dev` (`pytest`).
On a GPU cluster, install the PyTorch build matching your CUDA version first.

The previous requirements files still work:

```bash
pip install -r requirements.txt            # full pipeline (step1 + step2 + inference)
pip install -r requirements-inference.txt  # inference.py only, no GDAL/pyTMD
pip install -r requirements-test.txt       # losses.py unit tests only, no GDAL/pyTMD/ML stack
```

`huggingface_hub` is only required for the **inference-only** entry
point (`inference.py`), which downloads the pretrained model weights
from the Hugging Face Hub. The training commands (`sdb split`,
`sdb train`) do not need it.

[`geoai-datacubes`](https://github.com/buckai-observatory/geoai-datacubes)
is a sibling BuckAI Observatory package that DL_bathy uses to fetch
Sentinel-2 L2A imagery (including the SCL cloud/water mask) over any
AOI and time range, with no external accounts required.

### Tide model data (optional)

Step 1 defaults to predicting tide heights with
[`pyTMD`](https://pytmd.readthedocs.io/) using the **EOT20** global ocean
tide model (Hart-Davis et al., 2021). pyTMD is pure Python and pip-installable,
but EOT20's constituent files (~2 GB) are not bundled with pyTMD and must
be downloaded once:

1. Download the EOT20 ocean tide NetCDF files from
   [SEANOE](https://doi.org/10.17882/79489) (open access, no account
   required).
2. Arrange them as `<directory>/EOT20/ocean_tides/*.nc.gz` and point
   `TIDE_MODEL_DIRECTORY` in the `split` section of your config at `<directory>`
   (leave it as `None` to use pyTMD's default cache directory instead).

If you don't need tide model data at all, set `TIDE_SOURCE: custom` to
supply your own pre-computed tide grids, or `TIDE_SOURCE: none` to skip
tidal correction — see the note at the end of this README.

---

## Setup

All settings live in one YAML file with a `split` section (Step 1) and a
`train` section (Step 2). Write an example that lists every setting with
its default:

```bash
sdb init-config dl_bathy_config.yml
```

Settings you leave out take their defaults, which match the original
scripts. Relative paths are resolved against the folder containing the
config file. Unknown setting names are reported as errors.

### 1. Configure tide correction

In the `split` section, choose how Step 1 computes tide heights:

```yaml
split:
  TIDE_SOURCE: pytmd            # pytmd | custom | none
  TIDE_MODEL_NAME: EOT20        # any ocean tide model supported by pyTMD
  TIDE_MODEL_DIRECTORY: null    # dir with EOT20 files; null = pyTMD default cache
  CUSTOM_TIDE_DIR: custom_tide  # only used when TIDE_SOURCE is custom
```

- **`'pytmd'` (default)** — predicts tide height per scene with pyTMD/EOT20.
  Requires the EOT20 data described above.
- **`'custom'`** — bring-your-own tidal correction. Place one single-band
  GeoTIFF per Sentinel-2 scene in `CUSTOM_TIDE_DIR`, named
  `tide_<acquisition:%Y%m%d_%H%M%S>.tif`, covering the same extent/CRS as
  that scene (metres, positive = higher sea level). Useful if you already
  have tide predictions from another model (e.g. FES2014, DTU23, a local
  tide gauge) or want to reuse this repository's pre-pyTMD DTU23 workflow.
- **`'none'`** — skip tidal correction entirely (tide height = 0 m
  everywhere).

### 2. Choose a data-acquisition mode

Step 1 supports two `DATA_SOURCE` modes for the Sentinel-2 imagery,
set in the `split` section:

- **`DATA_SOURCE = 'geoai_datacubes'` (default)** — auto-fetch S2 L2A
  scenes over the AOI + time window using the `geoai-datacubes`
  package (`pip install -e ".[fetch]"`). No credentials, no manual
  downloads. Set the AOI and time range:

  ```yaml
  split:
    AOI: [145.14, -14.56, 145.66, -14.26]   # lon_min, lat_min, lon_max, lat_max
    TIME_RANGE: ["2020-06-01", "2020-09-30"]
    MAX_CLOUD_COVERAGE: 0.20
    FETCH_RESOLUTION_M: 10
  ```

  Fetched scenes land under `sentinel_images_L2A_fetched/` (setting
  `fetch_dir`) next to the config file.

- **`DATA_SOURCE = 'local_scenes'`** — bring-your-own pre-downloaded
  scenes from `LOCAL_SCENES_DIR`. Each scene must be a **13-band L2A
  GeoTIFF**: bands 1–12 = B01–B12, band 13 = SCL. This matches what
  `geoai-datacubes` writes by default; other downloaders that omit
  SCL are not supported by this mode.

### 3. Organise your input data

Only one required file: the LiDAR reference GeoTIFF (setting
`lidar_file`, relative to the config file).
The Sentinel-2 imagery is either auto-fetched (`geoai_datacubes` mode)
or read from `LOCAL_SCENES_DIR` (`local_scenes` mode).

```
project/
├── dl_bathy_config.yml
└── reference.tif   ← LiDAR ground-truth GeoTIFF
```

For L1C imagery, place scenes in `Dongsha_s2_img/` (setting `l1c_dir`) and set
`Sentinel2_level: L1C`. The L1C path is legacy and does not
support the `DATA_SOURCE` modes above; it reads directly from
`Dongsha_s2_img/`.

---

## Usage

### Step 1 — Patch preparation

```bash
sdb split --config dl_bathy_config.yml
```

Key settings (`split` section; see `sdb init-config` for all of them):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_SOURCE` | `'geoai_datacubes'` | `'geoai_datacubes'` (auto-fetch S2 + SCL) or `'local_scenes'` (BYO pre-downloaded 13-band L2A GeoTIFFs) |
| `AOI` | `[145.14, -14.56, 145.66, -14.26]` | WGS84 bbox `(lon_min, lat_min, lon_max, lat_max)`; used when `DATA_SOURCE='geoai_datacubes'` |
| `TIME_RANGE` | `('2020-06-01', '2020-09-30')` | ISO date window; used when `DATA_SOURCE='geoai_datacubes'` |
| `MAX_CLOUD_COVERAGE` | `0.20` | Scene-level cloud filter (0–1); used when `DATA_SOURCE='geoai_datacubes'` |
| `FETCH_RESOLUTION_M` | `10` | Output pixel size in metres; used when `DATA_SOURCE='geoai_datacubes'` |
| `LOCAL_SCENES_DIR` | `'sentinel_images_L2A'` | Folder of pre-downloaded scenes; used when `DATA_SOURCE='local_scenes'` |
| `Sentinel2_level` | `'L2A'` | `'L1C'` or `'L2A'` (the `DATA_SOURCE` modes are L2A-only; `'L1C'` uses the legacy `Dongsha_s2_img/` path) |
| `split_strategy` | `'consistent_spatial'` | `'per_image'` or `'consistent_spatial'` (recommended to prevent data leakage; buffers out patches near split boundaries when `stride < patch_size`, see issue #10) |
| `valid_pixel_threshold` | `0.1` | Minimum fraction of valid pixels per patch |
| `strict_nan_filter` | `False` | Discard patches with any NaN in the LiDAR target |
| `train_ratio` / `val_ratio` | `1.0` / `0.0` | Train/val split fractions (test = remainder); set both to `0.0` for inference-only sites |
| `patch_size` | `512` | Square patch side length (pixels) |
| `stride_value` | `256` | Sliding-window stride (pixels) |
| `Logcube` | `True` | Apply log-transform to Sentinel-2 reflectance |
| `apply_augmentation` | `True` | Enable rotation and flip augmentation |

**Outputs** (inside `Data/`, setting `output_dir`):

```
train/          valid/          test/
train_balanced/ valid_balanced/ test_balanced/   (only if apply_balance)
train_aug/      valid_aug/      test_aug/        (only if apply_augmentation)
*_Patches_Mask_split.jpg           ← Split visualisation (unbalanced)
*_Patches_Mask_balanced.jpg        ← Split visualisation (balanced, if apply_balance)
cloud_removal_counts.csv
```

Each GeoTIFF patch contains **14 bands**:
- Bands 1–12: Sentinel-2 reflectance (log-scaled if `Logcube=True`)
- Band 13: LiDAR depth (metres, NaN where invalid)
- Band 14: Validity mask (1 = valid, 0 = invalid)

---

### Step 2 — Training and inference

```bash
sdb train --config dl_bathy_config.yml
```

Key settings (`train` section):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TrainingDataPath` | `"Data"` | Path to patches from Step 1 |
| `encoder_weights` | `"imagenet"` | Pre-trained encoder weights; `null` trains from scratch without downloading |
| `Model_mode` | `"tu-convnext_large"` | Backbone: `tu-convnext_large`, `efficientnet-b4`, `resnet101`, `resnet50` |
| `loss_type` | `"swf"` | Loss: `sse` (RMSE), `mae`, `rpe`, `swf` (depth-weighted), `huber` |
| `num_epochs` | `1000` | Max epochs (early stopping will terminate sooner) |
| `batch_size` | `16` | Batch size per GPU |
| `lr` | `1e-4` | Initial learning rate |
| `earlystop_patience` | `10` | Epochs without improvement before stopping |
| `depth_type` | `"deep"` | `"deep"` or `"shallow"`. Note (issue #11): the code currently *excludes* the matching pixels, so `"deep"` trains on pixels shallower than `depth_threshold` |
| `depth_threshold` | `20` | Depth boundary (m) for deep/shallow classification |
| `test_depth_range` | `[-20, -0.01]` | `[deepest, shallowest]` depth (m) scored in the test metrics; test pixels outside it are excluded (on top of the `depth_type` filter). The range is written to the `Eval_depth_min_m` / `Eval_depth_max_m` columns of the metrics CSVs |

**Outputs** (inside `Model_output/`, setting `LiDAR_Model`):

```
best_model.pth                    ← Best checkpoint (monitored metric)
deeplabv3_bathy_final.pth         ← Final epoch checkpoint
metadata.json                     ← Run configuration and results summary
training_metrics.csv              ← Per-epoch RMSE / MAE / SWF / RPE / LR
training_curves.png               ← Training vs validation metric plots
hyperparams_<timestamp>.txt       ← Logged hyperparameters
predictions/
  *_pred.tif                      ← Predicted depth GeoTIFFs
  *_full.png                      ← GT / Pred / Error visualisation (full range)
  *_within3m.png                  ← Visualisation for depths 0–3 m
  prediction_metrics.csv          ← Per-patch + overall metrics (full range)
  within3m_prediction_metrics.csv ← Per-patch + overall metrics (0–3 m)
  Scatter_full.png                ← Scatter plot: predicted vs ground truth
  Scatter_within3m.png
console_output_step2.log
```

---

## Loss Functions

| Key | Description |
|-----|-------------|
| `sse` | Mean squared error (equivalent to RMSE minimisation) |
| `mae` | Mean absolute error |
| `rpe` | Mean relative percentage error |
| `swf` | **Smooth Weight Function** — depth-weighted RMSE. Shallow pixels receive higher weight via `w = 1 + β·exp(−|Z|/Z₀)`. Controlled by `SWF_beta` and `SWF_Z0`. |
| `huber` | Smooth L1 loss — L2 for small errors, L1 for outliers |

These are implemented in [`losses.py`](losses.py), independent of the rest
of the training code, so they can be unit-tested without the GDAL /
`geoai-datacubes` / pyTMD dependencies used elsewhere in the pipeline.

---

## Testing

```bash
pip install -e ".[train,dev]"
pytest tests/ -v
```

Tests cover the loss functions, configuration and command line, patch
extraction and splitting, and an end-to-end `sdb split` plus a 1-epoch
CPU `sdb train` on a tiny synthetic scene. They need no network,
`geoai-datacubes`, GDAL build or tide model, and run automatically on
every push and pull request via GitHub Actions
(`.github/workflows/tests.yml`), on Linux and Windows and without
PyTorch for `sdb split`.

### Smoke tests

`tests/` covers `losses.py` in CI; `smoke-tests/` covers the
network-dependent, end-to-end inference path (not run in CI, since it
needs live Sentinel-2 fetches and downloads real model weights) — see
[`smoke-tests/README.md`](smoke-tests/README.md).

---

## Multi-GPU Training

DataParallel is enabled automatically when multiple CUDA GPUs are detected. On SLURM clusters, set `SLURM_CPUS_PER_TASK` to control the DataLoader worker count.

---

## Notes

- **Data leakage**: Use `split_strategy = 'consistent_spatial'` (default) to ensure patches from the same geographic location are always assigned to the same split. With overlapping patches (`stride < patch_size`), some patches near split boundaries are discarded rather than risking leakage — see `assign_split` in `dl_bathy/split.py`.
- **Log scaling**: If `Logcube = True` in Step 1, set `scaling_method = "Log_band"` in Step 2 (no additional scaling is applied — the log transform was already applied to the saved patches).
- **Tide correction**: If you don't have EOT20 data (or any tide model) available, set `TIDE_SOURCE = 'none'` in Step 1 to skip tidal correction, or `TIDE_SOURCE = 'custom'` to supply your own tide grids.

---

## License

Released under the [MIT License](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, request
features, and submit pull requests. Participation is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Citing this software

If you use this code, please cite it — see [CITATION.cff](CITATION.cff) or
the Citation section above.
