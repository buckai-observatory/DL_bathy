# Satellite-Derived Bathymetry (SDB) — Deep Learning Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `step1_splitting.py` | Prepare, align, split, balance, and (optionally) augment training patches |
| 2 | `step2_train_test.py` | Train a DeepLabV3+ model, evaluate on the test set, and save predictions |

---

## Requirements

### Python packages

```
numpy pandas matplotlib scipy scikit-image
rasterio gdal
earthengine-api
torch torchvision
segmentation-models-pytorch
```

Install with:

```bash
pip install numpy pandas matplotlib scipy scikit-image rasterio gdal \
            earthengine-api torch torchvision segmentation-models-pytorch
```

### External tools

- **Google Earth Engine (GEE) account** — required for cloud/water masking in Step 1.
- **DTU23 tidal model executable** (`run_perth_new`) — required for tide correction in Step 1. See [DTU Space](https://www.space.dtu.dk) for access.

---

## Setup

### 1. Authenticate with Google Earth Engine

```bash
earthengine authenticate
```

Then open `step1_splitting.py` and set your project ID:

```python
GEE_PROJECT_ID = 'ee-your-project-id'
```

### 2. Configure DTU23 paths

In `step1_splitting.py`, set the paths to your DTU23 installation:

```python
# Windows (WSL) path to the directory containing run_perth_new
DTU23_WIN_PATH   = r"C:\path\to\DTU23"

# Linux / HPC path
DTU23_LINUX_PATH = "/path/to/DTU23/SOFTWARE"
```

### 3. Organise your input data

Place the following files/folders next to the scripts:

```
project/
├── step1_splitting.py
├── step2_train_test.py
├── reference.tif   ← Ground Truth GeoTIFF
└── sentinel_images_L2A/
    └── all_bands/
        ├── cloudfree_l2a/            ← Cloud-free Sentinel-2 scenes (*.tif)
        ├── (*.tif)                   ← All-band scenes
        └── L2A_with_clouds/          ← Scenes with clouds (masked in step 1)
```

For L1C imagery, place scenes in `Dongsha_s2_img/` and set `Sentinel2_level = 'L1C'`.

---

## Usage

### Step 1 — Patch preparation

```bash
python step1_splitting.py
```

Key configuration options (edit the `USER CONFIGURATION` block at the top of the script):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split_strategy` | `'consistent_spatial'` | `'per_image'` or `'consistent_spatial'` (recommended to prevent data leakage) |
| `valid_pixel_threshold` | `0.1` | Minimum fraction of valid pixels per patch |
| `strict_nan_filter` | `True` | Discard patches with any NaN in the LiDAR target |
| `train_ratio` / `val_ratio` | `0.80` / `0.10` | Train/val split fractions (test = remainder) |
| `patch_size` | `380` | Square patch side length (pixels) |
| `stride_value` | `190` | Sliding-window stride (pixels) |
| `Logcube` | `True` | Apply log-transform to Sentinel-2 reflectance |
| `apply_augmentation` | `True` | Enable rotation and flip augmentation |
| `Sentinel2_level` | `'L2A'` | `'L1C'` or `'L2A'` |

**Outputs** (inside `LiDAR_Model(augOriginal_allImg)/`):

```
train/          valid/          test/
train_balanced/ valid_balanced/ test_balanced/
train_aug/      valid_aug/      test_aug/
*_Patches_Mask_3way.jpg            ← Split visualisation (balanced)
*_Patches_Mask_3way_ORIGINAL.jpg   ← Split visualisation (unbalanced)
cloud_removal_counts.csv
console_output_step1.log
```

Each GeoTIFF patch contains **14 bands**:
- Bands 1–12: Sentinel-2 reflectance (log-scaled if `Logcube=True`)
- Band 13: LiDAR depth (metres, NaN where invalid)
- Band 14: Validity mask (1 = valid, 0 = invalid)

---

### Step 2 — Training and inference

```bash
python step2_train_test.py
```

Key configuration options (edit the `USER CONFIGURATION` block):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TrainingDataPath` | `"./LiDAR_Model(...)"` | Path to patches from Step 1 |
| `Model_mode` | `"tu-convnext_large"` | Backbone: `tu-convnext_large`, `efficientnet-b4`, `resnet101`, `resnet50` |
| `loss_type` | `"swf"` | Loss: `sse` (RMSE), `mae`, `rpe`, `swf` (depth-weighted), `huber` |
| `num_epochs` | `1000` | Max epochs (early stopping will terminate sooner) |
| `batch_size` | `16` | Batch size per GPU |
| `lr` | `1e-4` | Initial learning rate |
| `earlystop_patience` | `10` | Epochs without improvement before stopping |
| `depth_type` | `"deep"` | `"deep"` or `"shallow"` — which pixels to train on |
| `depth_threshold` | `20` | Depth boundary (m) for deep/shallow classification |

**Outputs** (inside `LiDAR_Model_output/`):

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

---

## Multi-GPU Training

DataParallel is enabled automatically when multiple CUDA GPUs are detected. On SLURM clusters, set `SLURM_CPUS_PER_TASK` to control the DataLoader worker count.

---

## Notes

- **Data leakage**: Use `split_strategy = 'consistent_spatial'` (default) to ensure patches from the same geographic location are always assigned to the same split.
- **Log scaling**: If `Logcube = True` in Step 1, set `scaling_method = "Log_band"` in Step 2 (no additional scaling is applied — the log transform was already applied to the saved patches).
- **Tide correction**: If DTU23 is unavailable, set `tide_grid = np.zeros_like(resampled_lidar)` in Step 1 to skip tidal correction.

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
