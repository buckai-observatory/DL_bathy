# Model Card — DL_bathy (satellite-derived bathymetry)

<!--
  TODO before the first release on Hugging Face Hub: fill in every
  section marked TODO below. Values that require empirical measurement
  (metrics, run time, memory footprint) should come from the final
  training run for the version being published, not from a dev-time
  best-guess. The scaffolding is designed so future releases only need
  to change the numbers -- the section structure stays the same.
-->

## Overview

- **Model name**: DL_bathy DeepLabV3+ satellite-derived bathymetry
- **Model type**: fully-convolutional segmentation network (regression head)
- **Task**: per-pixel water depth estimation (metres) from Sentinel-2 L2A imagery
- **Architecture**: [`smp.DeepLabV3Plus`](https://github.com/qubvel/segmentation_models.pytorch) with a 12-band input head, single-channel regression output, and an ImageNet-pretrained encoder
- **Default encoder**: `resnet50` (ResNet-50). Alternate encoders trained by the same pipeline: `tu-convnext_large` (ConvNeXt-Large, timm backend; the default before v0.2.0), `resnet101`, `efficientnet-b4`. The default follows Hsu & Moortgat (2026): ResNet-50 had the lowest within-region test RMSE (Table 2) and is close to the best in cross-regional transfer (Table 4); ConvNeXt-Large (best at Cartier Reef) and ResNet-101 (best at Ashmore Reef) remain available.
- **License**: MIT (matches the DL_bathy repository)

## Intended use

- Predicting shallow-water bathymetry (depth in metres) over any location
  covered by Sentinel-2, given a cloud-free (or SCL-masked) L2A tile
  containing bands B01–B12.
- Research and educational use: reproducing the results of Hsu & Moortgat
  (2026); benchmarking against other satellite-derived bathymetry (SDB)
  approaches; exploring transfer to new geographies.

## Out-of-scope / cautions

- **Deep water**: the training distribution is shallow-water reef and
  coastal environments (< ~25 m). Predictions for open ocean or turbid
  inland water bodies are extrapolations and should not be trusted
  without independent validation.
- **Turbidity, sun glint, breaking waves**: not explicitly handled by the
  model; masking these upstream (e.g. using SCL or a scene-level QA step)
  is the caller's responsibility.
- **Not a real-time / navigational product**: outputs are research-grade,
  not certified for maritime navigation, dredging planning, or any
  safety-critical use.

## Training data

- **Ground truth**: airborne LiDAR bathymetry (see
  [Hsu & Moortgat, 2026](https://doi.org/10.3390/rs18111768) for the
  full site list and processing chain).
- **Public reference subset**: an open subset of the ground-truth data
  is shipped in `Ground_Truth.zip` in the [DL_bathy](https://github.com/buckai-observatory/DL_bathy)
  repository for reproducibility and demo purposes. The full training
  dataset includes commercially licensed LiDAR that cannot be
  redistributed; this is documented in the repo's `Data availability`
  section.
- **Sentinel-2 imagery**: L2A reflectance bands 1–12 (B01–B12). Cloud
  and shadow pixels are excluded via the SCL band during training.
- **How the released weights' training data was made**: the Sentinel-2
  scenes were downloaded from Google Earth Engine, and the labels were
  tide-corrected with the DTU23 tide model (January 2026). The current
  pipeline (`sdb split`) downloads scenes with geoai-datacubes (since
  August 2026) and corrects tides with pyTMD/EOT20 (since October 2026);
  retraining with it will not reproduce the released weights exactly.

- **Patches used by the released checkpoints**:
  - Patch size: 512 × 512 pixels.
  - Split: 70 % training, 15 % validation, 15 % test.
  - Augmentation: each training and validation patch was also saved rotated
    by 90°, 180° and 270° and flipped horizontally, so these sets have five
    copies of each patch. Test patches were not augmented.
  - Counts, totalled over the three sites: 2,590 training patches (518
    before augmentation), 520 validation patches (104 before augmentation)
    and 144 test patches from 10 scene dates (2020–2024). Every patch passed
    the 10 % valid-pixel filter.

## Training configuration

Values below are read from each released checkpoint's `metadata.json`
on the Hugging Face Hub and from the training log of the run that produced
it (the released `deeplabv3_bathy_final.pth` files have the same SHA-256 as
the files those runs saved). All four deep-learning checkpoints share the
same settings apart from the encoder; they were trained in January 2026.

- Encoder backbone: `resnet50` (default), `resnet101`, `efficientnet-b4` or `tu-convnext_large`, one checkpoint each
- Input channels: 12 (Sentinel-2 bands B01–B12)
- Input normalisation: `log(band / 10000 + 1e-6)` of the L2A digital numbers
  (`Logcube: true`, `band_normalization: 10000`, `Epsilon: 1e-6`; each
  checkpoint's `metadata.json` records `scaling_method: Log_band`), with
  invalid values set to 0. `inference.predict_depth` applies this to raw
  bands by default.
- Output: single-channel regression, no activation (raw depth in metres)
- Depth range: trained on 0–20 m. Each checkpoint's `metadata.json` records
  `depth_type: "deep"` because it was trained before issue #11 was fixed,
  when `"deep"` kept depths shallower than `depth_threshold`. With the
  current code, the same setting is `depth_type: "shallow"` (the default).
- Tide correction: the training labels were made with DTU23 tide heights
  **added** to the LiDAR depths, the sign later found to be wrong (issue #47;
  `sdb split` subtracts them from v0.2.0). At the demo site the old sign shifts
  predictions by about 0.1 m, because the mean tide at the Sentinel-2 overpass
  time there is close to zero.
- Loss: SWF (smooth weight function, depth-weighted squared error), per-pixel
  weight `1 + beta * exp(-|z| / Z0)` with `SWF_beta = 5.0`, `SWF_Z0 = 10.0`
  (`loss_type: swf`)
- Optimiser: AdamW, weight-decay 1e-4, initial LR 1e-4
- Gradient clipping: max norm 5.0
- Patch filtering: patches with < 10 % valid pixels are skipped
  (`min_valid_ratio: 0.1`)
- Scheduler: ReduceLROnPlateau (factor 0.5, patience 5)
- Batch size: 16
- Epochs: up to 1000 with early stopping after 10 epochs without
  validation improvement; epochs actually run, best validation loss (SWF
  units), GPU and total run time per checkpoint:

  | Encoder | Epochs run | Best validation loss | GPU | Run time |
  |---|---|---|---|---|
  | `tu-convnext_large` | 23 | 4.50 | NVIDIA L40S | 20 h 45 min |
  | `resnet101` | 33 | 3.23 | NVIDIA A40 | 25 h 46 min |
  | `resnet50` (default) | 40 | 3.60 | NVIDIA A40 | 26 h 33 min |
  | `efficientnet-b4` | 23 | 4.97 | NVIDIA L40S | 20 h 10 min |

- Hardware: one GPU per run (`gpu_count_used: 1`), 16 data-loader workers
- Run time is the logged wall-clock time of the whole run: data loading,
  training, and inference on the 144 test patches.

## Evaluation

The study-wide evaluation of these models, including transfer to other
sites, is reported in [Hsu & Moortgat (2026)](https://doi.org/10.3390/rs18111768).
TODO (Amy): optionally copy the paper's held-out test-set metrics here.

**Public demo scene.** The table below scores all four released checkpoints
against the public ground truth shipped with the repository, so anyone can
reproduce it with `notebooks/dl_bathy_demo.ipynb` or the smoke test:

- Scene: Sentinel-2A L2A, 19 August 2020 (tile 55LBE), fetched with
  `geoai-datacubes` over the demo AOI `[145.14, -14.56, 145.66, -14.26]`
  (Great Barrier Reef), search window 2020-06-01 – 2020-09-30, ≤ 20 % cloud.
- Ground truth: `Great_Barrier_Reef_A_2020_30m_MSL_cog_depth_0_to_-30_T55LCE_test.tif`
  from `Ground_Truth.zip`, resampled to the prediction grid.
- Pixels scored: SCL water pixels where the ground truth is 0–20 m deep
  (the training range), n = 487,070. Error = prediction − ground truth, with
  depths negative below sea level, so a positive bias means the predictions
  are too shallow.
- Vertical reference: the ground truth is in local mean sea level, the same
  reference as the GBR training labels (their MSL values were used as EGM2008
  heights without a shift), so the comparison has no datum offset (issue #87).

| Encoder | RMSE (m) | MAE (m) | Bias (m) | CPU inference time |
|---|---|---|---|---|
| `tu-convnext_large` | 3.00 | 1.86 | +1.10 | 521 s |
| `resnet50` (default) | 2.49 | 1.74 | +1.06 | 56 s |
| `resnet101` | 2.62 | 1.76 | +1.00 | 64 s |
| `efficientnet-b4` | 4.05 | 2.65 | +2.23 | 159 s |

Measured on 26 November 2026 with DL_bathy commit `01300ae` on an Apple
silicon Mac (M1 Max, CPU only; inference time covers the whole scene, not
loading the weights). Including ground truth deeper than 20 m (0–30 m,
n = 756,342) raises RMSE to 13.7–15.0 m for every encoder, because the models
were not trained beyond 20 m (see *Out-of-scope*). Over the whole scene,
`efficientnet-b4` also predicts some water pixels far above sea level (up to
+15.5 m); the other checkpoints stay below +0.7 m.

**Error by depth.** The bias is not a constant offset. It changes with depth
in the same way for every checkpoint. The table gives the bias (m) by
ground-truth depth band, for the same scene and pixels:

| Ground-truth depth | Share of pixels | `tu-convnext_large` | `resnet50` | `resnet101` | `efficientnet-b4` |
|---|---|---|---|---|---|
| 0–2 m | 9.9 % | −0.49 | −0.54 | −0.65 | −0.28 |
| 2–5 m | 33.4 % | +0.23 | +0.23 | +0.08 | +0.53 |
| 5–10 m | 19.8 % | +0.56 | +0.83 | +0.54 | +1.89 |
| 10–15 m | 19.0 % | +1.11 | +1.37 | +1.33 | +2.98 |
| 15–20 m | 18.0 % | +4.15 | +3.40 | +3.75 | +6.38 |
| **0–20 m** | 100 % | **+1.10** | **+1.06** | **+1.00** | **+2.23** |

- Every checkpoint predicts too deep in water shallower than 2 m, and too
  shallow from a few metres down, increasingly so with depth. This is the
  regression to the mean that Hsu & Moortgat (2026) describe at the transfer
  sites (Ashmore and Cartier reefs).
- Depths of 10–20 m (37 % of the pixels) account for most of the overall bias:
  for `resnet50`, +0.87 m of the +1.06 m.
- The notebook and the smoke test print this breakdown with
  `dl_bathy.scoring.depth_band_errors`.

## How to use

Minimal example (install with
`pip install "dl-bathy[inference,fetch] @ git+https://github.com/buckai-observatory/DL_bathy"`,
or `pip install -e ".[inference,fetch]"` in a clone of the repository):

```python
from inference import predict_from_geoai_datacubes

# Predict bathymetry over a small AOI in the Great Barrier Reef (the same
# region the public Ground_Truth.zip covers).
depths, meta = predict_from_geoai_datacubes(
    aoi=[145.14, -14.56, 145.66, -14.26],
    time_range=("2020-06-01", "2020-09-30"),
    max_cloud_coverage=0.20,
)
print(f"Predicted depth map shape: {depths.shape}, dtype: {depths.dtype}")
```

For finer-grained control, load the model once and reuse it across many
AOIs:

```python
from inference import load_model, predict_depth
import rasterio

model = load_model()
with rasterio.open("my_s2_scene.tif") as src:
    s2_bands = src.read()[:12].astype("float32")
depths = predict_depth(model, s2_bands)
```

To use another released checkpoint, pass its file and encoder, e.g.
`load_model(filename="resnet101/deeplabv3_bathy_final.pth", encoder_name="resnet101")`;
the README lists all four.

See `notebooks/dl_bathy_demo.ipynb` in the repository for a Colab-ready
end-to-end walkthrough on the public GBR subset.

## Citation

If you use these weights, please cite:

- Hsu, H.-J., & Moortgat, J. (2026). *From Local Training to Large-Scale Mapping: A Comparative Assessment of Machine Learning and Deep Learning for Transferable Satellite-Derived Bathymetry*. **Remote Sensing** 18(11): 1768. https://doi.org/10.3390/rs18111768
- The DL_bathy software: see `CITATION.cff` in the [DL_bathy repository](https://github.com/buckai-observatory/DL_bathy).

## Version history

TODO — one bullet per HF Hub release, with training config deltas and
new evaluation numbers.

- `v0.1.0` — first public release, reproducing the setup used in
  Hsu & Moortgat (2026).
