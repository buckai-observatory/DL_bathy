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
- **Default encoder**: `tu-convnext_large` (ConvNeXt-Large, timm backend). Alternate encoders trained by the same pipeline: `efficientnet-b4`, `resnet50`, `resnet101`.
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
- **Sentinel-2 imagery**: L2A reflectance bands 1–12 (B01–B12), fetched
  at the training AOIs / dates via the geoai-datacubes pipeline. Cloud
  and shadow pixels are excluded via the SCL band during training.

TODO: total number of training patches, patch size (px), split ratios,
augmentation strategy, and per-site sample counts (as fixed for the
release being published).

## Training configuration

TODO once the release checkpoint is trained. Should mirror the
`params` dict at the top of `step2_train_test.py` for the training run
that produced the released weights.

- Encoder backbone: `tu-convnext_large`
- Input channels: 12 (Sentinel-2 bands B01–B12)
- Output: single-channel regression, no activation (raw depth in metres)
- Loss: TODO — SSE (RMSE), MAE, RPE, SWF (depth-weighted RMSE), or Huber (which one, and its hyperparameters)
- Optimiser: AdamW, weight-decay 1e-4, initial LR TODO
- Scheduler: ReduceLROnPlateau (factor 0.5, patience TODO)
- Batch size / epochs / early-stopping: TODO
- Hardware: TODO (single GPU, multi-GPU DataParallel, etc.)
- Wall-clock training time: TODO

## Evaluation

TODO — from the release training run:

- Held-out test set: RMSE (m), MAE (m), R², bias
- Per-depth-bin performance (e.g. RMSE at 0–5 m, 5–10 m, 10–15 m, > 15 m)
- Transfer performance on a geographically-held-out site, if measured

## How to use

Minimal example (needs `pip install huggingface_hub geoai-datacubes` and
this repository on the Python path):

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
