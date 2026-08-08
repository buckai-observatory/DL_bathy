---
title: 'DL_bathy: A deep learning pipeline for satellite-derived bathymetry from Sentinel-2 imagery'
tags:
  - Python
  - remote sensing
  - Earth observation
  - satellite-derived bathymetry
  - Sentinel-2
  - deep learning
  - DeepLabV3+
  - coral reefs
  - shallow water
authors:
  - name: Hsiao-Jou Hsu
    orcid: 0009-0008-5863-3229
    corresponding: true
    affiliation: '1'
  - name: Joachim Moortgat
    orcid: 0000-0002-0259-3597
    affiliation: '1, 2'
affiliations:
  - name: 'School of Earth Sciences, The Ohio State University, Columbus, OH, USA'
    index: 1
  - name: 'BuckAI Observatory, College of Arts and Sciences, The Ohio State University, Columbus, OH, USA'
    index: 2
date: 2026-08-08
bibliography: paper.bib
---

# Summary

`DL_bathy` is an open-source, two-step Python pipeline for training deep
learning models to estimate shallow-water bathymetry (water depth, 0-20 m)
from Sentinel-2 multispectral imagery. Step 1 (`step1_splitting.py`) turns
raw Sentinel-2 scenes and reference depth data into standardized,
georeferenced training patches: it masks clouds and non-water pixels using
Google Earth Engine [@gorelick2017gee], applies tidal correction via the
DTU23 global ocean tide model [@andersen2023dtu23], and produces
spatially-aware train/validation/test splits with optional balancing and
augmentation. Step 2 (`step2_train_test.py`) trains a DeepLabV3+
segmentation model [@chen2018deeplabv3plus] on these patches, with a choice
of encoder backbone (ResNet-50/101 [@he2016resnet], EfficientNet-B4
[@tan2019efficientnet], or ConvNeXt-Large [@liu2022convnext]) and a choice
of masked, depth-aware loss function — standard RMSE, MAE, relative
percentage error, Huber loss, or a Smooth Weight Function (SWF)-weighted
RMSE that emphasizes shallow-water accuracy — implemented in PyTorch
[@paszke2019pytorch] using the Segmentation Models PyTorch library
[@iakubovskii2019smp]. The pipeline evaluates trained models on a held-out
test set and produces per-patch depth predictions, error maps, and
aggregate accuracy metrics.

# Statement of need

Shallow-water bathymetry underpins coastal navigation, reef monitoring, and
hazard assessment, but conventional airborne LiDAR and sonar surveys are
costly and cover only a small fraction of the globe's shallow coastal
waters. Satellite-derived bathymetry (SDB) offers a scalable alternative,
but classical SDB methods based on log-linear regression of reflectance are
strongly site-dependent and typically reliable only to about 10 m depth.
Machine learning approaches such as Random Forest improve on this but still
generalize poorly to new regions, and existing deep-learning SDB studies
often rely on shallow architectures, limited training data, or
within-region random data splits that overstate transferability.

`DL_bathy` provides an accessible, reproducible reference implementation of
a high-capacity, general-purpose deep-learning SDB pipeline, developed to
support the comparative assessment of machine learning and deep learning
architectures for transferable SDB reported in @hsu2026sdb. That study
found that general-purpose convolutional backbones trained with this
pipeline match or outperform a task-specific transformer-based bathymetry
network with several times more parameters, while remaining more robust to
geographic transfer than a Random Forest baseline. Releasing the training
and evaluation code lets other researchers reproduce these comparisons,
adapt the pipeline to new sites and sensors, and build on the SWF loss
formulation and spatially-continuous data-splitting strategy without
re-implementing them from scratch.

# State of the field

Traditional SDB methods include linear regression on log-transformed
reflectance and log band-ratio approaches, refined over time to handle
varying water optical conditions but remaining largely site-specific.
Machine learning methods (Random Forest, gradient boosting, XGBoost) improve
accuracy but remain sensitive to training-region characteristics. Prior
deep-learning SDB work has generally used shallow architectures, limited
training data, or evaluation protocols that mix training and test pixels
from the same region, which can substantially overstate cross-regional
transferability. `DL_bathy` instead couples modern, ImageNet-pretrained
convolutional backbones with a rigorous train/test protocol that reserves
entire geographic regions for testing, and benchmarks directly against a
task-specific bathymetry network (Swin-BathyUNet) on the public
MagicBathyNet dataset.

# Software design

The pipeline is organized as two standalone, configuration-driven scripts
plus a small shared module:

- **`step1_splitting.py`** — fetches cloud/water masks from Google Earth
  Engine, computes DTU23 tidal corrections per acquisition, extracts
  fixed-size image patches co-registered with reference bathymetry,
  applies depth-based balancing, and (optionally) augments patches with
  rotations and flips. Output patches are 14-band GeoTIFFs: 12 Sentinel-2
  reflectance bands, a depth band, and a validity mask.
- **`step2_train_test.py`** — loads patches via a lazy-loading PyTorch
  `Dataset`, builds a DeepLabV3+ model with the selected encoder backbone,
  and trains it with early stopping, learning-rate scheduling, and
  optional multi-GPU (`DataParallel`) support. After training, it runs
  inference on the test set and writes per-patch prediction GeoTIFFs,
  error visualizations, and aggregate metrics.
- **`losses.py`** — the five masked, depth-aware loss functions
  (RMSE/SSE, MAE, RPE, SWF, Huber) used by Step 2, factored into a
  standalone module with a small `pytest` unit-test suite so the loss
  logic can be verified independently of the GDAL/Earth-Engine-heavy
  data pipeline.

Both scripts follow a `USER CONFIGURATION` block pattern at the top of the
file, so common changes (data paths, backbone choice, loss function,
hyperparameters) do not require touching the rest of the code.

# Research impact statement

This software was developed to support @hsu2026sdb, which compares
Random Forest and four deep-learning architectures (ResNet-50, ResNet-101,
EfficientNet-B4, ConvNeXt-Large) for transferable satellite-derived
bathymetry across coral reef sites in the South China Sea and Australian
waters, and benchmarks the approach against a task-specific bathymetry
network on the public MagicBathyNet dataset.

# Acknowledgements

This work was supported by the U.S. National Science Foundation's CAIG
program and The Ohio State University's BuckAI Observatory and School of
Earth Sciences.

# AI usage disclosure

Repository hygiene documentation (`LICENSE`, `CITATION.cff`,
`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this `paper.md`) and the
`losses.py` test suite were prepared with the assistance of Anthropic's
Claude. The loss-function extraction and tests were reviewed and verified
by running the test suite against the existing implementation before being
committed. All modeling code, experimental design, and results reported in
@hsu2026sdb are the authors' own work.

# References
