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
from Sentinel-2 multispectral imagery. Step 1 (`sdb split`) turns
Sentinel-2 scenes and reference depth data into standardized,
georeferenced training patches: it fetches Sentinel-2 L2A scenes over an
AOI and time window via the `geoai-datacubes` package
[@moortgat2026geoaidatacubes] (or reads user-supplied pre-downloaded
scenes), derives cloud and non-water masks from the L2A Scene
Classification Layer (SCL), applies tidal correction via the EOT20 global
ocean tide model [@hartdavis2021eot20] using the `pyTMD` package
[@sutterley2024pytmd] (or a user-supplied tide grid), and produces spatially-aware
train/validation/test splits with optional balancing and augmentation.
Step 2 (`sdb train`) trains a DeepLabV3+
segmentation model [@chen2018deeplabv3plus] on these patches, with a choice
of encoder backbone (ResNet-50/101 [@he2016resnet], EfficientNet-B4
[@tan2019efficientnet], or ConvNeXt-Large [@liu2022convnext]) and a choice
of masked, depth-aware loss function — standard RMSE, MAE, relative
percentage error, Huber loss, or a Smooth Weight Function (SWF)-weighted
RMSE that emphasizes shallow-water accuracy — implemented in PyTorch
[@paszke2019pytorch] using the Segmentation Models PyTorch library
[@iakubovskii2019smp]. The pipeline evaluates trained models on a held-out
test set and produces per-patch depth predictions, error maps, and
aggregate accuracy metrics. Pretrained weights are published on the
Hugging Face Hub, so users without reference depth data can run
prediction on new Sentinel-2 scenes directly.

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
reflectance [@lyzenga1978passive] and the log band-ratio method
[@stumpf2003determination], refined over time to handle varying water
optical conditions but remaining largely site-specific. Machine learning
methods (Random Forest, gradient boosting, XGBoost) improve accuracy but
remain sensitive to training-region characteristics. Prior deep-learning
SDB work has generally used shallow architectures, limited training data,
or evaluation protocols that mix training and test pixels from the same
region, which can substantially overstate cross-regional transferability.
`DL_bathy` instead couples modern, ImageNet-pretrained convolutional
backbones with a train/test protocol that reserves entire geographic
regions for testing. @hsu2026sdb benchmarks it against a task-specific
bathymetry network, Swin-BathyUNet [@agrafiotis2025swinbathyunet], on the
public MagicBathyNet dataset [@agrafiotis2024magicbathynet].
Swin-BathyUNet's open implementation targets that network and its
training setup; `DL_bathy` provides the data preparation (Sentinel-2
fetching, cloud and water masks, tide correction, spatially-aware splits)
together with interchangeable general-purpose backbones and loss
functions.

# Software design

`DL_bathy` is a pip-installable Python package (with optional `train`,
`inference`, `fetch` and `tide` extras) with a
command-line interface, `sdb`, and one YAML configuration file whose
`split` and `train` sections hold every setting. `sdb init-config`
writes an example file with all defaults, and unknown keys are rejected
so that typos fail loudly. The package has four parts:

- **`dl_bathy.split`** (`sdb split`) fetches Sentinel-2 L2A scenes over
  an AOI and time window via `geoai-datacubes`, or reads user-supplied
  scenes, and derives cloud and non-water masks from the Scene
  Classification Layer. It computes per-acquisition tide corrections with
  `pyTMD` and EOT20 (or reads user-supplied tide grids, or skips the
  correction), extracts fixed-size patches co-registered with reference
  bathymetry, balances them by depth and optionally augments them with
  rotations and flips. Output patches are 14-band GeoTIFFs: 12 Sentinel-2
  reflectance bands, a depth band, and a validity mask. The
  `consistent_spatial` split assigns patches to train, validation or test
  by spatial block, so that the same location falls in the same split in
  every scene and overlapping patches never straddle two splits.
- **`dl_bathy.train`** (`sdb train`) loads patches through a lazy-loading
  PyTorch `Dataset`, builds a DeepLabV3+ model with the selected encoder,
  and trains it with early stopping, learning-rate scheduling, and
  optional multi-GPU (`DataParallel`) support. After training it runs
  inference on the test set and writes per-patch prediction GeoTIFFs,
  error maps and aggregate metrics.
- **`losses.py`** holds the five masked, depth-aware loss functions
  (RMSE, MAE, RPE, SWF and Huber), separate from the training loop so
  that they can be tested on their own.
- **`inference.py`** rebuilds the trained model from the weights on the
  Hugging Face Hub. `predict_depth` runs it on any array of Sentinel-2
  bands, and `predict_from_geoai_datacubes` fetches the imagery for an
  AOI and predicts depth in one call. A Colab notebook demonstrates this
  path against the public Great Barrier Reef ground truth shipped in the
  repository.

The heavy dependencies are optional extras, so `sdb split` installs and
runs without PyTorch. The test suite runs on synthetic scenes in
continuous integration on Linux and Windows, including an end-to-end
`sdb split` and a one-epoch CPU `sdb train`, and needs no network
access, tide model or GPU. The original `step1_splitting.py` and
`step2_train_test.py` scripts remain as thin wrappers around the package.

# Research impact statement

This software was developed to support @hsu2026sdb, which compares
Random Forest and four deep-learning architectures (ResNet-50, ResNet-101,
EfficientNet-B4, ConvNeXt-Large) for transferable satellite-derived
bathymetry across coral reef sites in the South China Sea and Australian
waters, and benchmarks the approach against a task-specific bathymetry
network on the public MagicBathyNet dataset.

# Acknowledgements

This work was supported by The Ohio State University's BuckAI Observatory
and School of Earth Sciences.

# AI usage disclosure

Anthropic's Claude (through Claude Code) was used to help write the
repository documentation (`README.md`, `CITATION.cff`, `CONTRIBUTING.md`,
`CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `MODEL_CARD.md`) and this paper; to
replace the Google Earth Engine data access with `geoai-datacubes`; to
write the inference module and demo notebook; to replace the Fortran DTU23
tide model with `pyTMD` and EOT20; to package the code with the `sdb`
command line and YAML configuration; and to write the unit tests,
smoke tests and continuous-integration configuration. It was also used to
find and fix bugs, including train/test leakage between overlapping
patches and an inverted `depth_type` setting. The modelling approach
(backbone choices, the SWF loss and the data-splitting strategy), the
experimental design and the results reported in @hsu2026sdb are the
authors' own work. All AI-assisted changes
were reviewed by the authors and checked by running the test suite before
being merged.

# References
