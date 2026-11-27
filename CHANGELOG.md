# Changelog

All notable changes to `DL_bathy` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Anything landed on `main` since the last tagged release goes here until
the next tag is cut.

## `v0.2.0` — 2026-11-26

First release since `v0.1.0`. DL_bathy is now an installable package with
the `sdb` command and YAML configuration. Scenes are fetched with
geoai-datacubes instead of Google Earth Engine, and tides come from pyTMD and
EOT20 instead of DTU23. The four pretrained checkpoints are on the Hugging
Face Hub, with an inference module, a model card and a demo notebook. Some of
the fixes below change the training data and test metrics compared with
`v0.1.0`.

### Added

- **Documentation website** at <https://buckai-observatory.org/DL_bathy/>:
  `site/build_site.py` renders README.md, docs/, the project files, the
  notebooks and an API reference generated from the docstrings (quartodoc)
  with Quarto; `.github/workflows/site.yml` builds it on pull requests and
  deploys it to GitHub Pages from `main`. New `docs` extra.
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
- **Tide correction**: the `TIDE_SOURCE` setting — the DTU23 Fortran tide
  model wrapper is replaced by `pyTMD` + EOT20 by default, with `'custom'`
  (bring-your-own tide grids) and `'none'` fallbacks.
- **Inference-only path**: `inference.py`, `MODEL_CARD.md`, and
  `notebooks/dl_bathy_demo.ipynb` — run bathymetry prediction from
  Hugging Face Hub-hosted weights without retraining.
- **Requirements files**: `requirements.txt` (the full pipeline),
  `requirements-inference.txt` (inference only) and `requirements-test.txt`
  (the package with the `train` and `dev` extras, as CI installs it), so
  running inference or the tests does not need geoai-datacubes or pyTMD.
- **`smoke-tests/`**: network-dependent, end-to-end scripts (not run in
  CI) covering environment checks and the full inference path against
  the public GBR demo AOI — the scripted equivalent of running
  `notebooks/dl_bathy_demo.ipynb` without Jupyter.
- **Errors by depth band (issue #87)**: `dl_bathy.scoring.depth_band_errors`
  breaks the errors against ground truth down by ground-truth depth band. The
  demo notebook and the smoke test print it next to the overall RMSE, MAE and
  bias, and `MODEL_CARD.md` gives the demo results by depth band. The overall
  bias of about +1 m comes mostly from depths of 10–20 m, where the models
  predict too shallow, not from a constant or datum offset.
- **Reproducible environment with pixi**: `[tool.pixi]` in `pyproject.toml`
  and `pixi.lock` pin every package for macOS (Apple silicon and Intel),
  Linux and Windows. `pixi run --locked -e test test` runs the tests and
  `pixi run --locked -e notebook lab` opens the demo notebook. The new
  `pixi-locked` workflow installs the locked environment on all four systems
  and runs the test suite in it.
- **Demo notebook in continuous integration**: the new `demo-notebook`
  workflow runs `notebooks/dl_bathy_demo.ipynb` from top to bottom on Ubuntu
  with Python 3.12. On pull requests that change the notebook or the code it
  calls, it runs against the pull request's code; monthly and on demand it
  runs the notebook unchanged, installing DL_bathy from GitHub as on Colab.
- **`Draft PDF` workflow**: builds the JOSS paper PDF from `paper.md` and
  `paper.bib` whenever either changes.

### Changed

- **Scenes are fetched with geoai-datacubes** (`DATA_SOURCE:
  geoai_datacubes`, the default): `sdb split` downloads Sentinel-2 L2A scenes
  with their Scene Classification Layer over `AOI` and `TIME_RANGE` and masks
  clouds and land with that layer, so no Google Earth Engine account or
  manual download is needed. `DATA_SOURCE: local_scenes` reads downloaded
  13-band scenes instead. `v0.1.0` read downloaded scenes and fetched the
  cloud and water masks from Google Earth Engine.
- **Random Forest baseline scope (issue #60)**: the README and `paper.md`
  state that the Random Forest baseline of Hsu & Moortgat (2026) is not part
  of DL_bathy, and `paper.md` no longer says the released code reproduces
  that comparison.
- **Tide correction sign (issue #47)**: `sdb split` now subtracts the tide
  height from the LiDAR depths instead of adding it, so a label is the depth
  below the sea surface when the image was taken (deeper at high tide).
  Training data made with earlier versions, and the released weights, used
  the old sign. A check against EOT20 on 22 scenes at the demo site puts the
  effect at about 0.1 m there. Retrain to get models made with the new sign.
- **The default inference model is now ResNet-50** (`inference.DEFAULT_ENCODER =
  "resnet50"`, `HF_DEFAULT_FILENAME = "resnet50/deeplabv3_bathy_final.pth"`),
  following Hsu & Moortgat (2026): lowest within-region test RMSE (Table 2),
  close to the best in cross-regional transfer (Table 4), about five times
  smaller and seven times faster on CPU than ConvNeXt-Large. ConvNeXt-Large,
  the previous default, loads with `load_model(filename="deeplabv3_bathy_final.pth",
  encoder_name="tu-convnext_large")`. The README lists all four checkpoints.
  The training default (`Model_mode`) is unchanged.
- The model weights moved to the BuckAI Observatory organization on Hugging Face:
  `inference.HF_DEFAULT_REPO_ID` is now `BuckAIObservatory/dl-bathy-weights`
  (the old `Amy5133/dl-bathy-weights` name redirects).
- `losses.masked_sum_swf_and_count` defaults to `Z0=10.0`, the training
  default, instead of 20.0 (issue #24). Training always passes the configured
  value, so it is unaffected.

### Fixed

- **Issue #16** (model card): `MODEL_CARD.md` gives the GPU, run time and
  patch counts of the four released checkpoints, read from their training
  logs. The released files match those runs by SHA-256. Early-stopping and
  scheduler patience are confirmed by the logs. The model card also gives the
  patch size, split ratios, augmentation and patch counts.
- **`tide` extra missing `h5py`**: pyTMD reads the EOT20 files through
  h5netcdf, which needs `h5py`, so `TIDE_SOURCE: pytmd` failed with
  `No module named 'h5py'` at the first scene. The `tide` extra and
  `requirements.txt` now include `h5py`, `sdb split` checks for it before
  fetching anything, and the README and error message give EOT20's real
  layout (`<directory>/EOT20/ocean_tides/*_ocean_eot20.nc`, from
  `ocean_tides.zip` inside the SEANOE download).
- **Issue #77**: with `TIDE_SOURCE: pytmd`, `sdb split` checks for pyTMD and
  the EOT20 files before fetching any scene. If they are missing, it says how
  to get them or how to skip the tide correction, instead of failing after
  the download.
- **Issue #14**: with `TIDE_SOURCE: pytmd`, the tide grid passed the scene's
  projected coordinates (metres, e.g. UTM for every geoai-datacubes scene) to
  pyTMD as longitude and latitude, so the tide heights were wrong. Pixel
  centres are now reprojected to WGS84 first.
- **Issue #58** (README gaps for new users): the README explains how to split
  the three rasters in `Ground_Truth.zip` into train / validation / test
  patches, and which device training runs on. `sdb split` with a missing
  `lidar_file` now exits with a one-line error before creating any folder.
  The demo notebook uses the repository's `Ground_Truth.zip` when run from a
  clone, scores against the `_test` raster explicitly, and the files it
  writes in `notebooks/` are git-ignored. (Loading the other three released
  checkpoints was documented with the ResNet-50 default change.)
- **Issue #59** (confusing output on a default CPU/Mac run): the logged RPE
  leaves out depths shallower than 0.01 m (as in the paper) instead of
  reporting tens of millions of percent; the RPE training loss is unchanged.
  A null `num_workers` now uses the CPU count capped at 8 (or
  `SLURM_CPUS_PER_TASK`) instead of 16. With `split_strategy:
  consistent_spatial`, patches are no longer buffered out when every patch goes
  to one split (e.g. `train_ratio: 1.0`), since there is no leakage to prevent.
  Patch transforms no longer use the deprecated `Affine *` operator, and a patch
  without finite labels no longer warns about an empty slice.
- **Issue #36**: `cloud_removal_counts.csv` gave the wrong date for
  geoai-datacubes scenes and for every scene after a failed one, and its
  `Cloud_Pixels_Removed` column counted all non-water pixels. Each row now has
  its own scene's date, and the column is renamed `Non_water_pixels_masked`.
- **Issue #56**: when the ground truth or the predictions had no test pixel
  deeper than `depth_threshold`, `sdb train` printed precision, recall and F1
  as 0.000. They are undefined then, so they are now NaN, printed as `n/a`
  with a note saying why.
- **Issue #34**: rerunning `sdb split` into the same `output_dir` augmented the
  already-augmented patches again (`*_rot90_rot90.tif`, ...). Only the source
  patches are augmented now.
- **Issue #35**: `sdb split` exited with status 0 even when every scene
  failed, so the error only surfaced later as "No training patches found" in
  `sdb train`. It now prints a summary of found, failed and skipped scenes
  and exits with status 1 (`dl_bathy.split.run` raises `NoPatchesError`)
  when no scene produced patches.
- `predict_from_geoai_datacubes` returns NaN for pixels that the scene's
  Scene Classification Layer does not mark as water (new `water_only`
  option, on by default). The demo notebook now says that the model covers
  0-20 m and that deep open water shows up as about 0 m.
- **Issue #29**: with the default `DATA_SOURCE: geoai_datacubes`, every scene
  failed acquisition-time parsing (`Sentinel-2_full_size.tiff` has no date in
  its name), so `sdb split` wrote no patches. The time is now read from the
  `userdata.json` that geoai-datacubes writes next to each scene, so tides are
  also computed at the actual acquisition time rather than midnight.
- **Issue #30**: `inference.predict_depth` (and so `predict_from_geoai_datacubes`)
  fed raw reflectance to the model, which was trained on
  `log(band/10000 + 1e-6)`. It now applies the same normalisation as
  `sdb split` (shared `dl_bathy.bands.normalize_reflectance`), sets invalid
  values to 0 as in training, and returns NaN for those pixels. Pass
  `normalize=False` for bands that are already normalised.
- **Issue #31**: `predict_depth` ran the whole scene in one forward pass, which
  failed for any height or width that is not a multiple of 16 (DeepLabV3+'s
  output stride) and needs far more memory than a laptop or Colab CPU has
  for a fetched AOI. It now reflect-pads to a multiple of 16 and runs on
  overlapping tiles (`tile_size`, default the training patch size of 512;
  `overlap`, default 64) blended at the seams.
- **Issue #32**: the demo notebook's install cell asked pip for `DL_bathy`, which
  is not on PyPI, so it installed nothing; it now installs
  `dl-bathy[inference,fetch]` from GitHub. Its maps used a 0 to 25 m colour
  scale for depths that are negative, and its metrics now cover only the
  0-20 m range the released weights were trained on.
- **Issue #33**: without validation patches (as with the default
  `val_ratio: 0.0`), `sdb train` never saved `best_model.pth` and stopped
  after `earlystop_patience` epochs, so the test step then failed. It now
  warns, steps the learning-rate scheduler on the training metric instead,
  skips early stopping and saves the final model as `best_model.pth`.
- `filter_valid_patches` no longer drops 13-band patches (without a mask band)
  because of an undefined variable.
- README parameter tables now show the actual defaults (`strict_nan_filter`,
  split ratios, patch size and stride, output folder names).
- **Issue #10 (train/test leakage)**: `consistent_spatial` splitting now hashes
  on blocks twice the patch size and buffers out any patch that would
  straddle a block boundary, so overlapping sliding-window patches can no
  longer land on both sides of the train/test split. This changes which
  patches end up in train/val/test versus prior releases, and discards
  patches near block edges (none with no overlap, i.e. `stride >=
  patch_size`).
- **Issue #11 (`depth_type` inverted)**: `depth_type="deep"` now actually
  trains on pixels deeper than `depth_threshold` (and `"shallow"` on
  shallower ones) — the two branches' comparisons were swapped, so each
  setting silently trained on the other's pixel range. A second, separate
  copy of this same inverted comparison in the test-set scoring loop was
  missed in the original fix and is now corrected too, so test metrics are
  computed on the same depth range the model was trained on.
- **Issue #12**: the test metrics always left out pixels deeper than 20 m,
  whatever `depth_threshold` was. The new `train` setting `test_depth_range`
  (default `[-20, -0.01]`, the old range) sets the scored range, and
  `prediction_metrics.csv` and `within3m_prediction_metrics.csv` record it in
  the new `Eval_depth_min_m` and `Eval_depth_max_m` columns.
- **Default depth range is 0–20 m again**: `depth_type` now defaults to
  `"shallow"` and `test_depth_range` to `[-20, -0.01]`. Before the #11 fix,
  the default `"deep"` actually trained on 0–20 m, which is what the paper
  and the released weights use (their `metadata.json` still says `"deep"`).
  After the fix, keeping `"deep"` as the default would have switched default
  training to 20–30 m. Configs that set `depth_type: deep` explicitly now
  train on depths greater than `depth_threshold`; use `shallow` to get the
  old behaviour.
- `predict_depth`'s docstring said output depth is "positive = deeper";
  it's actually negative (0 = sea level), matching the LiDAR target
  convention used everywhere else in the pipeline.
- **Issue #37**: `set_reproducible` turned on `cudnn.benchmark`, which lets
  cuDNN choose different (possibly non-deterministic) algorithms on each run
  and undid `cudnn.deterministic`. It is now off, and
  `torch.use_deterministic_algorithms(True, warn_only=True)` is set.

## `v0.1.0` — 2026-08-08

First public release, archived on Zenodo at
[`10.5281/zenodo.21845182`](https://doi.org/10.5281/zenodo.21845182).

### Added

- **Two-step training pipeline** for satellite-derived bathymetry from
  Sentinel-2 imagery.
  - `step1_splitting.py` — alignment of downloaded Sentinel-2 scenes with
    the reference depths, cloud and water masks from Google Earth Engine,
    tidal correction with the DTU23 model, patch generation,
    spatially-consistent train / val / test splits, class balancing, and
    rotation / flip augmentation.
  - `step2_train_test.py` — DeepLabV3+ training with a choice of
    encoder backbones (ConvNeXt-Large, EfficientNet-B4, ResNet-50/101),
    mask-aware losses (SSE / MAE / RPE / SWF / Huber), early stopping,
    LR scheduling, multi-GPU training, and full-scene test inference
    with per-patch GeoTIFF predictions and visualisations.
- **Extracted loss module** — `losses.py` isolates the depth-weighted,
  validity-mask-aware loss functions so they can be unit-tested
  without the heavy geospatial dependencies used elsewhere. Unit
  tests live in `test_losses.py`.
- **Public ground-truth subset** — `Ground_Truth.zip` ships the Great
  Barrier Reef bathymetric reference used by the demo path, so the
  training pipeline can be inspected end-to-end without needing
  proprietary LiDAR data.
- **JOSS paper draft** — `paper.md` and `paper.bib`.
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
