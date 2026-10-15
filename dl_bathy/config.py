"""
Configuration for DL_bathy.

Settings live in one YAML file with a ``split`` section (patch preparation,
``sdb split``) and a ``train`` section (training and inference, ``sdb train``).
Any key left out takes the default below, which reproduces the USER
CONFIGURATION blocks of the original ``step1_splitting.py`` /
``step2_train_test.py`` scripts. Relative paths are resolved against the
folder that contains the config file.
"""

import copy
import os

import yaml

SPLIT_DEFAULTS = {
    # --- Sentinel-2 processing level ---
    "Sentinel2_level": "L2A",          # 'L1C' or 'L2A' (geoai-datacubes path is L2A-only)

    # --- Data acquisition ---
    # 'geoai_datacubes': fetch S2 L2A scenes over AOI + TIME_RANGE (needs geoai-datacubes)
    # 'local_scenes'   : read pre-downloaded 13-band L2A GeoTIFFs (bands 1-12 + SCL)
    "DATA_SOURCE": "geoai_datacubes",
    "AOI": [145.14, -14.56, 145.66, -14.26],   # lon_min, lat_min, lon_max, lat_max (WGS84)
    "TIME_RANGE": ["2020-06-01", "2020-09-30"],
    "MAX_CLOUD_COVERAGE": 0.20,
    "FETCH_RESOLUTION_M": 10,
    "fetch_dir": "sentinel_images_L2A_fetched",   # where geoai-datacubes writes scenes
    "LOCAL_SCENES_DIR": "sentinel_images_L2A",    # folder or glob for 'local_scenes'
    "l1c_dir": "Dongsha_s2_img",                  # legacy L1C scenes (*B02.tif)

    # --- Inputs / outputs ---
    "lidar_file": "your_lidar_reference.tif",     # LiDAR reference GeoTIFF (ground truth)
    "output_dir": "Data",                         # train/valid/test patch folders
    "cloud_mask_dir": "Cloud_Masks",              # cached water masks rev_<YYYYMMDD>.tif

    # --- Tide correction ---
    # 'pytmd' (EOT20 via pyTMD), 'custom' (one GeoTIFF per scene in CUSTOM_TIDE_DIR) or 'none'
    "TIDE_SOURCE": "pytmd",
    "TIDE_MODEL_NAME": "EOT20",
    "TIDE_MODEL_DIRECTORY": None,                 # None = pyTMD's default cache directory
    "CUSTOM_TIDE_DIR": "custom_tide",
    "tide_cache_dir": "PyTMD_tide",               # cached pyTMD tide grids

    # --- Split ---
    "train_ratio": 1.0,
    "val_ratio": 0.0,
    "split_strategy": "consistent_spatial",       # or 'per_image'

    # --- Patches ---
    "patch_size": 512,
    "stride_value": 256,
    "valid_pixel_threshold": 0.1,
    "strict_nan_filter": False,

    # --- Depth ---
    "rough_mean_sea_surface": 0,
    "deepestDepth2Train": -30,

    # --- Balancing and augmentation ---
    "apply_balance": False,
    "apply_augmentation": True,
    "augmentation_rotations": [90, 180, 270],
    "augmentation_flip": True,

    # --- Reflectance normalisation ---
    "Logcube": True,
    "band_normalization": 10000,
    "Epsilon": 1e-6,

    "seed": 42,
    "dpi": 300,
}

TRAIN_DEFAULTS = {
    "TrainingDataPath": "Data",       # Patch dataset produced by `sdb split`
    "LiDAR_Model": "Model_output",    # Output folder for models, logs and plots
    "Model_mode": "tu-convnext_large",
    "encoder_weights": "imagenet",    # null trains the encoder from scratch (no download)
    "depth_scale": 1,
    "min_valid_ratio": 0.1,
    "min_valid_ratio_for_val": 0.1,
    "grad_clip_norm": 5.0,
    "output_activation": "none",
    "clamp_output": False,
    "num_epochs": 1000,
    "batch_size": 16,
    "lr": 1e-4,
    "earlystop_patience": 10,
    "reduce_lr_patience": 5,
    "depth_threshold": 20,
    "depth_type": "deep",
    "loss_type": "swf",
    "SWF_beta": 5.0,
    "SWF_Z0": 10.0,
    "scaling_method": "Log_band",
    "lambda_l1": 0,
    "lambda_l2": 0,
    "lambda_custom": 0,
    "train_source": "train_aug",
    "valid_source": "valid_aug",
    "test_source": "test",
    "num_workers": None,              # None -> SLURM_CPUS_PER_TASK, or 16
    "seed": 42,
    "dpi": 300,
}

PATH_KEYS = {
    "split": ["fetch_dir", "LOCAL_SCENES_DIR", "l1c_dir", "lidar_file", "output_dir",
              "cloud_mask_dir", "TIDE_MODEL_DIRECTORY", "CUSTOM_TIDE_DIR", "tide_cache_dir"],
    "train": ["TrainingDataPath", "LiDAR_Model"],
}


def _merge(defaults, user, section):
    unknown = set(user) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{section}' section: {', '.join(sorted(unknown))}")
    merged = copy.deepcopy(defaults)
    merged.update(user)
    return merged


def load_config(path=None, section="split"):
    """Return the settings for *section* ('split' or 'train') as a dict.

    With no *path*, the defaults are returned and relative paths are taken
    relative to the current working directory.
    """
    if section not in ("split", "train"):
        raise ValueError("section must be 'split' or 'train'")
    defaults = SPLIT_DEFAULTS if section == "split" else TRAIN_DEFAULTS
    base_dir = os.getcwd()
    user = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown_sections = set(data) - {"split", "train"}
        if unknown_sections:
            raise ValueError(f"Unknown section(s) in {path}: {', '.join(sorted(unknown_sections))}")
        user = data.get(section) or {}
        base_dir = os.path.dirname(os.path.abspath(path))
    cfg = _merge(defaults, user, section)
    for key in PATH_KEYS[section]:
        if cfg[key] is not None:
            cfg[key] = os.path.join(base_dir, os.path.expanduser(cfg[key]))
    cfg["base_dir"] = base_dir
    return cfg


def example_config():
    """Return an example YAML configuration listing every setting with its default."""
    return yaml.safe_dump({"split": SPLIT_DEFAULTS, "train": TRAIN_DEFAULTS},
                          sort_keys=False, default_flow_style=None)
