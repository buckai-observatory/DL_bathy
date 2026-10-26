"""Tiny synthetic Sentinel-2 L2A / LiDAR scene for tests (no network or tide model needed)."""
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

CRS = "EPSG:32755"
TRANSFORM = from_origin(300000, 8400000, 10, 10)
SIZE = 512  # large enough that the consistent_spatial split's buffer zones
            # (see dl_bathy.split.assign_split, issue #10) still leave
            # patches for all of train/val/test
DATE = "2023-05-01"
CLOUD = (slice(100, 120), slice(20, 40))   # 20 x 20 px cloud


def write_tif(path, data, dtype, nodata=None):
    data = data if data.ndim == 3 else data[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=data.shape[0], dtype=dtype, crs=CRS, transform=TRANSFORM, nodata=nodata) as dst:
        dst.write(data.astype(dtype))


@pytest.fixture
def scene(tmp_path):
    """Write a synthetic 13-band L2A scene, LiDAR and a config; return the config path."""
    rng = np.random.default_rng(0)
    _, cols = np.mgrid[0:SIZE, 0:SIZE]
    depth = -1.0 - 24.0 * cols / (SIZE - 1)                 # 1 m (west) to 25 m (east)
    depth[:8, :] = -9999                                    # LiDAR gap (nodata)
    write_tif(tmp_path / "lidar.tif", depth, "float32", nodata=-9999)

    # Bands 1-12: reflectance darkening with depth; band 13: SCL (6 = water, 9 = cloud)
    refl = 400 + 1200 * np.exp(depth.clip(-30, 0) / 8.0)
    bands = [refl * (1 + 0.05 * b) + rng.normal(0, 20, refl.shape) for b in range(12)]
    scl = np.full((SIZE, SIZE), 6.0)
    scl[CLOUD] = 9
    write_tif(tmp_path / "s2" / f"S2_{DATE}_02-30-00_scene.tif", np.stack(bands + [scl]).clip(1, None), "uint16")

    config = {
        "split": {
            "DATA_SOURCE": "local_scenes", "LOCAL_SCENES_DIR": "s2",
            "lidar_file": "lidar.tif", "output_dir": "Data", "TIDE_SOURCE": "none",
            "patch_size": 64, "stride_value": 32, "train_ratio": 0.6, "val_ratio": 0.2, "dpi": 50,
        },
        "train": {
            "TrainingDataPath": "Data", "LiDAR_Model": "Model_output",
            "Model_mode": "resnet50", "encoder_weights": None,
            "num_epochs": 1, "batch_size": 2, "num_workers": 0, "dpi": 50,
            "train_source": "train", "valid_source": "valid", "test_source": "test",
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path
