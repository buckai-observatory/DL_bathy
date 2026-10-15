import random

import numpy as np
import pytest
import rasterio

from dl_bathy.cli import main
from dl_bathy.patch_io import read_patch_tif
from dl_bathy.split import (assign_split, balance_metadata_by_depth,
                                extract_patches_no_nans_gen)


def test_patch_generator_filters_invalid_and_nan_patches():
    cube = np.ones((12, 8, 8))
    lidar = -np.ones((8, 8))
    lidar[0, 0] = np.nan
    valid = np.ones((8, 8), bool)
    valid[4:, 4:] = False                                   # bottom-right quarter invalid
    gen = extract_patches_no_nans_gen(cube, lidar, valid, patch_size=4, stride=4, min_valid_ratio=0.5)
    assert [c for *_, c in gen] == [(0, 0, 4, 4), (0, 4, 4, 8), (4, 0, 8, 4)]
    gen = extract_patches_no_nans_gen(cube, lidar, valid, patch_size=4, stride=4,
                                      min_valid_ratio=0.5, strict_nan_filter=True)
    assert [c for *_, c in gen] == [(0, 4, 4, 8), (4, 0, 8, 4)]


def test_consistent_spatial_split_is_deterministic_and_respects_ratios():
    coords = [(r, c, r + 64, c + 64) for r in range(0, 6400, 64) for c in range(0, 6400, 64)]
    first = [assign_split(c, "consistent_spatial", 0.6, 0.2) for c in coords]
    assert first == [assign_split(c, "consistent_spatial", 0.6, 0.2) for c in coords]
    frac = {s: first.count(s) / len(first) for s in ("train", "val", "test")}
    assert frac["train"] == pytest.approx(0.6, abs=0.05)
    assert frac["val"] == pytest.approx(0.2, abs=0.05)


def test_zero_ratios_send_everything_to_test():
    rng = random.Random(0)
    for strategy in ("consistent_spatial", "per_image"):
        assert {assign_split((r, 0, 0, 0), strategy, 0.0, 0.0, rng) for r in range(100)} == {"test"}


@pytest.mark.xfail(strict=True, reason=(
    "Known issue #10: with stride < patch_size, overlapping patches can be assigned "
    "to different splits, so the same pixels appear in train and test."))
def test_consistent_spatial_split_has_no_pixel_overlap_between_train_and_test():
    patch, stride, size = 512, 256, 4096
    owner = {}
    for r in range(0, size - patch + 1, stride):
        for c in range(0, size - patch + 1, stride):
            s = assign_split((r, c, r + patch, c + patch), "consistent_spatial", 0.7, 0.0)
            for rr in range(r, r + patch, stride):
                for cc in range(c, c + patch, stride):
                    owner.setdefault((rr, cc), set()).add(s)
    assert not any({"train", "test"} <= sets for sets in owner.values())


def test_balance_equalises_depth_bins():
    np.random.seed(0)
    meta = [{"mean_depth": -1.5}] * 10 + [{"mean_depth": -5.5}] * 2 + [{"mean_depth": -9.5}] * 4
    out = balance_metadata_by_depth(meta, bins=np.arange(-10, 1, 1))
    depths = [m["mean_depth"] for m in out]
    assert {d: depths.count(d) for d in set(depths)} == {-1.5: 4, -5.5: 4, -9.5: 4}


def test_sdb_split_end_to_end(scene):
    assert main(["split", "--config", str(scene)]) == 0
    data = scene.parent / "Data"
    counts = {d: len(list((data / d).glob("*.tif"))) for d in ("train", "valid", "test", "train_aug")}
    assert counts["train"] > 0 and counts["valid"] > 0 and counts["test"] > 0
    assert counts["train_aug"] == 5 * counts["train"]           # original + 3 rotations + flip

    X, y, mask, profile = read_patch_tif(next((data / "train").glob("*.tif")))
    assert X.shape == (12, 64, 64) and y.shape == (64, 64)
    with rasterio.open(next((data / "train").glob("*.tif"))) as src:
        assert src.count == 14 and src.tags()["SET"] == "train"
    valid_depth = y[mask > 0]
    assert np.all(valid_depth <= -1) and np.all(valid_depth >= -25.01)
    assert np.all(np.isfinite(X))                                 # log-scaled reflectance
    assert (data / "cloud_removal_counts.csv").read_text().splitlines()[1] == "20230501,400"


def test_custom_tide_grid_is_added_to_depths(scene):
    import yaml
    from tests.conftest import write_tif, SIZE
    cfg = yaml.safe_load(scene.read_text())
    cfg["split"].update(TIDE_SOURCE="custom", CUSTOM_TIDE_DIR="tide", apply_augmentation=False)
    scene.write_text(yaml.safe_dump(cfg))
    # Acquisition 2023-05-01 02:30:00 -> tide_20230501_023000.tif
    write_tif(scene.parent / "tide" / "tide_20230501_023000.tif", np.full((SIZE, SIZE), 1.5), "float32")
    assert main(["split", "--config", str(scene)]) == 0
    _, y, mask, _ = read_patch_tif(next((scene.parent / "Data" / "train").glob("*.tif")))
    valid = y[mask > 0]
    assert valid.max() <= -1 + 1.5 + 1e-4 and valid.min() >= -25 + 1.5 - 1e-3


def test_scenes_without_scl_band_are_rejected(tmp_path):
    from dl_bathy.split import _scl_water_mask_from_scene
    from tests.conftest import write_tif
    write_tif(tmp_path / "s.tif", np.ones((12, 4, 4)), "uint16")
    with pytest.raises(RuntimeError, match="13-band"):
        _scl_water_mask_from_scene(str(tmp_path / "s.tif"))
