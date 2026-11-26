import random

import numpy as np
import pytest
import rasterio
import yaml

from dl_bathy.cli import main
from dl_bathy.patch_io import read_patch_tif
from dl_bathy.split import (assign_split, augment_folder_tifs, balance_metadata_by_depth,
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
    assert (data / "cloud_removal_counts.csv").read_text().splitlines() == [
        "Date,Non_water_pixels_masked", "20230501,400"]


def test_rerunning_split_does_not_augment_augmented_patches(scene):
    """A second `sdb split` into the same output_dir must not rotate the _rot*/_flip files again (#34)."""
    data = scene.parent / "Data"
    assert main(["split", "--config", str(scene)]) == 0
    first = sorted(p.name for p in (data / "train_aug").glob("*.tif"))
    assert main(["split", "--config", str(scene)]) == 0
    second = sorted(p.name for p in (data / "train_aug").glob("*.tif"))
    assert second == first
    assert not any("_rot90_" in name or "_flip_" in name for name in second)
    assert len(second) == 5 * len(list((data / "train").glob("*.tif")))


def test_augment_folder_tifs_only_augments_the_given_files(tmp_path):
    from tests.conftest import write_tif
    for name in ("a.tif", "b.tif"):
        write_tif(tmp_path / name, np.arange(16, dtype="float32").reshape(4, 4), "float32")
    augment_folder_tifs(str(tmp_path), rotations=(90,), flip=True, files=["a.tif"])
    assert sorted(p.name for p in tmp_path.glob("*.tif")) == ["a.tif", "a_flip.tif", "a_rot90.tif", "b.tif"]
    with rasterio.open(tmp_path / "a_rot90.tif") as src:
        np.testing.assert_array_equal(src.read(1), np.rot90(np.arange(16).reshape(4, 4)))


def test_custom_tide_grid_is_subtracted_from_depths(scene):
    # A +1.5 m tide (sea 1.5 m above mean sea level) makes every pixel 1.5 m
    # deeper below the sea surface, i.e. 1.5 m more negative (#47).
    import yaml
    from tests.conftest import write_tif, SIZE
    cfg = yaml.safe_load(scene.read_text())
    cfg["split"].update(TIDE_SOURCE="custom", CUSTOM_TIDE_DIR="tide", apply_augmentation=False)
    scene.write_text(yaml.safe_dump(cfg))
    # Acquisition 2023-05-01 02:30:00 -> tide_20230501_023000.tif
    write_tif(scene.parent / "tide" / "tide_20230501_023000.tif", np.full((SIZE, SIZE), 1.5), "float32")
    assert main(["split", "--config", str(scene)]) == 0
    patches = sorted((scene.parent / "Data").glob("*/*.tif"))
    assert patches
    valid = np.concatenate([y[m > 0] for _, y, m, _ in map(read_patch_tif, patches)])
    # LiDAR spans -1 m to -25 m; with the tide subtracted it spans -2.5 m to -26.5 m.
    assert -2.5 - 0.5 < valid.max() <= -2.5 + 1e-4
    assert valid.min() >= -26.5 - 1e-3


def test_geoai_datacubes_scene_uses_acquisition_time_from_userdata(scene):
    # geoai-datacubes writes <fetch_dir>/Sentinel-2_<date>_<id>/Sentinel-2_full_size.tiff
    # plus userdata.json. The file name has no date, so the time must come
    # from userdata.json; the custom tide grid is looked up by that time.
    import json
    import shutil
    import yaml
    from tests.conftest import write_tif, SIZE
    src = next((scene.parent / "s2").glob("*.tif"))
    folder = scene.parent / "s2" / "Sentinel-2_2023-05-01_S2A_55LCE_20230501_0_L2A"
    folder.mkdir()
    shutil.move(src, folder / "Sentinel-2_full_size.tiff")
    (folder / "userdata.json").write_text(json.dumps({"acquisitionDate": "2023-05-01T00:32:48.909000Z"}))
    cfg = yaml.safe_load(scene.read_text())
    cfg["split"].update(TIDE_SOURCE="custom", CUSTOM_TIDE_DIR="tide", apply_augmentation=False)
    scene.write_text(yaml.safe_dump(cfg))
    write_tif(scene.parent / "tide" / "tide_20230501_003248.tif", np.full((SIZE, SIZE), 1.5), "float32")

    assert main(["split", "--config", str(scene)]) == 0
    files = list((scene.parent / "Data" / "train").glob("*.tif"))
    assert files
    _, y, mask, _ = read_patch_tif(files[0])
    assert y[mask > 0].max() <= -1 + 1.5 + 1e-4
    # The date comes from the folder name, not from 'full' in the file name (#36).
    counts = (scene.parent / "Data" / "cloud_removal_counts.csv").read_text().splitlines()
    assert counts == ["Date,Non_water_pixels_masked", "20230501,400"]


def test_scene_acquisition_datetime_fallbacks(tmp_path):
    from datetime import datetime
    from dl_bathy.split import scene_acquisition_datetime
    assert scene_acquisition_datetime(str(tmp_path / "S2_2023-05-01_02-30-00_x.tif"), "L2A") == datetime(2023, 5, 1, 2, 30)
    folder = tmp_path / "Sentinel-2_2020-06-15_S2A_55LCE_20200615_0_L2A"
    assert scene_acquisition_datetime(str(folder / "Sentinel-2_full_size.tiff"), "L2A") == datetime(2020, 6, 15)
    with pytest.raises(ValueError, match="acquisition time"):
        scene_acquisition_datetime(str(tmp_path / "scene" / "image.tif"), "L2A")


def test_scenes_without_scl_band_are_rejected(tmp_path):
    from dl_bathy.split import _scl_water_mask_from_scene
    from tests.conftest import write_tif
    write_tif(tmp_path / "s.tif", np.ones((12, 4, 4)), "uint16")
    with pytest.raises(RuntimeError, match="13-band"):
        _scl_water_mask_from_scene(str(tmp_path / "s.tif"))


def test_pytmd_tide_grid_queries_lon_lat_for_projected_scene(tmp_path, monkeypatch):
    """Issue #14: a UTM scene must be sampled at WGS84 lon/lat, not at metres."""
    import sys
    import types
    from datetime import datetime

    from rasterio.warp import transform as warp_transform

    from dl_bathy.split import generate_tide_grid
    from tests.conftest import CRS, TRANSFORM, write_tif

    write_tif(tmp_path / "s.tif", np.ones((6, 6)), "uint16")       # 60 m x 60 m UTM scene
    with rasterio.open(tmp_path / "s.tif") as src:
        left, bottom, right, top = src.bounds
    xs, ys = warp_transform(CRS, "EPSG:4326", [left, right, right, left], [top, top, bottom, bottom])

    calls = []

    def tide_elevations(lons, lats, delta_time, crs=None, **kwargs):
        lons, lats = np.asarray(lons), np.asarray(lats)
        calls.append((lons, lats, crs))
        return 0.1 * lons + 0.01 * lats

    fake = types.ModuleType("pyTMD")
    fake.compute = types.SimpleNamespace(tide_elevations=tide_elevations)
    monkeypatch.setitem(sys.modules, "pyTMD", fake)
    monkeypatch.setitem(sys.modules, "h5py", types.ModuleType("h5py"))

    _, tide = generate_tide_grid(str(tmp_path / "s.tif"), datetime(2023, 5, 1, 2, 30),
                                 str(tmp_path / "tide"))

    (lons, lats, crs), = calls
    assert crs == 4326
    assert np.all((lons >= -180) & (lons <= 180) & (lats >= -90) & (lats <= 90))
    # The coarse 1/16 degree grid covers the scene and stays close to it
    assert lons.min() <= min(xs) and lons.max() >= max(xs)
    assert lats.min() <= min(ys) and lats.max() >= max(ys)
    assert lons.max() - lons.min() <= 0.25 and lats.max() - lats.min() <= 0.25
    assert lons.size <= 25

    # Interpolated back onto the pixel grid at each pixel's lon/lat
    rows, cols = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    px, py = rasterio.transform.xy(TRANSFORM, rows.ravel(), cols.ravel())
    plon, plat = warp_transform(CRS, "EPSG:4326", px, py)
    expected = (0.1 * np.array(plon) + 0.01 * np.array(plat)).reshape(6, 6)
    assert tide.shape == (6, 6)
    np.testing.assert_allclose(tide, expected, atol=1e-4)


def test_sdb_split_fails_when_every_scene_fails(scene, capsys):
    # Every scene is unreadable as a Sentinel-2 cube, so no patches are written:
    # `sdb split` must report it and exit non-zero instead of succeeding (#35).
    from tests.conftest import write_tif, SIZE
    src = next((scene.parent / "s2").glob("*.tif"))
    write_tif(src, np.ones((2, SIZE, SIZE)), "uint16")
    assert main(["split", "--config", str(scene)]) == 1
    captured = capsys.readouterr()
    assert "1 found, 0 with patches, 1 failed" in captured.out
    assert "No patches were written" in captured.err
    assert "Insufficient bands" in captured.err


def test_split_run_raises_when_no_scene_has_patches(scene):
    import yaml
    from dl_bathy.config import load_config
    from dl_bathy.split import NoPatchesError, run
    from tests.conftest import write_tif, SIZE
    src = next((scene.parent / "s2").glob("*.tif"))
    write_tif(src, np.ones((2, SIZE, SIZE)), "uint16")
    with pytest.raises(NoPatchesError, match="1 failed"):
        run(load_config(str(scene), "split"))


def test_sdb_split_succeeds_and_reports_when_only_some_scenes_fail(scene, capsys):
    from tests.conftest import write_tif, SIZE
    write_tif(scene.parent / "s2" / "S2_2023-05-02_02-30-00_broken.tif",
              np.ones((2, SIZE, SIZE)), "uint16")
    assert main(["split", "--config", str(scene)]) == 0
    out = capsys.readouterr().out
    assert "2 found, 1 with patches, 1 failed" in out
    assert "failed: " in out and "broken.tif: Insufficient bands" in out


def test_cloud_counts_keep_their_dates_when_an_earlier_scene_fails(scene):
    # The broken scene sorts first and fails before its mask is counted: the
    # one row must keep the date of the scene it was counted for (#36).
    from tests.conftest import write_tif, SIZE
    write_tif(scene.parent / "s2" / "S2_2023-04-30_02-30-00_broken.tif",
              np.ones((2, SIZE, SIZE)), "uint16")
    assert main(["split", "--config", str(scene)]) == 0
    counts = (scene.parent / "Data" / "cloud_removal_counts.csv").read_text().splitlines()
    assert counts == ["Date,Non_water_pixels_masked", "20230501,400"]


def test_split_overlays_colour_every_split_including_test(scene, monkeypatch):
    # Issue #38: the unbalanced overlay had no colour map for 'test', so
    # test patches were not drawn.
    import dl_bathy.split as split
    calls = []
    monkeypatch.setattr(split, "_split_overlay",
                        lambda lidar, metas, mask, cmaps, title, out_path, dpi: calls.append((metas, cmaps, title)))
    assert main(["split", "--config", str(scene)]) == 0
    assert calls
    for metas, cmaps, title in calls:
        assert set(cmaps) == set(metas) == {"train", "val", "test"}
        assert "Test" in title


def test_split_with_a_missing_lidar_file_fails_cleanly_before_creating_folders(scene, capsys):
    # Issue #58: a missing reference raster created Data/ folders and then
    # died with a raw RasterioIOError traceback.
    cfg = yaml.safe_load(scene.read_text())
    cfg["split"]["lidar_file"] = "does_not_exist.tif"
    scene.write_text(yaml.safe_dump(cfg))
    assert main(["split", "--config", str(scene)]) == 1
    assert "lidar_file not found" in capsys.readouterr().err
    assert not (scene.parent / "Data").exists()


# --- Issue #59 ----------------------------------------------------------------

@pytest.mark.parametrize("train_ratio, val_ratio, expected", [
    (1.0, 0.0, "train"), (0.0, 1.0, "val"), (0.0, 0.0, "test")])
def test_assign_split_keeps_every_patch_when_one_split_is_used(train_ratio, val_ratio, expected):
    # With a single split there is no train/test leakage to prevent, so no
    # patch near a block edge may be buffered out (None).
    coords = [(r, c, r + 64, c + 64) for r in range(0, 512, 32) for c in range(0, 512, 32)]
    assert {assign_split(xy, "consistent_spatial", train_ratio, val_ratio) for xy in coords} == {expected}


def test_assign_split_still_buffers_when_splits_are_mixed():
    coords = [(r, c, r + 64, c + 64) for r in range(0, 512, 32) for c in range(0, 512, 32)]
    assert None in {assign_split(xy, "consistent_spatial", 0.7, 0.15) for xy in coords}


def test_mean_depth_of_a_patch_without_finite_labels_is_nan_without_warning(recwarn):
    from dl_bathy.split import _mean_depth
    y = np.full((4, 4), np.nan, dtype="float32")
    assert np.isnan(_mean_depth(y, np.ones((4, 4))))
    assert not [w for w in recwarn if "empty slice" in str(w.message)]
    y[0, 0] = -3.0
    assert _mean_depth(y, np.ones((4, 4))) == -3.0


def test_patch_transform_matches_the_affine_composition(tmp_path):
    from rasterio.transform import Affine
    from dl_bathy.patch_io import save_patches_from_metadata
    base = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 8400000.0)
    patch = tmp_path / "p.npz"
    np.savez_compressed(patch, X=np.zeros((12, 8, 8), "float32"),
                        y=np.zeros((8, 8), "float32"), M=np.ones((8, 8), "float32"))
    save_patches_from_metadata([{"path": str(patch), "coords": (16, 24, 24, 32)}], str(tmp_path / "out"),
                               base, "EPSG:32755", "20200819", "scene.tif", "train")
    with rasterio.open(next((tmp_path / "out").glob("*.tif"))) as src:
        assert src.transform == Affine(10.0, 0.0, 500240.0, 0.0, -10.0, 8399840.0)


# --- Issue #77 ----------------------------------------------------------------

def test_split_without_tide_model_files_fails_before_fetching(scene, tmp_path, monkeypatch, capsys):
    # Without the EOT20 files, sdb split fetched the scenes first and then
    # failed with only a file path. It must now stop before any fetch or folder.
    pytest.importorskip("pyTMD")
    pytest.importorskip("h5py")
    import dl_bathy.split as split
    fetched = []
    monkeypatch.setattr(split, "find_sentinel_files", lambda cfg: fetched.append(1) or [])
    (tmp_path / "empty_tides").mkdir()
    cfg = yaml.safe_load(scene.read_text())
    cfg["split"].update(TIDE_SOURCE="pytmd", TIDE_MODEL_DIRECTORY="empty_tides")
    scene.write_text(yaml.safe_dump(cfg))
    assert main(["split", "--config", str(scene)]) == 1
    err = capsys.readouterr().err
    assert "sdb split: error:" in err
    for text in ("EOT20", "empty_tides", "https://doi.org/10.17882/79489",
                 "TIDE_SOURCE: none", "TIDE_SOURCE: custom"):
        assert text in err
    assert not fetched
    assert not (scene.parent / "Data").exists()


@pytest.mark.parametrize("missing", ["pyTMD", "h5py"])
def test_split_without_pytmd_fails_before_fetching(scene, monkeypatch, capsys, missing):
    # h5py: pyTMD reads EOT20 through h5netcdf, which fails without it.
    import builtins
    import types
    import dl_bathy.split as split
    real_import = builtins.__import__

    def no_pytmd(name, *args, **kwargs):
        if name == missing:
            raise ImportError(f"No module named '{missing}'")
        if name in ("pyTMD", "h5py"):
            return types.ModuleType(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pytmd)
    monkeypatch.setattr(split, "find_sentinel_files", lambda cfg: pytest.fail("fetched scenes"))
    cfg = yaml.safe_load(scene.read_text())
    cfg["split"]["TIDE_SOURCE"] = "pytmd"
    scene.write_text(yaml.safe_dump(cfg))
    assert main(["split", "--config", str(scene)]) == 1
    err = capsys.readouterr().err
    assert "requires the pyTMD and h5py packages" in err and missing in err
    assert not (scene.parent / "Data").exists()
