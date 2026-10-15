import os

import pytest
import yaml

from dl_bathy.cli import main
from dl_bathy.config import SPLIT_DEFAULTS, TRAIN_DEFAULTS, load_config


def test_defaults_match_original_scripts():
    cfg = load_config(None, "split")
    assert cfg["patch_size"] == 512 and cfg["stride_value"] == 256
    assert cfg["split_strategy"] == "consistent_spatial"
    assert cfg["DATA_SOURCE"] == "geoai_datacubes" and cfg["TIDE_SOURCE"] == "pytmd"
    tcfg = load_config(None, "train")
    assert tcfg["Model_mode"] == "tu-convnext_large" and tcfg["loss_type"] == "swf"


def test_relative_paths_resolve_against_config_folder(tmp_path):
    path = tmp_path / "c.yml"
    path.write_text(yaml.safe_dump({"split": {"lidar_file": "x.tif"}}))
    cfg = load_config(path, "split")
    assert cfg["lidar_file"] == os.path.join(tmp_path, "x.tif")
    assert cfg["output_dir"] == os.path.join(tmp_path, "Data")


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "c.yml"
    path.write_text(yaml.safe_dump({"split": {"patch_sise": 64}}))
    with pytest.raises(ValueError, match="patch_sise"):
        load_config(path, "split")


def test_init_config_writes_every_setting(tmp_path):
    out = tmp_path / "example.yml"
    assert main(["init-config", str(out)]) == 0
    data = yaml.safe_load(out.read_text())
    assert set(data["split"]) == set(SPLIT_DEFAULTS)
    assert set(data["train"]) == set(TRAIN_DEFAULTS)
    load_config(out, "train")  # the example must load cleanly
