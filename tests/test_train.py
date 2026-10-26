import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from dl_bathy.cli import main
from dl_bathy.config import load_config
from dl_bathy.patch_io import read_patch_tif
from dl_bathy.train import BathyDataset


def _shallowest_and_deepest_train_files(scene):
    """Split the scene, then return (shallowest_file, deepest_file) among
    the train patches, by mean valid depth -- the synthetic scene's depth
    gradient (1 m west to 25 m east) means these reliably differ."""
    main(["split", "--config", str(scene)])
    files = sorted((scene.parent / "Data" / "train").glob("*.tif"))
    means = []
    for f in files:
        _, y, m, _ = read_patch_tif(str(f))
        means.append(float(np.abs(y[m > 0]).mean()))
    order = np.argsort(means)
    return files[order[0]], files[order[-1]]


def test_dataset_depth_type_deep_keeps_only_pixels_beyond_threshold(scene):
    _, deep_file = _shallowest_and_deepest_train_files(scene)
    params = load_config(scene, "train")
    params.update(depth_threshold=10, depth_type="deep")
    _, y, m = BathyDataset([str(deep_file)], params)[0]
    # Regression test for issue #11: depth_type="deep" must keep pixels
    # deeper than the threshold (and exclude shallower ones), not the
    # reverse.
    assert (m > 0).sum() > 0
    assert torch.all(torch.abs(y[m > 0]) >= 10)


def test_dataset_depth_type_shallow_keeps_only_pixels_within_threshold(scene):
    shallow_file, _ = _shallowest_and_deepest_train_files(scene)
    params = load_config(scene, "train")
    params.update(depth_threshold=10, depth_type="shallow")
    _, y, m = BathyDataset([str(shallow_file)], params)[0]
    assert (m > 0).sum() > 0
    assert torch.all(torch.abs(y[m > 0]) <= 10)


def test_sdb_train_end_to_end_on_cpu(scene):
    main(["split", "--config", str(scene)])
    assert main(["train", "--config", str(scene)]) == 0
    out = scene.parent / "Model_output"
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["num_epochs_trained"] == 1 and meta["Model_mode"] == "resnet50"
    assert (out / "best_model.pth").exists()
    assert len(list((out / "predictions").glob("*_pred.tif"))) == len(list((scene.parent / "Data" / "test").glob("*.tif")))
    assert (out / "predictions" / "prediction_metrics.csv").exists()
    assert "Epoch 001" in (out / "console_output_step2.log").read_text()


def test_test_depth_range_defaults_to_previous_20m_cutoff():
    assert load_config(None, "train")["test_depth_range"] == [-20, -0.01]


def test_sdb_train_scores_only_test_depth_range_and_records_it(scene):
    pd = pytest.importorskip("pandas")
    cfg = yaml.safe_load(scene.read_text())
    cfg["train"]["test_depth_range"] = [-10, -0.01]
    scene.write_text(yaml.safe_dump(cfg))
    main(["split", "--config", str(scene)])
    assert main(["train", "--config", str(scene)]) == 0
    preds = scene.parent / "Model_output" / "predictions"

    full = pd.read_csv(preds / "prediction_metrics.csv")
    assert (full["GT_min_depth_m"] >= -10).all() and (full["GT_max_depth_m"] <= -0.01).all()
    assert (full["Eval_depth_min_m"] == -10).all() and (full["Eval_depth_max_m"] == -0.01).all()

    within3m = pd.read_csv(preds / "within3m_prediction_metrics.csv")
    assert (within3m["Eval_depth_min_m"] == -3).all() and (within3m["Eval_depth_max_m"] == -0.01).all()
