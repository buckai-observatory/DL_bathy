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


def test_depth_defaults_train_and_score_0_to_20_m():
    # The paper and the released weights cover 0-20 m. The defaults must
    # train on that range (depth_type="shallow" keeps |y| <= 20) and score
    # the same range, or test-set scoring silently sees zero pixels.
    cfg = load_config(None, "train")
    assert cfg["depth_type"] == "shallow" and cfg["depth_threshold"] == 20
    assert cfg["test_depth_range"] == [-20, -0.01]


def test_sdb_train_with_default_depth_settings_scores_0_to_20_m(scene):
    pd = pytest.importorskip("pandas")
    main(["split", "--config", str(scene)])
    assert main(["train", "--config", str(scene)]) == 0
    full = pd.read_csv(scene.parent / "Model_output" / "predictions" / "prediction_metrics.csv")
    assert len(full) > 0
    assert (full["GT_min_depth_m"] >= -20).all() and (full["GT_max_depth_m"] <= -0.01).all()


def test_sdb_train_scores_only_test_depth_range_and_records_it(scene):
    pd = pytest.importorskip("pandas")
    cfg = yaml.safe_load(scene.read_text())
    cfg["train"]["depth_type"] = "shallow"  # so it overlaps the overridden range below
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


def test_sdb_train_test_time_depth_type_matches_training_time_filter(scene):
    # Regression test: the test-set scoring loop had its own copy of the
    # depth_type filter that was still inverted after issue #11's fix to
    # BathyDataset.__getitem__ -- test-set metrics were scoring the
    # opposite depth range from what the model was actually trained on.
    pd = pytest.importorskip("pandas")
    cfg = yaml.safe_load(scene.read_text())
    cfg["train"]["depth_type"] = "deep"
    cfg["train"]["depth_threshold"] = 10
    cfg["train"]["test_depth_range"] = [-25, -0.01]  # wide enough not to constrain on its own
    scene.write_text(yaml.safe_dump(cfg))
    main(["split", "--config", str(scene)])
    assert main(["train", "--config", str(scene)]) == 0
    preds = scene.parent / "Model_output" / "predictions"

    full = pd.read_csv(preds / "prediction_metrics.csv")
    assert len(full) > 0
    assert (full["GT_max_depth_m"] <= -10).all()  # 'deep' keeps only |y| >= threshold


def test_sdb_train_without_validation_patches_runs_all_epochs_and_saves_best(scene):
    # Regression test for issue #33: with an empty validation folder every
    # val metric is NaN, so best_model.pth was never saved, early stopping
    # fired after earlystop_patience epochs, and test inference then failed.
    cfg = yaml.safe_load(scene.read_text())
    cfg["train"].update(num_epochs=3, earlystop_patience=1)
    scene.write_text(yaml.safe_dump(cfg))
    main(["split", "--config", str(scene)])
    for f in (scene.parent / "Data" / "valid").glob("*.tif"):
        f.unlink()
    assert main(["train", "--config", str(scene)]) == 0
    out = scene.parent / "Model_output"
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["num_epochs_trained"] == 3
    assert (out / "best_model.pth").exists()
    assert len(list((out / "predictions").glob("*_pred.tif"))) == len(list((scene.parent / "Data" / "test").glob("*.tif")))


def test_set_reproducible_turns_off_cudnn_benchmark(monkeypatch):
    # cudnn.benchmark=True picks algorithms per run and undoes
    # cudnn.deterministic=True, so set_reproducible must turn it off (#37).
    from dl_bathy.train import set_reproducible
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        set_reproducible(0)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def test_binary_metrics_are_nan_not_zero_for_a_single_depth_class():
    """All test pixels shallower than depth_threshold: precision/recall are undefined (#56)."""
    from dl_bathy.train import calculate_binary_metrics
    y_true = np.array([-1.0, -2.0, -2.5, np.nan])
    y_pred = np.array([-1.2, -1.8, -2.9, -1.0])
    cm = calculate_binary_metrics(y_true, y_pred, threshold=3)
    assert (cm["TP"], cm["TN"], cm["FP"], cm["FN"]) == (0, 3, 0, 0)
    assert np.isnan(cm["Precision"]) and np.isnan(cm["Recall"]) and np.isnan(cm["F1_Score"])
    assert cm["Accuracy"] == pytest.approx(1.0)


def test_binary_metrics_unchanged_when_both_classes_are_present():
    from dl_bathy.train import calculate_binary_metrics
    y_true = np.array([-1.0, -5.0, -6.0, -2.0, -7.0])
    y_pred = np.array([-1.0, -5.0, -1.0, -4.0, -8.0])
    cm = calculate_binary_metrics(y_true, y_pred, threshold=3)
    assert (cm["TP"], cm["TN"], cm["FP"], cm["FN"]) == (2, 1, 1, 1)
    assert cm["Precision"] == 2 / (3 + 1e-8)
    assert cm["Recall"] == 2 / (3 + 1e-8)
    assert cm["Accuracy"] == 3 / (5 + 1e-8)
    # A defined zero stays zero: deep pixels exist but none is predicted deep.
    zero = calculate_binary_metrics(np.array([-5.0, -1.0]), np.array([-1.0, -5.0]), threshold=3)
    assert zero["Precision"] == 0 and zero["Recall"] == 0


# --- Small clean-ups (issue #38) ---------------------------------------------

def test_load_model_returns_no_unused_grad_scaler(recwarn):
    from dl_bathy.train import load_model
    out = load_model("resnet50", torch.device("cpu"), lr=1e-3, reduce_lr_patience=1,
                     output_activation=None, encoder_weights=None)
    assert len(out) == 3                      # model, optimizer, scheduler
    assert not [w for w in recwarn if "GradScaler" in str(w.message)]


def test_dataset_raises_for_unreadable_patch(scene, tmp_path):
    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"not a tif")
    with pytest.raises(RuntimeError, match="broken.tif"):
        BathyDataset([str(bad)], load_config(scene, "train"))[0]


def test_filter_valid_patches_reports_unreadable_files(scene, tmp_path, capsys):
    from dl_bathy.train import filter_valid_patches
    main(["split", "--config", str(scene)])
    good = sorted(str(f) for f in (scene.parent / "Data" / "train").glob("*.tif"))
    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"not a tif")
    capsys.readouterr()
    assert filter_valid_patches(good + [str(bad)], 0.0) == good
    assert "Skipped 1 unreadable patch file(s)" in capsys.readouterr().out
    assert filter_valid_patches(good, 0.0) == good
    assert "unreadable" not in capsys.readouterr().out


def test_dataloaders_pin_memory_only_on_cuda(scene, monkeypatch):
    import dl_bathy.train as train
    pinned = []

    def recording_loader(*args, **kwargs):
        pinned.append(kwargs["pin_memory"])
        return torch.utils.data.DataLoader(*args, **kwargs)

    monkeypatch.setattr(train, "DataLoader", recording_loader)
    main(["split", "--config", str(scene)])
    assert main(["train", "--config", str(scene)]) == 0
    assert pinned and all(p == torch.cuda.is_available() for p in pinned)
