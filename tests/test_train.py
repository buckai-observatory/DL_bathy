import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from dl_bathy.cli import main
from dl_bathy.config import load_config
from dl_bathy.train import BathyDataset


def test_dataset_depth_type_masks_pixels_beyond_threshold(scene):
    main(["split", "--config", str(scene)])
    params = load_config(scene, "train")
    files = sorted((scene.parent / "Data" / "train").glob("*.tif"))
    params.update(depth_threshold=10, depth_type="deep")
    _, y, m = BathyDataset([str(f) for f in files], params)[len(files) - 1]
    # Documents current behaviour (issue #11): depth_type="deep" excludes pixels deeper than the threshold
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
