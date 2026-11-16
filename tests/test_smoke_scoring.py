"""The smoke test scores predictions only over the 0-20 m training range (#57)."""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "_run_predict", Path(__file__).resolve().parents[1] / "smoke-tests" / "_run_predict.py")
run_predict = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_predict)


def test_ground_truth_deeper_than_20_m_is_not_scored():
    gt = np.array([-1.0, -10.0, -20.0, -25.0, -30.0, np.nan, 0.5], dtype="float32")
    pred = np.array([-2.0, -12.0, -19.0, -5.0, -5.0, -3.0, -1.0], dtype="float32")
    scores = run_predict._depth_errors(pred, gt)
    assert scores["scored_range_m"] == [-20.0, 0.0]
    assert scores["overlapping_valid_pixels"] == 3        # -1, -10 and -20 m only
    err = np.array([-1.0, -2.0, 1.0])
    assert np.isclose(scores["rmse_m"], np.sqrt(np.mean(err ** 2)))
    assert np.isclose(scores["mae_m"], 4.0 / 3)
    assert np.isclose(scores["bias_m"], -2.0 / 3)


def test_no_pixels_in_range_gives_counts_but_no_metrics():
    scores = run_predict._depth_errors(np.array([-5.0]), np.array([-25.0]))
    assert scores["overlapping_valid_pixels"] == 0
    assert "rmse_m" not in scores
