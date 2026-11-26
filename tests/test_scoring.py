"""Errors by ground-truth depth band (#87)."""
import numpy as np
import pytest

from dl_bathy.scoring import DEPTH_BANDS, depth_band_errors, format_depth_band_table


def by_band(rows):
    return {("all" if r["band_m"] == "all" else tuple(r["band_m"])): r for r in rows}


def test_bands_split_the_scored_pixels_and_add_up_to_the_overall_bias():
    truth = np.array([-0.5, -1.5, -3.0, -7.0, -12.0, -20.0, -25.0, np.nan, -4.0])
    pred = np.array([-1.0, -2.0, -2.0, -6.0, -10.0, -16.0, -5.0, -3.0, np.nan])
    rows = by_band(depth_band_errors(pred, truth))
    # -25 m (deeper than 20 m) and the two pixels with a NaN are not scored.
    assert rows["all"]["pixels"] == 6
    assert [rows[b]["pixels"] for b in DEPTH_BANDS] == [2, 1, 1, 1, 1]
    assert rows[(15, 20)]["bias_m"] == pytest.approx(4.0)       # -20 m is in the deepest band
    assert rows[(0, 2)]["bias_m"] == pytest.approx(-0.5)        # too deep in shallow water
    assert rows["all"]["bias_m"] == pytest.approx(np.mean([-0.5, -0.5, 1.0, 1.0, 2.0, 4.0]))
    assert sum(rows[b]["adds_to_bias_m"] for b in DEPTH_BANDS) == pytest.approx(rows["all"]["bias_m"])
    assert sum(rows[b]["share"] for b in DEPTH_BANDS) == pytest.approx(1.0)
    assert "adds_to_bias_m" not in rows["all"]


def test_metrics_of_one_band():
    truth = np.array([-6.0, -7.0, -8.0])
    pred = np.array([-5.0, -9.0, -8.0])
    rows = by_band(depth_band_errors(pred, truth))
    band = rows[(5, 10)]
    assert band["bias_m"] == pytest.approx(-1.0 / 3)
    assert band["median_m"] == pytest.approx(0.0)
    assert band["mae_m"] == pytest.approx(1.0)
    assert band["rmse_m"] == pytest.approx(np.sqrt(5.0 / 3))


def test_empty_band_has_a_count_but_no_metrics():
    rows = by_band(depth_band_errors(np.array([-1.0]), np.array([-1.2])))
    assert rows[(10, 15)] == {"band_m": [10, 15], "pixels": 0}
    assert rows["all"]["pixels"] == 1


def test_custom_bands_and_shape_check():
    rows = depth_band_errors(np.full((2, 2), -3.0), np.full((2, 2), -4.0), bands=((0, 5),))
    assert rows[0]["pixels"] == 4 and rows[0]["bias_m"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        depth_band_errors(np.zeros(3), np.zeros(4))


def test_table_lists_every_band_and_the_total():
    table = format_depth_band_table(depth_band_errors(np.array([-1.0, -12.0]), np.array([-1.5, -14.0])))
    lines = table.splitlines()
    assert len(lines) == 1 + len(DEPTH_BANDS) + 1
    assert lines[1].split()[0] == "0-2" and lines[-1].split()[0] == "all"
    assert "+0.50" in lines[1] and "+2.00" in lines[4]
