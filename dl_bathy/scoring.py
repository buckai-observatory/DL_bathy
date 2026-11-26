"""Scoring predicted depths against ground truth, overall and by depth band.

A single bias over 0-20 m hides how the error changes with depth: the released
models predict too deep in very shallow water and too shallow in deeper water
(issue #87). The demo notebook and the smoke test therefore report the errors
per ground-truth depth band as well. Needs only NumPy.
"""
import numpy as np

#: Ground-truth depth bands in metres (positive down) used to break down the errors.
DEPTH_BANDS = ((0, 2), (2, 5), (5, 10), (10, 15), (15, 20))


def depth_band_errors(predicted, ground_truth, bands=DEPTH_BANDS):
    """Errors of predicted depths against ground truth, per depth band and overall.

    Parameters
    ----------
    predicted, ground_truth : array-like
        Elevations in metres on the same grid, negative below sea level.
    bands : sequence of (shallow, deep)
        Ground-truth depth bands in metres, positive down, not overlapping.
        Each band includes its shallow end; the deepest band also includes its
        deep end, so the default bands cover 0-20 m inclusive.

    Returns
    -------
    list of dict
        One row per band, then one for all scored pixels. Each row has
        ``band_m`` (``[shallow, deep]``, or ``"all"``) and ``pixels``, and,
        when it has pixels, ``share`` (of all scored pixels), ``bias_m``,
        ``median_m``, ``mae_m`` and ``rmse_m``. Band rows also have
        ``adds_to_bias_m``: the band's contribution to the overall bias, so the
        band contributions add up to the overall ``bias_m``.

    Notes
    -----
    The error is prediction minus ground truth, so a positive bias means the
    predictions are too shallow. A pixel is scored when both values are finite
    and its ground-truth depth lies in one of the bands.
    """
    pred = np.asarray(predicted, dtype="float64")
    truth = np.asarray(ground_truth, dtype="float64")
    if pred.shape != truth.shape:
        raise ValueError(f"predicted {pred.shape} and ground_truth {truth.shape} differ in shape")
    pred, truth = pred.ravel(), truth.ravel()
    depth = -truth
    deepest = max(deep for _, deep in bands)
    finite = np.isfinite(pred) & np.isfinite(truth)
    in_band = [finite & (depth >= shallow) & ((depth <= deep) if deep == deepest else (depth < deep))
               for shallow, deep in bands]
    scored = np.logical_or.reduce(in_band)
    total = int(scored.sum())
    error = pred - truth

    def row(band, sel, is_band):
        e = error[sel]
        out = {"band_m": band, "pixels": int(e.size)}
        if e.size:
            out.update(share=e.size / total, bias_m=float(e.mean()), median_m=float(np.median(e)),
                       mae_m=float(np.abs(e).mean()), rmse_m=float(np.sqrt(np.mean(e ** 2))))
            if is_band:
                out["adds_to_bias_m"] = float(e.sum() / total)
        return out

    rows = [row([shallow, deep], sel, True) for (shallow, deep), sel in zip(bands, in_band)]
    rows.append(row("all", scored, False))
    return rows


def format_depth_band_table(rows):
    """The rows of :func:`depth_band_errors` as an aligned text table."""
    lines = [f"{'depth (m)':>10} {'pixels':>9} {'share':>6} {'bias':>7} {'median':>7} "
             f"{'MAE':>6} {'RMSE':>6} {'adds to bias':>13}"]
    for r in rows:
        band = "all" if r["band_m"] == "all" else "{:g}-{:g}".format(*r["band_m"])
        if not r["pixels"]:
            lines.append(f"{band:>10} {0:>9,}")
            continue
        adds = f"{r['adds_to_bias_m']:>+13.2f}" if "adds_to_bias_m" in r else ""
        lines.append(f"{band:>10} {r['pixels']:>9,} {r['share']:>6.1%} {r['bias_m']:>+7.2f} "
                     f"{r['median_m']:>+7.2f} {r['mae_m']:>6.2f} {r['rmse_m']:>6.2f} {adds}".rstrip())
    return "\n".join(lines)
