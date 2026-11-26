"""Run inference.predict_from_geoai_datacubes end-to-end on the public GBR
demo AOI, optionally score it against Ground_Truth.zip, and write a JSON log.

Reads three env vars set by _common.sh:

    OUTDIR        scratch dir for fetched scenes (big; usually /tmp)
    LOGDIR        small JSON log dir (ignored by git; holds local paths)
    SCRIPT_NAME   id used for the log filename

This is the scripted, non-interactive equivalent of running
notebooks/dl_bathy_demo.ipynb end-to-end: it exercises the exact
reviewer-facing path described in MODEL_CARD.md and the README's
"Inference-only quickstart" -- fetch a real Sentinel-2 scene, download
real weights from the Hugging Face Hub, predict, and (if Ground_Truth.zip
is present) compare against the public ground truth. Not run in CI
(needs network + real weights); run manually or via smoke-tests/predict_demo_aoi.sh.

Usage:
    python smoke-tests/_run_predict.py
    python smoke-tests/_run_predict.py --repo-id org/other-weights --filename best_model.pth \
        --encoder-name tu-convnext_large
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dl_bathy.scoring import DEPTH_BANDS, depth_band_errors, format_depth_band_table  # noqa: E402

# Same AOI/time range as the README quickstart and the demo notebook --
# matches the public Ground_Truth.zip coverage.
DEFAULT_AOI = [145.14, -14.56, 145.66, -14.26]
DEFAULT_TIME_RANGE = ("2020-06-01", "2020-09-30")
DEFAULT_MAX_CLOUD_COVERAGE = 0.20
# The released weights were trained on 0-20 m (see MODEL_CARD.md) and the
# ground truth reaches -30 m, so, as in the demo notebook, only ground-truth
# depths in this range (m, negative below sea level) are scored.
SCORED_RANGE = (-20.0, 0.0)


def _depth_errors(depths: np.ndarray, gt_aligned: np.ndarray, scored_range=SCORED_RANGE) -> dict:
    """RMSE / MAE / bias of *depths* against *gt_aligned* over the pixels where
    both are finite and the ground truth lies within *scored_range*, plus the
    same errors per ground-truth depth band (``by_depth_band``). The bias
    changes strongly with depth (#87), so the overall bias alone misleads."""
    valid = (np.isfinite(gt_aligned) & np.isfinite(depths)
             & (gt_aligned >= scored_range[0]) & (gt_aligned <= scored_range[1]))
    result = {"scored_range_m": list(scored_range), "overlapping_valid_pixels": int(valid.sum())}
    if valid.any():
        err = depths[valid] - gt_aligned[valid]
        result.update({
            "rmse_m": float(np.sqrt(np.mean(err ** 2))),
            "mae_m": float(np.mean(np.abs(err))),
            "bias_m": float(np.mean(err)),
        })
        bands = [b for b in DEPTH_BANDS if -scored_range[1] <= b[0] and b[1] <= -scored_range[0]]
        if bands:
            result["by_depth_band"] = depth_band_errors(depths, gt_aligned, bands)[:-1]
    return result


def _score_against_ground_truth(depths: np.ndarray, meta: dict, outdir: str) -> dict | None:
    """Reproduce the demo notebook's GT comparison (cells 13-15, 0-20 m only). Returns
    None if Ground_Truth.zip isn't available rather than failing the run --
    the fetch+predict path is the thing being smoke-tested, not the GT
    comparison."""
    gt_zip = REPO_ROOT / "Ground_Truth.zip"
    if not gt_zip.exists():
        return None

    import rasterio
    from rasterio.warp import Resampling, reproject

    # The zip already holds a Ground_Truth/ folder.
    with zipfile.ZipFile(gt_zip) as zf:
        zf.extractall(outdir)
    extract_dir = os.path.join(outdir, "Ground_Truth")

    # Ground_Truth.zip holds train / val / test rasters; score against the
    # test one, as the demo notebook does.
    gt_candidates = sorted(glob.glob(os.path.join(extract_dir, "**", "*_test.tif"), recursive=True))
    if not gt_candidates:
        return {"error": "Ground_Truth.zip contained no *_test.tif"}
    gt_path = gt_candidates[0]

    with rasterio.open(gt_path) as src:
        gt_arr = src.read(1).astype("float32")
        gt_crs = src.crs
        gt_tf = src.transform
        gt_nodata = src.nodata

    gt_aligned = np.full_like(depths, np.nan, dtype="float32")
    reproject(
        source=gt_arr, destination=gt_aligned,
        src_transform=gt_tf, src_crs=gt_crs,
        dst_transform=meta["transform"], dst_crs=meta["crs"],
        src_nodata=gt_nodata, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

    scores = _depth_errors(depths, gt_aligned)
    print(f"[predict_demo_aoi] ground truth scored over {abs(SCORED_RANGE[1]):g}-{abs(SCORED_RANGE[0]):g} m: "
          + ", ".join(f"{k}={v:.2f}" for k, v in scores.items() if k.endswith("_m") and k != "scored_range_m")
          + f" (n={scores['overlapping_valid_pixels']})")
    if "by_depth_band" in scores:
        print("[predict_demo_aoi] errors by ground-truth depth (positive = predictions too shallow):")
        print(format_depth_band_table(scores["by_depth_band"]))
    return {"ground_truth_file": gt_path, **scores}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=None, help="Override inference.HF_DEFAULT_REPO_ID")
    parser.add_argument("--filename", default=None, help="Override inference.HF_DEFAULT_FILENAME")
    parser.add_argument("--encoder-name", default=None,
                        help="Encoder of the checkpoint (default inference.DEFAULT_ENCODER, resnet50); "
                             "see the checkpoint table in the README")
    parser.add_argument("--aoi", nargs=4, type=float, default=DEFAULT_AOI, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--time-range", nargs=2, default=list(DEFAULT_TIME_RANGE), metavar=("START", "END"))
    parser.add_argument("--max-cloud-coverage", type=float, default=DEFAULT_MAX_CLOUD_COVERAGE)
    args = parser.parse_args()

    outdir = os.environ.get("OUTDIR", "/tmp/dl_bathy_smoke")
    logdir = os.environ.get("LOGDIR", str(REPO_ROOT / "smoke-tests" / "logs"))
    script_name = os.environ.get("SCRIPT_NAME", "predict_demo_aoi")
    log_path = os.path.join(logdir, f"{script_name}.json")

    log: dict = {
        "test": script_name,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "aoi": list(args.aoi),
        "time_range": list(args.time_range),
        "max_cloud_coverage": args.max_cloud_coverage,
    }
    t0 = time.time()

    try:
        import inference

        load_kwargs = {}
        if args.repo_id:
            load_kwargs["repo_id"] = args.repo_id
        if args.filename:
            load_kwargs["filename"] = args.filename
        if args.encoder_name:
            load_kwargs["encoder_name"] = args.encoder_name
        model = inference.load_model(**load_kwargs) if load_kwargs else None

        depths, meta = inference.predict_from_geoai_datacubes(
            aoi=args.aoi,
            time_range=tuple(args.time_range),
            model=model,
            max_cloud_coverage=args.max_cloud_coverage,
            save_folder=os.path.join(outdir, "s2_fetched"),
        )

        log.update({
            "status": "passed",
            "scene_id": meta.get("scene_id"),
            "depth_shape": list(depths.shape),
            "depth_dtype": str(depths.dtype),
            "depth_min_m": float(np.nanmin(depths)),
            "depth_max_m": float(np.nanmax(depths)),
            "depth_mean_m": float(np.nanmean(depths)),
        })

        gt_score = _score_against_ground_truth(depths, meta, outdir)
        if gt_score is not None:
            log["ground_truth_comparison"] = gt_score

    except Exception as exc:  # noqa: BLE001 -- smoke test: capture everything, log it, exit non-zero
        log.update({
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })

    log["duration_s"] = round(time.time() - t0, 1)

    os.makedirs(logdir, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n[predict_demo_aoi] status={log['status']}  duration={log['duration_s']}s")
    print(f"[predict_demo_aoi] log written to {log_path}")
    if log["status"] != "passed":
        print(log.get("traceback", log.get("error", "")), file=sys.stderr)

    return 0 if log["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
