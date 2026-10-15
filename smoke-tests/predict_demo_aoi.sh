#!/bin/bash
#SBATCH --job-name=dl-bathy-smoke-predict
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=smoke-tests/logs/%x.%j.out
#SBATCH --error=smoke-tests/logs/%x.%j.err
# ---------------------------------------------------------------------------
# Smoke test: the inference-only path, end-to-end, on the public GBR demo
# AOI -- fetch a real Sentinel-2 scene via geoai-datacubes, download real
# weights from the Hugging Face Hub, predict depths, and (if
# Ground_Truth.zip is present) score against the public ground truth.
#
# This is the scripted equivalent of running
# notebooks/dl_bathy_demo.ipynb end-to-end without opening Jupyter --
# useful for a reviewer's terminal, CI (if you choose to wire it in; not
# wired in by default since it needs network + real weights), or a
# pre-release sanity check (see docs/RELEASING.md).
#
# Runs either as a regular bash script (local) or as a SLURM job:
#   bash   smoke-tests/predict_demo_aoi.sh
#   sbatch smoke-tests/predict_demo_aoi.sh
#
# Fetched scenes land in $OUTDIR (default /tmp/dl_bathy_smoke, so they
# never accidentally land in git). A small JSON summary is written to
# smoke-tests/logs/predict_demo_aoi.json so the run history can be
# committed.
#
# Pass extra args straight through to _run_predict.py, e.g. to point at
# a different HF repo/checkpoint:
#   bash smoke-tests/predict_demo_aoi.sh --repo-id org/other-weights --filename best_model.pth
# ---------------------------------------------------------------------------

# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python smoke-tests/_run_predict.py "$@"
