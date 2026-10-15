# Shared boilerplate for every smoke-test script in this folder.
#
# Sourced (not executed) by each *.sh script. Sets up OUTDIR / LOGDIR /
# PYTHONPATH and prints a banner.
#
# A note on SLURM-vs-bash: every smoke-test script here is a valid bash
# script AND a valid SLURM job script -- the `#SBATCH` lines at the top
# are comments to bash but directives to sbatch. So:
#
#   bash   smoke-tests/predict_demo_aoi.sh        # local
#   sbatch smoke-tests/predict_demo_aoi.sh        # cluster
#
# both work. The SLURM headers default to modest resource asks; override
# with `sbatch --mem=8G --time=00:30:00 ...` if you need more.

# Bail on any error, undefined variable, or pipe failure.
set -euo pipefail

# Resolve the repo root regardless of where the script is invoked from
# (relative path from bash, $SLURM_SUBMIT_DIR from sbatch).
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Big outputs (fetched scenes, prediction GeoTIFFs) go to OUTDIR (default
# outside the repo so they never land in git by accident). Logs (small
# JSON summaries) go inside the repo under smoke-tests/logs/ so the run
# history can be committed.
export OUTDIR="${OUTDIR:-/tmp/dl_bathy_smoke}"
export LOGDIR="$REPO_ROOT/smoke-tests/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

# Make `import inference` / `import losses` work without requiring an
# editable install first.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Per-script id (filename without extension). Used for the log JSON path
# and the banner below.
SCRIPT_NAME="$(basename "${BASH_SOURCE[1]:-$0}" .sh)"
export SCRIPT_NAME

echo "=========================================================================="
echo "smoke-test: $SCRIPT_NAME"
echo "  REPO_ROOT  : $REPO_ROOT"
echo "  OUTDIR     : $OUTDIR"
echo "  LOGDIR     : $LOGDIR"
echo "  PYTHON     : $(command -v python || echo MISSING)"
echo "  SLURM_JOB  : ${SLURM_JOB_ID:-(not running under sbatch)}"
echo "=========================================================================="
