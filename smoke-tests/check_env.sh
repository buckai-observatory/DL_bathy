#!/bin/bash
#SBATCH --job-name=dl-bathy-check-env
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=smoke-tests/logs/%x.%j.out
#SBATCH --error=smoke-tests/logs/%x.%j.err
# ---------------------------------------------------------------------------
# Verify the currently-active Python env satisfies requirements.txt (the
# full pipeline). Also useful before opening an issue: "which deps am I
# actually missing?"
#
# Usage:
#   bash smoke-tests/check_env.sh                # fast import check
#   sbatch smoke-tests/check_env.sh               # run as a SLURM job (rare)
#
# Exit codes:
#   0  every package importable
#   1  at least one missing package
# ---------------------------------------------------------------------------
set -uo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Map requirements.txt names to their importable Python module names where
# they differ. Single source of truth so we don't drift from requirements.txt.
PKG_TO_IMPORT="
numpy:numpy
pandas:pandas
matplotlib:matplotlib
scipy:scipy
scikit-image:skimage
rasterio:rasterio
gdal:osgeo.gdal
geoai-datacubes:geoai_datacubes
torch:torch
torchvision:torchvision
segmentation-models-pytorch:segmentation_models_pytorch
pyTMD:pyTMD
huggingface_hub:huggingface_hub
"

echo "[check_env] import check against requirements.txt (fast, no network)"
echo
python - <<PY
import importlib, sys
mapping = """${PKG_TO_IMPORT}"""
missing = []
for line in mapping.strip().splitlines():
    pkg, mod = line.split(":", 1)
    try:
        importlib.import_module(mod)
        print(f"  OK    {pkg:28s}  ({mod})")
    except Exception as e:
        missing.append((pkg, mod, str(e).splitlines()[0]))
        print(f"  MISS  {pkg:28s}  ({mod})  -- {str(e).splitlines()[0][:60]}")

print()
print(f"missing: {len(missing)}")
if missing:
    print()
    print("Install the missing ones with:")
    print("  pip install " + " ".join(p for p, _, _ in missing))
    print("Or just run the tier you actually need:")
    print("  pip install -r requirements.txt            # full pipeline")
    print("  pip install -r requirements-inference.txt  # inference.py only")
    print("  pip install -r requirements-test.txt       # losses.py tests only")
sys.exit(0 if not missing else 1)
PY
