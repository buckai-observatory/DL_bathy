# DL_bathy smoke tests

`tests/` (pytest, CI-wired) covers `losses.py` only, deliberately avoiding
GDAL / network / model-weight dependencies so it runs fast on every push.
That leaves the rest of the pipeline — fetching real Sentinel-2 imagery,
loading real weights, predicting, scoring against ground truth — without
automated coverage. `smoke-tests/` fills that gap with network-dependent,
end-to-end scripts you run manually (or via SLURM), not in CI.

Pattern borrowed from the sibling
[`geoai-datacubes`](https://github.com/buckai-observatory/geoai-datacubes)
project's `smoke-tests/` directory.

## Scripts

| Script | What it checks |
|---|---|
| `check_env.sh` | Every package in `requirements.txt` imports cleanly; reports exactly what's missing. |
| `predict_demo_aoi.sh` | The full inference-only path end-to-end: fetch a real Sentinel-2 scene over the public GBR demo AOI via `geoai-datacubes`, download real weights from the Hugging Face Hub, predict depths, and (if `Ground_Truth.zip` is present at the repo root) score against the public ground truth. The scripted equivalent of running `notebooks/dl_bathy_demo.ipynb` end-to-end without Jupyter. |

Every script is both a valid bash script and a valid SLURM job script (the
`#SBATCH` header lines are comments to bash, directives to `sbatch`):

```bash
bash   smoke-tests/check_env.sh
bash   smoke-tests/predict_demo_aoi.sh

# or on a SLURM cluster:
sbatch smoke-tests/check_env.sh
sbatch smoke-tests/predict_demo_aoi.sh
```

`predict_demo_aoi.sh` passes extra arguments through to
`_run_predict.py`, e.g. to point at a different checkpoint:

```bash
bash smoke-tests/predict_demo_aoi.sh --repo-id org/other-weights --filename best_model.pth
```

## Outputs

- Large files (fetched scenes) land in `$OUTDIR` (default
  `/tmp/dl_bathy_smoke`) — never in git.
- A small JSON summary per script lands in `smoke-tests/logs/*.json` and
  **is** committed, so the run history (did this still work on this date,
  against this weights revision?) is part of the repo.

## When to run these

- Before cutting a release (see `docs/RELEASING.md`'s post-release step).
- After updating the Hugging Face Hub checkpoint, to confirm
  `inference.load_model()` still resolves and predictions look sane.
- If a JOSS reviewer (or anyone else) asks "does this actually work end
  to end" — this is the fastest way to show them, without walking them
  through Jupyter.
