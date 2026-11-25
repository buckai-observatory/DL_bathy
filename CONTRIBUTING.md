# Contributing to `DL_bathy`

Thanks for your interest in `DL_bathy`. This document covers the three
things JOSS asks community-driven projects to make explicit: **how to seek
support**, **how to report problems**, and **how to contribute code or
docs**.

This is a small academic project maintained by graduate researchers.
Expect informal, direct, low-ceremony collaboration; issues and pull
requests are triaged in batches rather than within hours.

---

## 1. Seeking support / asking questions

Open an **Issue** on the repository:

* <https://github.com/buckai-observatory/DL_bathy/issues>

That's the right place for "how do I…" questions, usage questions, and
anything that isn't a clear bug.

For OSU-internal users, the BuckAI Observatory office hours are the
fastest path. Non-OSU users can email
[hsu.771@osu.edu](mailto:hsu.771@osu.edu) for project-level questions;
please prefer opening an Issue for anything that might benefit other users.

---

## 2. Reporting bugs and requesting features

Open a GitHub Issue:

* <https://github.com/buckai-observatory/DL_bathy/issues>

### What to include in a bug report

Please give us enough information to reproduce the problem:

* **Environment** — output of `python -V`, `pip show torch`,
  `pip show segmentation-models-pytorch`, `pip show rasterio`, and the
  operating system.
* **What you ran** — `sdb split` or `sdb train` (or `inference.py`), plus
  the relevant entries from your YAML config file.
* **What happened** — the full traceback, plus, for `sdb train`, the
  relevant lines from `console_output_step2.log` in the output folder.
* **What you expected** — one or two sentences is enough.

### What to include in a feature request

State the workflow you want to support and which step blocks it. "Support
a third backbone for Step 2" is more actionable than "more model options."

---

## 3. Contributing code

Pull requests are welcome, including from first-time contributors.

### Development setup

```bash
git clone https://github.com/buckai-observatory/DL_bathy.git
cd DL_bathy

pip install -e ".[train,dev]"
```

This installs the `sdb` command and everything the test suite needs
(`pip install -r requirements-test.txt` does the same). Use `".[all,dev]"`
to work on geoai-datacubes fetching or pyTMD tides as well. Only touching
`inference.py`? `requirements-inference.txt` is a lighter subset — see the
README's Requirements section.

Sentinel-2 imagery is fetched at run time via
[`geoai-datacubes`](https://github.com/buckai-observatory/geoai-datacubes)
in `DATA_SOURCE='geoai_datacubes'` mode, or read from a folder of
pre-downloaded 13-band L2A GeoTIFFs in `DATA_SOURCE='local_scenes'`
mode. See the README's "Choose a data-acquisition mode" section.
Tide correction defaults to `pyTMD` + EOT20, which needs the EOT20
constituent files on disk — see the README's "Tide model data" section.
Set `TIDE_SOURCE: custom` or `none` in the `split` section of your config
to avoid that dependency entirely.

### Pull request checklist

* Keep settings in the YAML config (`sdb init-config` writes the
  defaults) — don't hardcode paths or parameters in the code.
* If you change patch-band ordering, output filenames, or config defaults,
  update the corresponding section of `README.md` in the same PR.
* Describe what you tested the change against (even a small local run) in
  the PR description — CI (`.github/workflows/tests.yml`) runs the unit
  tests and a small synthetic `sdb split` / `sdb train`, but not the
  network-dependent path (Sentinel-2 fetches, tide model data, the released
  weights), so PR descriptions are the main record of what was verified
  there.

---

## Code of Conduct

Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).
