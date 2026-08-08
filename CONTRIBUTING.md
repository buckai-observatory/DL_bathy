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

Open a **GitHub Discussion** or **Issue** on the repository:

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
* **What you ran** — `step1_splitting.py` or `step2_train_test.py`, plus
  the relevant entries from the `USER CONFIGURATION` block you edited.
* **What happened** — the full traceback, plus the relevant lines from
  `console_output_step1.log` / `console_output_step2.log`.
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

pip install numpy pandas matplotlib scipy scikit-image rasterio gdal \
            earthengine-api torch torchvision segmentation-models-pytorch
```

You will also need a Google Earth Engine account (for Step 1 cloud/water
masking) and, optionally, the DTU23 tidal model executable for tide
correction — see the README's Requirements section.

### Pull request checklist

* Keep the `USER CONFIGURATION` block pattern — don't hardcode paths or
  parameters elsewhere in the script.
* If you change patch-band ordering, output filenames, or config defaults,
  update the corresponding section of `README.md` in the same PR.
* Describe what you tested the change against (even a small local run) in
  the PR description — we don't yet have CI running the full pipeline
  end-to-end (large data / GEE auth requirements), so PR descriptions are
  the main record of what was verified.

---

## Code of Conduct

Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).
