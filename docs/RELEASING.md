# Releasing DL_bathy

This is the checklist for cutting a new tagged release. The release
chain mints a fresh [Zenodo](https://zenodo.org/) DOI (which the
GitHub–Zenodo integration handles automatically once configured) and,
in future, a fresh checkpoint on the [Hugging Face
Hub](https://huggingface.co/) alongside it.

## Pre-flight

Before running through the release steps, confirm the following are up
to date on `main`:

- `paper.md` and `paper.bib` compile via the `Draft PDF` GitHub Actions
  workflow. If either was edited since the last release, trigger the
  workflow manually and skim the generated PDF for layout bugs.
- `CITATION.cff` — `date-released` matches the release date; version
  number matches the tag you're about to cut; `doi` still points at the
  current Zenodo concept DOI (or the specific version DOI, depending on
  which convention the project has settled on — the concept DOI resolves
  to *"latest published version"*, so it's usually the friendlier default
  in a citation).
- `CHANGELOG.md` — everything currently under **Unreleased** is
  accurate. Promote it to a new dated section named after the tag being
  cut (e.g. `## v0.2.0 — 2026-11-15`).
- `MODEL_CARD.md` — the training-config and evaluation sections match
  the checkpoint that will accompany this release on the Hugging Face
  Hub. Bump the `Version history` section with a bullet for the new
  release.
- Tests pass locally (`pytest tests/ -x -q`).
- No un-merged PRs that were supposed to be part of this release.

## Cutting the release

1. **Create the tag on `main`** using semver:

   ```bash
   git checkout main && git pull
   git tag -a v0.2.0 -m "DL_bathy v0.2.0"
   git push origin v0.2.0
   ```

2. **Publish a GitHub Release** from the tag (via the web UI or `gh`):

   ```bash
   gh release create v0.2.0 --generate-notes --title "DL_bathy v0.2.0"
   ```

   Edit the auto-generated notes so the top-level summary matches the
   corresponding `CHANGELOG.md` section — GitHub's auto-notes bullet
   every merged PR, which is usually more granular than a user-facing
   release note wants.

3. **Zenodo mints a new version DOI automatically** on GitHub-Release
   publication, provided the GitHub–Zenodo integration is enabled for
   this repository. Verify at <https://zenodo.org/account/settings/github/>
   — the toggle for `buckai-observatory/DL_bathy` should be **On**. If
   it isn't, enable it once, then re-create the GitHub release (Zenodo
   only picks up releases created after the toggle flipped).

4. **Upload the corresponding trained checkpoint to the Hugging Face
   Hub** under the repository referenced in
   [`MODEL_CARD.md`](../MODEL_CARD.md) (default:
   `buckai-observatory/dl-bathy-weights`). Tag the HF revision with the
   same version string (`v0.2.0`) so
   `inference.load_model(revision="v0.2.0")` reproducibly pins to the
   checkpoint that ships with a specific software release.

5. **Update the `Version history` section of `MODEL_CARD.md`** with the
   HF revision + a short summary of what changed since the previous
   release (encoder / training-data / metric deltas).

## Post-release

- Verify the new Zenodo DOI resolves and its record shows the correct
  version + files.
- Run the demo notebook (`notebooks/dl_bathy_demo.ipynb`) end-to-end
  against the new HF checkpoint to make sure the download resolves and
  the predictions look sane.
- Add an **Unreleased** section back at the top of `CHANGELOG.md` for
  ongoing work between now and the next tag.

## Semantic versioning conventions in this project

- **MAJOR** (`v1.0.0` → `v2.0.0`): backwards-incompatible API changes.
  For DL_bathy, this includes changes to the on-disk patch band layout
  (bands 1–12 = S2 reflectance, band 13 = LiDAR depth, band 14 =
  validity mask), changes to the pretrained-weights loading contract
  (`inference.load_model` signature), or changes to the CLI-level
  parameters expected in the `USER CONFIGURATION` blocks at the top of
  `step1_splitting.py` and `step2_train_test.py`.
- **MINOR** (`v0.1.0` → `v0.2.0`): new features, new encoder options,
  new loss functions, new evaluation metrics — anything that adds
  capability without breaking existing callers.
- **PATCH** (`v0.1.0` → `v0.1.1`): bug fixes, docstring / documentation
  improvements, dependency-version bumps that don't affect the API.
