# Contributors

<!-- DRAFT for issue #15: every role below must be confirmed by the
     person it describes before this is merged. Delete this comment once
     all authors have signed off. -->

Roles in the *software* (design, implementation, testing, supervision,
data, review). Author order follows `paper.md` and `CITATION.cff`.

## Principal investigator

- **Moortgat, Joachim** ([@jmoortgat](https://github.com/jmoortgat)) —
  Professor, School of Earth Sciences, The Ohio State University; founding
  Director, BuckAI Observatory. Project supervision; Sentinel-2 retrieval
  through `geoai-datacubes` (#2–#4); the inference-only entry point,
  Hugging Face weight hosting, Colab demo and model card (#5); changelog
  and release process (#6); paper text (#27); documentation website (#28);
  code review and issue reports.

## Code contributions

- **Hsu, Hsiao-Jou (Amy)** ([@Amy-Hsu](https://github.com/Amy-Hsu)) —
  School of Earth Sciences, The Ohio State University. Lead developer
  (March 2026 – present): pipeline design; patch generation and SCL-based
  cloud / non-water masking (`sdb split`); model architectures, losses and
  training / evaluation loop (`sdb train`); tide correction with
  pyTMD/EOT20 (#7, #17); packaging as `dl-bathy` with YAML configuration,
  the `sdb` CLI and the synthetic test suite (#8, #13, #26); bug fixes
  (#19–#22, #39–#46); reference-depth data preparation and training of the
  released checkpoints.

## How to be listed

Anyone whose pull request is merged is added here with a one-line
description of their contribution. See [CONTRIBUTING.md](CONTRIBUTING.md).
