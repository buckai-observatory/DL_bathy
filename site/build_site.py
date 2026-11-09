"""Build the Quarto sources of the documentation website from the repository.

The Markdown files, notebooks and docstrings on GitHub are the only sources.
This script copies them into site/_build/, with
  * README.md as the home page, every docs/*.md as one page, and the
    project files (MODEL_CARD.md, CHANGELOG.md, CONTRIBUTING.md,
    CODE_OF_CONDUCT.md) as pages under "Project";
  * every notebooks/*.ipynb as one page, shown with its saved outputs (not
    executed), with its code cells folded;
  * links rewritten: links between site pages point to the site, and links to
    any other file of the repository point to that file on GitHub;
  * at the top of each page, a link back to the same file on GitHub (and, for
    a notebook, a link to open it in Colab);
  * a generated _quarto.yml, including the quartodoc configuration of the API
    reference, which `quartodoc build` writes from the docstrings into api/.
New files in docs/ or notebooks/ appear on the site and in its navbar without
further changes.

Usage (from the repository root, with the package and the `docs` extra
installed):
    python3 site/build_site.py
    cd site/_build && PYTHONPATH=../.. python -m quartodoc build && cd ../..
    quarto render site/_build          # output in site/_build/_site/
"""
import json
import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "_build"
GITHUB = "https://github.com/buckai-observatory/DL_bathy"
COLAB = "https://colab.research.google.com/github/buckai-observatory/DL_bathy/blob"
SITE_URL = "https://buckai-observatory.org/DL_bathy/"
BRANCH = "main"

DOC_ORDER = []                                         # navbar order of docs/*.md; others follow by name
PROJECT = {                                            # project files -> short navbar names
    "MODEL_CARD.md": "Model card",
    "CHANGELOG.md": "Changelog",
    "CONTRIBUTING.md": "Contributing",
    "CODE_OF_CONDUCT.md": "Code of conduct",
}
NB_TITLES = {                                          # navbar entries; other notebooks follow
    "dl_bathy_demo.ipynb": "Inference demo",
}
# source file (relative to the repository root) -> page file in the site
PAGES = {"README.md": "index.md"}
for f in sorted((REPO / "docs").glob("*.md"),
                key=lambda f: (DOC_ORDER.index(f.name) if f.name in DOC_ORDER else len(DOC_ORDER), f.name)):
    PAGES[f"docs/{f.name}"] = f.name
for name in PROJECT:
    if (REPO / name).exists():
        PAGES[name] = name
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"),
                   key=lambda f: (list(NB_TITLES).index(f.name) if f.name in NB_TITLES else len(NB_TITLES), f.name))
for f in NOTEBOOKS:
    PAGES[f"notebooks/{f.name}"] = f"notebooks/{f.name}"
ASSETS = ["notebooks/images"]                          # copied into the site as they are

# API reference: one section per module, listing its public functions and
# classes. quartodoc renders them from the docstrings.
API_SECTIONS = [
    ("Configuration", "Settings for `sdb split` and `sdb train`.", "dl_bathy.config",
     ["load_config", "example_config"]),
    ("Patch preparation (`sdb split`)", "Scenes, masks, tide correction, patches and splits.", "dl_bathy.split",
     ["run", "find_sentinel_files", "extract_acquisition_time", "get_bbox_from_transform",
      "sample_tide_pytmd", "generate_tide_grid", "get_tide_grid_for_scene",
      "extract_patches_no_nans_gen", "assign_split", "balance_metadata_by_depth",
      "augment_folder_tifs"]),
    ("Patch files", "Reading and writing 14-band patch GeoTIFFs.", "dl_bathy.patch_io",
     ["save_patches_from_metadata", "read_patch_tif"]),
    ("Training and evaluation (`sdb train`)", "Model, dataset, training loop and metrics.", "dl_bathy.train",
     ["run", "BathyDataset", "load_model", "build_deeplab_head", "init_classifier_head",
      "replace_bn_with_gn", "regularization", "count_parameters", "filter_valid_patches",
      "compute_global_stats_from_files", "calculate_binary_metrics", "save_prediction_geotiff",
      "set_reproducible", "log_params"]),
    ("Loss functions", "Masked, depth-aware losses.", "losses",
     ["masked_sum_sse_and_count", "masked_sum_abs_and_count", "masked_sum_rpe_and_count",
      "masked_sum_swf_and_count", "masked_sum_huber_and_count"]),
    ("Inference", "Pretrained weights from the Hugging Face Hub.", "inference",
     ["predict_from_geoai_datacubes", "load_model", "predict_depth", "build_model"]),
    ("Band normalisation", "Shared by `sdb split` and inference.", "dl_bathy.bands",
     ["normalize_reflectance"]),
]

LINK = re.compile(r"(\]\()([^)\s]+)(\))")
H1 = re.compile(r"^# (.+)$", re.M)


def site_name(target: str) -> str:
    return re.sub(r"\.(md|ipynb)$", ".html", target)


def resolve(target: str, source: str) -> str:
    """Rewrite one link target of the file `source` (both relative to the repository)."""
    if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
        return target
    path, _, anchor = target.partition("#")
    repo_path = (REPO / Path(source).parent / path).resolve().relative_to(REPO).as_posix()
    here = Path(PAGES[source]).parent
    if repo_path in PAGES:
        url = os.path.relpath(site_name(PAGES[repo_path]), here)
    elif any(repo_path == a or repo_path.startswith(a + "/") for a in ASSETS):
        url = os.path.relpath(repo_path, here)
    elif (REPO / repo_path).is_dir():
        url = f"{GITHUB}/tree/{BRANCH}/{repo_path}".rstrip("/.")
    else:
        url = f"{GITHUB}/blob/{BRANCH}/{repo_path}"
    return url + (f"#{anchor}" if anchor else "")


def is_notebook(url: str) -> bool:
    return (url.split("#")[0].endswith(".ipynb") or "colab.research.google.com/github" in url
            or re.search(r"(^|/)notebooks/[^/#]+\.html", url) is not None)


NEW_TAB = '{target="_blank"}'


def relink(text: str, source: str) -> str:
    """Rewrite the links of a page; links to notebooks (site page, GitHub, Colab) open a new tab."""
    def one(m):
        url = resolve(m.group(2), source)
        tail = text[m.end():m.end() + 1]
        attr = NEW_TAB if is_notebook(url) and tail != "{" else ""
        return m.group(1) + url + m.group(3) + attr
    return LINK.sub(one, text)


def md_page(source: str) -> tuple[str, str]:
    text = (REPO / source).read_text()
    # the home page does not link to itself
    text = "\n".join(l for l in text.splitlines() if SITE_URL not in l) + "\n"
    m = H1.search(text)
    title = m.group(1).strip() if m else Path(source).stem
    if m:
        text = text[:m.start()] + text[m.end():].lstrip("\n")
    back = (f"[View this page on GitHub]({GITHUB}/blob/{BRANCH}/{source})"
            "{.github-link}\n\n")
    return title, f'---\ntitle: "{title}"\n---\n\n' + back + relink(text, source)


def notebook_page(source: str) -> dict:
    nb = json.loads((REPO / source).read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = relink("".join(cell["source"]), source)
    # the first heading becomes the page title
    first = "".join(nb["cells"][0]["source"])
    m = H1.search(first) if nb["cells"][0]["cell_type"] == "markdown" else None
    title = m.group(1).strip() if m else Path(source).stem
    if m:
        nb["cells"][0]["source"] = first[:m.start()] + first[m.end():].lstrip("\n")
    has_outputs = any(c.get("outputs") for c in nb["cells"])
    shown = ("This page shows the notebook with the outputs of its last run; open it in "
             "Colab or Jupyter to run the code." if has_outputs else
             "This page shows the notebook's code and text; open it in Colab or Jupyter "
             "to run it.")
    links = (f"[View this notebook on GitHub]({GITHUB}/blob/{BRANCH}/{source}){{.github-link target=\"_blank\"}} "
             f"[Open in Colab]({COLAB}/{BRANCH}/{source}){{.github-link target=\"_blank\"}}\n\n" + shown)
    # browser-tab title without LaTeX
    plain = re.sub(r"\$([^$]*)\$", lambda m: re.sub(r"\\[A-Za-z]+|[_^{}]", "", m.group(1)), title)
    front = {"cell_type": "raw", "metadata": {},
             "source": "---\ntitle: " + json.dumps(title) + "\npagetitle: " + json.dumps(plain) + "\n---"}
    intro = {"cell_type": "markdown", "metadata": {}, "source": links}
    nb["cells"] = [front, intro] + nb["cells"]
    for cell in nb["cells"]:
        cell.pop("id", None)
    nb["nbformat_minor"] = max(nb.get("nbformat_minor", 4), 4)
    return nb


def quartodoc_yaml() -> str:
    lines = ["quartodoc:",
             "  package: null                  # objects below are given by their full paths",
             "  dir: api",
             "  title: \"API reference\"",
             "  sidebar: api/_sidebar.yml",
             "  sections:"]
    for title, desc, module, names in API_SECTIONS:
        lines += [f"    - title: \"{title}\"", f"      desc: \"{desc}\"", "      contents:"]
        lines += [f"        - {module}.{n}" for n in names]
    return "\n".join(lines) + "\n"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    docs, project, nbs = [], [], []
    for source, target in PAGES.items():
        dst = OUT / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        if source.endswith(".ipynb"):
            dst.write_text(json.dumps(notebook_page(source), indent=1))
            nbs.append((target, NB_TITLES.get(Path(source).name, Path(source).stem)))
        else:
            title, body = md_page(source)
            dst.write_text(body)
            if source in PROJECT:
                project.append((target, PROJECT[source]))
            else:
                docs.append((target, "Home" if target == "index.md" else title))
    for a in ASSETS:
        if (REPO / a).exists():
            shutil.copytree(REPO / a, OUT / a)
    shutil.copy(REPO / "site" / "styles.css", OUT / "styles.css")
    # quartodoc writes the sidebar on `quartodoc build`; an empty one lets
    # `quarto render` run on its own as well
    (OUT / "api").mkdir()
    (OUT / "api" / "_sidebar.yml").write_text("website:\n  sidebar: []\n")
    doc_items = "\n".join(f"      - href: {t}\n        text: \"{n}\"" for t, n in docs)
    nb_items = "\n".join(f"          - href: {t}\n            text: \"{n}\"\n            target: _blank" for t, n in nbs)
    project_items = "\n".join(f"          - href: {t}\n            text: \"{n}\"" for t, n in project)
    (OUT / "_quarto.yml").write_text(f"""project:
  type: website
  output-dir: _site

execute:
  enabled: false                 # notebooks are shown with their saved outputs

metadata-files:
  - api/_sidebar.yml

website:
  title: "DL_bathy"
  site-url: {SITE_URL}
  repo-url: {GITHUB}
  repo-actions: [issue]
  navbar:
    left:
{doc_items}
      - text: "Notebooks"
        menu:
{nb_items}
      - href: api/index.qmd
        text: "API reference"
      - text: "Project"
        menu:
{project_items}
    right:
      - icon: github
        href: {GITHUB}
        aria-label: GitHub
  page-footer:
    center: "Generated from the Markdown files, notebooks and docstrings of the [repository]({GITHUB}); edit those, not this site."

format:
  html:
    theme:
      light: cosmo
      dark: darkly
    css: styles.css
    toc: true
    code-fold: true
    code-summary: "Code"

""" + quartodoc_yaml())
    print(f"wrote {len(docs) + len(project)} pages and {len(nbs)} notebooks to {OUT}")


if __name__ == "__main__":
    main()
