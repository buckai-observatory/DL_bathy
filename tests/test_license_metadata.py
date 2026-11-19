"""pyproject.toml declares the license as an SPDX expression (#55)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text()


def test_license_is_an_spdx_expression_with_the_license_file():
    assert re.search(r'^license = "MIT"$', PYPROJECT, re.M)
    assert re.search(r'^license-files = \["LICENSE"\]$', PYPROJECT, re.M)
    assert (ROOT / "LICENSE").exists()


def test_no_license_classifier_and_setuptools_new_enough():
    # A License classifier is not allowed next to an SPDX expression, and
    # setuptools reads `license = "MIT"` only from version 77 on.
    assert "License ::" not in PYPROJECT
    assert '"setuptools>=77"' in PYPROJECT
