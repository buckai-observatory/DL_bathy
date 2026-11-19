"""requirements-test.txt installs what the test suite needs (#53)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _requirements(name):
    lines = (ROOT / name).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def test_requirements_test_installs_the_package_with_the_ci_extras():
    assert _requirements("requirements-test.txt") == ["-e .[train,dev]"]
    ci = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    assert 'pip install -e ".[train,dev]"' in ci


def test_train_and_dev_extras_exist():
    pyproject = (ROOT / "pyproject.toml").read_text()
    extras = re.search(r"^\[project\.optional-dependencies\]\n(.*?)^\[", pyproject, re.M | re.S).group(1)
    names = set(re.findall(r"^(\w+)\s*=", extras, re.M))
    assert {"train", "dev"} <= names
