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


def _check_env_packages():
    text = (ROOT / "smoke-tests" / "check_env.sh").read_text()
    block = re.search(r'PKG_TO_IMPORT="\n(.*?)\n"', text, re.S).group(1)
    return dict(line.split(":", 1) for line in block.splitlines())


def test_requirements_do_not_ask_for_the_unused_gdal_bindings():
    # #52: an unpinned `gdal` is built against the system libgdal and fails
    # to install; nothing imports osgeo (rasterio ships its own GDAL).
    for name in ("requirements.txt", "requirements-inference.txt"):
        assert "gdal" not in [r.lower() for r in _requirements(name)]
    assert "gdal" not in [p.lower() for p in _check_env_packages()]
    for path in list((ROOT / "dl_bathy").glob("*.py")) + list(ROOT.glob("*.py")):
        assert not re.search(r"^\s*(from|import)\s+osgeo\b", path.read_text(), re.M), path


def test_check_env_checks_exactly_the_packages_in_requirements_txt():
    assert sorted(_check_env_packages()) == sorted(_requirements("requirements.txt"))
