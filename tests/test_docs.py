"""Documentation statements that must keep matching the code (#54)."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_python_badge_matches_requires_python():
    minimum = re.search(r'requires-python\s*=\s*">=([\d.]+)"', (ROOT / "pyproject.toml").read_text()).group(1)
    assert f"[![Python {minimum}+]" in README
    assert f"badge/Python-{minimum}%2B-" in README


def test_readme_links_the_default_weights_repository():
    repo_id = re.search(r'^HF_DEFAULT_REPO_ID\s*=\s*"([^"]+)"', (ROOT / "inference.py").read_text(), re.M).group(1)
    assert f"https://huggingface.co/{repo_id}" in README


def test_docs_do_not_say_tests_cover_only_the_losses():
    smoke = (ROOT / "smoke-tests" / "README.md").read_text(encoding="utf-8")
    for text in (README, smoke):
        assert "covers `losses.py` only" not in text
        assert "`tests/` covers `losses.py` in CI" not in text
