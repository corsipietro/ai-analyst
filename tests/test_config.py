"""Tests of the configuration, and a guard against domain words in the generic core."""
from pathlib import Path

from sim_analyst.config import ROOT, deep_merge, load_domain

DOMAIN_WORDS = ["debris", "satellite", "launch", "orbit", "shell", "collision", "toy_shell"]


def test_deep_merge_overrides_only_given_keys():
    """A domain can change one role's model without repeating the rest of the settings."""
    base = {"roles": {"explorer": {"model": "a", "temperature": 0}}, "prompt": "v1"}
    merged = deep_merge(base, {"roles": {"explorer": {"model": "b"}}})
    assert merged == {"roles": {"explorer": {"model": "b", "temperature": 0}}, "prompt": "v1"}
    assert base["roles"]["explorer"]["model"] == "a"


def test_load_domain_resolves_paths(tmp_path):
    """Paths in config.yaml are relative to the domain folder; settings overrides are merged."""
    folder = tmp_path / "demo"
    folder.mkdir()
    (folder / "config.yaml").write_text(
        "name: demo\ndata:\n  database: ../data/results.duckdb\ncontext: context.md\n"
        "settings:\n  limits: {max_rows: 50}\n", encoding="utf-8")
    (folder / "context.md").write_text("A demo system.", encoding="utf-8")
    settings = tmp_path / "settings.yaml"
    settings.write_text("prompt: v1\nlimits: {max_rows: 200, max_points: 40}\n", encoding="utf-8")
    domain = load_domain("demo", domains_dir=tmp_path, settings_file=settings)
    assert domain.data.database == (tmp_path / "data" / "results.duckdb").resolve()
    assert domain.context == "A demo system."
    assert domain.settings["limits"] == {"max_rows": 50, "max_points": 40}


def test_core_has_no_domain_words():
    """The package and the generic prompts must not mention any simulated system."""
    offenders = []
    for path in sorted((ROOT / "sim_analyst").rglob("*")):
        if path.suffix in {".py", ".md", ".yaml"}:
            text = path.read_text(encoding="utf-8").lower()
            offenders += [f"{path.name}: {w}" for w in DOMAIN_WORDS if w in text]
    assert offenders == []


def test_run_settings_use_method_v3_and_an_answer_language():
    """settings.yaml selects the v3 method and holds the answer language and the bootstrap threshold."""
    from sim_analyst.config import SETTINGS_FILE, load_yaml
    settings = load_yaml(SETTINGS_FILE)
    assert settings["prompt"] == "v3"
    assert settings["answer_language"] == "English"
    assert settings["bootstrap"]["min_n"] == 30
