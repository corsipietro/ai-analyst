"""Read the run settings (settings.yaml) and a domain's configuration (domains/<name>/config.yaml).

The configuration says only WHERE the data are and which text files describe the system.
Model names, sampling and limits come from settings.yaml; a domain may override them.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # the repository root (settings.yaml and domains/ live here)
SETTINGS_FILE = ROOT / "settings.yaml"
DOMAINS_DIR = ROOT / "domains"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class DataSpec:
    """Where the data are: one database file, or one file per table (Parquet or CSV)."""

    database: Path | None = None                 # a DuckDB file
    tables: dict[str, Path] = field(default_factory=dict)   # table name -> Parquet/CSV file
    sha256: str | None = None                    # expected fingerprint of `database`, if given


@dataclass(frozen=True)
class Domain:
    """Everything the analyst needs for one simulated system."""

    name: str
    folder: Path
    data: DataSpec
    context: str                                 # the text of context.md
    bench_file: Path | None
    settings: dict                               # settings.yaml merged with the domain overrides


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of base where the keys of override replace (recursively) those of base."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict:
    """Read a YAML file and return its content as a dict (an empty file gives {})."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(path: Path = SETTINGS_FILE) -> dict:
    """Return the shared run settings."""
    return load_yaml(path)


def parse_data_spec(raw: dict, folder: Path) -> DataSpec:
    """Turn the 'data' section of config.yaml into a DataSpec, with paths relative to the domain folder."""
    if not isinstance(raw, dict) or ("database" in raw) == ("tables" in raw):
        raise ValueError("config.yaml 'data' needs exactly one of 'database' (a DuckDB file) "
                         "or 'tables' (table name -> Parquet/CSV file).")
    if "database" in raw:
        return DataSpec(database=(folder / raw["database"]).resolve(), sha256=raw.get("sha256"))
    tables = {name: (folder / path).resolve() for name, path in raw["tables"].items()}
    return DataSpec(tables=tables)


def parse_override_items(items: list[str]) -> dict:
    """Read 'key=value' texts (from --set) into {'dotted.key': value}; the value is read as YAML (false, 2, 0.5, v4)."""
    flat = {}
    for item in items:
        key, separator, text = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"override {item!r} must look like key=value (e.g. graph=rewoo or roles.planner.reasoning=false)")
        flat[key.strip()] = yaml.safe_load(text)
    return flat


def parse_overrides(items: list[str]) -> dict:
    """The same overrides as a nested dict, ready for deep_merge: 'roles.planner.top_k=10' -> {roles: {planner: {top_k: 10}}}."""
    nested = {}
    for key, value in parse_override_items(items).items():
        parts = key.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def load_domain(name: str, domains_dir: Path = DOMAINS_DIR,
                settings_file: Path = SETTINGS_FILE, overrides: dict | None = None) -> Domain:
    """Read domains/<name>/config.yaml and the files it points to; overrides (nested) go on top of all settings."""
    folder = domains_dir / name
    raw = load_yaml(folder / "config.yaml")
    data = parse_data_spec(raw.get("data"), folder)
    context = (folder / raw.get("context", "context.md")).read_text(encoding="utf-8")
    bench = folder / raw["bench"] if raw.get("bench") else None
    settings = deep_merge(load_settings(settings_file), raw.get("settings", {}))
    settings = deep_merge(settings, overrides or {})
    return Domain(name=raw.get("name", name), folder=folder, data=data,
                  context=context, bench_file=bench, settings=settings)


def sha256_of(path: Path) -> str:
    """Return the SHA-256 fingerprint of a file (as `shasum -a 256` prints it)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_fingerprint(spec: DataSpec) -> None:
    """Raise if the database file does not match the fingerprint written in config.yaml."""
    if spec.database is not None and spec.sha256:
        actual = sha256_of(spec.database)
        if actual != spec.sha256:
            raise ValueError(f"{spec.database.name} has changed: sha256 {actual}, expected {spec.sha256}.")


def load_prompt(version: str) -> tuple[str, dict]:
    """Return the generic method (method_<version>.md) and the role requests (roles_<version>.yaml)."""
    method = (PROMPTS_DIR / f"method_{version}.md").read_text(encoding="utf-8")
    roles = load_yaml(PROMPTS_DIR / f"roles_{version}.yaml")
    return method, roles
