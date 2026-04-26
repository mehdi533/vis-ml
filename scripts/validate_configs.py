#!/usr/bin/env python3
"""Validate YAML configs for syntax and inheritance resolution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.utils import load_yaml


def _iter_yaml_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.yaml") if path.is_file())


def _validate_yaml_parse(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_keys(path: Path, payload: dict[str, Any], keys: list[str], errors: list[str]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        errors.append(f"{path}: missing required keys: {', '.join(missing)}")


def _validate_model_training_config(path: Path, payload: Any, errors: list[str]) -> None:
    if not isinstance(payload, dict):
        return
    norm_path = str(path).replace("\\", "/")
    if "/configs/model/base/" in norm_path:
        return
    if "sweep" not in payload:
        return

    _require_keys(path, payload, ["output_dir", "model", "data", "split", "training", "sweep"], errors)

    data = payload.get("data")
    if isinstance(data, dict):
        _require_keys(path, data, ["csv_path", "target_cols"], errors)

    sweep = payload.get("sweep")
    if isinstance(sweep, dict):
        for key in ["models", "losses", "scalers", "seeds"]:
            if key not in sweep:
                errors.append(f"{path}: sweep missing '{key}'")
                continue
            value = sweep.get(key)
            if not isinstance(value, list) or not value:
                errors.append(f"{path}: sweep.{key} must be a non-empty list")


def validate_configs(root: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    files = _iter_yaml_files(root)

    for path in files:
        try:
            _validate_yaml_parse(path)
        except Exception as exc:
            errors.append(f"{path}: YAML parse failed: {exc}")
            continue

        try:
            resolved = load_yaml(path)
        except Exception as exc:
            errors.append(f"{path}: inheritance resolution failed: {exc}")
            continue

        if "configs/model/" in str(path).replace('\\\\', '/'):
            _validate_model_training_config(path, resolved, errors)

    return len(files), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate repository config YAML files.")
    parser.add_argument("--root", default="configs", help="Config root directory.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Config root not found or not a directory: {root}")

    total, errors = validate_configs(root)
    if errors:
        print(f"Validated {total} YAML files: FAILED ({len(errors)} issue(s))")
        for msg in errors:
            print(f"- {msg}")
        return 1

    print(f"Validated {total} YAML files: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
