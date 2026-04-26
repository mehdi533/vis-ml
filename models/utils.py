# utils.py
# Shared parsing, config, naming, and export utilities for model workflows.

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# YAML / arg parsing helpers
# -----------------------------

def _resolve_yaml_path(raw_path: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    if relative_to is not None:
        candidate = (relative_to.parent / path).resolve()
        if candidate.exists():
            return candidate

    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    if relative_to is not None:
        return (relative_to.parent / path).resolve()
    return repo_candidate


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_yaml_recursive(path: Path, visiting: set[Path]) -> Any:
    resolved_path = path.resolve()
    if resolved_path in visiting:
        chain = " -> ".join(str(p) for p in list(visiting) + [resolved_path])
        raise ValueError(f"Detected circular YAML inheritance: {chain}")

    visiting.add(resolved_path)
    try:
        with resolved_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            return payload

        inherit_raw = payload.pop("inherits", None)
        if inherit_raw is None:
            return payload

        if isinstance(inherit_raw, (str, Path)):
            inherit_paths = [inherit_raw]
        elif isinstance(inherit_raw, list):
            inherit_paths = inherit_raw
        else:
            raise TypeError(
                f"'inherits' must be a string or list in {resolved_path}, got {type(inherit_raw).__name__}."
            )

        merged: Dict[str, Any] = {}
        for parent_raw in inherit_paths:
            parent_path = _resolve_yaml_path(parent_raw, relative_to=resolved_path)
            parent_cfg = _load_yaml_recursive(parent_path, visiting)
            if not isinstance(parent_cfg, dict):
                raise TypeError(f"Inherited config must be a mapping: {parent_path}")
            merged = _deep_merge_dict(merged, parent_cfg)

        return _deep_merge_dict(merged, payload)
    finally:
        visiting.remove(resolved_path)


def load_yaml(path: str | Path) -> Any:
    resolved_path = _resolve_yaml_path(path)
    return _load_yaml_recursive(resolved_path, set())


def write_yaml(path: str | Path, data: Dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def normalize_arg_list(values, default=None):
    if values is None:
        return default
    items = []
    for value in values:
        if isinstance(value, str):
            items.extend([part.strip() for part in value.split(",") if part.strip()])
        else:
            items.append(value)
    return items


def parse_size_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return [int(v) for v in items] if items else None
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (list, tuple)) for v in value):
            return [[int(x) for x in group] for group in value]
        return [int(v) for v in value]
    return [int(value)]


def parse_group_head_indices(value):
    if value is None:
        return None

    if isinstance(value, str):
        groups = []
        for chunk in value.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            group = [int(part.strip()) for part in chunk.split(",") if part.strip()]
            if group:
                groups.append(group)
        return groups or None

    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (list, tuple)) for v in value):
            return [[int(x) for x in group] for group in value]
        return [[int(v) for v in value]]

    return [[int(value)]]


def resolve_feature_indices(
    feature_cols: Sequence[str],
    *,
    idx_key: str,
    col_key: str,
    cfg: Dict,
):
    idx_values = cfg.get(idx_key)
    col_values = cfg.get(col_key)

    if idx_values is not None and col_values is not None:
        raise ValueError(f"Specify only one of '{idx_key}' or '{col_key}'.")

    if col_values is not None:
        names = [str(v) for v in col_values]
        name_to_idx = {name: i for i, name in enumerate(feature_cols)}
        missing = [name for name in names if name not in name_to_idx]
        if missing:
            raise KeyError(f"Unknown feature names in '{col_key}': {missing}")
        return [name_to_idx[name] for name in names]

    if idx_values is not None:
        idx = [int(v) for v in idx_values]
        bad = [i for i in idx if i < 0 or i >= len(feature_cols)]
        if bad:
            raise IndexError(f"Out-of-range indices in '{idx_key}': {bad}")
        return idx

    return None


# -----------------------------
# Override-grid / config cloning
# -----------------------------

def _grid_values(value):
    if isinstance(value, dict) and "values" in value:
        options = value["values"]
        if isinstance(options, list):
            return options
        return [options]
    if isinstance(value, list):
        return value
    return [value]


def build_override_grid(grid_cfg: Optional[Dict]) -> list[Dict[str, Any]]:
    if not grid_cfg:
        return [{}]
    if "rows" in grid_cfg:
        rows = grid_cfg.get("rows") or [{}]
        if not isinstance(rows, list):
            raise ValueError("'rows' override grid must be a list of dictionaries.")

        base_cfg = {key: value for key, value in grid_cfg.items() if key != "rows"}
        bad_base = [
            key for key, value in base_cfg.items() if isinstance(value, dict) and "values" in value
        ]
        if bad_base:
            raise ValueError(
                "When 'rows' is used in an override grid, sibling keys must be fixed values, "
                f"not value grids. Invalid keys: {bad_base}"
            )

        out: list[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Each item in an override-grid 'rows' list must be a dictionary.")
            merged = dict(base_cfg)
            merged.update(row)
            out.append(merged)
        return out

    keys = list(grid_cfg.keys())
    values = [_grid_values(grid_cfg[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def clone_cfg_with_overrides(
    cfg: Dict,
    *,
    model_overrides: Optional[Dict] = None,
    training_overrides: Optional[Dict] = None,
    data_overrides: Optional[Dict] = None,
) -> Dict:
    run_cfg = deepcopy(cfg)
    run_cfg.setdefault("model", {})
    run_cfg.setdefault("training", {})
    run_cfg.setdefault("data", {})

    if model_overrides:
        run_cfg["model"].update(model_overrides)
    if training_overrides:
        run_cfg["training"].update(training_overrides)
    if data_overrides:
        run_cfg["data"].update(data_overrides)

    return run_cfg


# -----------------------------
# Data-config resolution
# -----------------------------

def load_feature_name_registry(path: str | Path | None) -> Optional[Dict]:
    if not path:
        return None
    return load_yaml(path)


def _append_unique(items: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in items:
            items.append(value)


def _target_uses_prefix(targets: Sequence[str], prefix: str) -> bool:
    return any(str(target).startswith(prefix) for target in targets)


def _collect_registry_input_allowlist(
    registry: Dict,
    include_groups: Optional[Sequence[str]] = None,
    group_overrides: Optional[Dict[str, Dict[str, Sequence[str]]]] = None,
) -> tuple[list[str], list[str]]:
    group_names = list(include_groups or ["x_op", "x_cont", "x_sched"])
    allowed_cols: list[str] = []
    allowed_prefixes: list[str] = []
    group_overrides = group_overrides or {}

    for group_name in group_names:
        group_cfg = registry.get(group_name, {})
        if not isinstance(group_cfg, dict):
            continue
        override_cfg = group_overrides.get(group_name, {}) or {}
        keep_fields = {str(value) for value in override_cfg.get("keep_fields", []) or []}
        drop_fields = {str(value) for value in override_cfg.get("drop_fields", []) or []}
        keep_prefix_keys = {str(value) for value in override_cfg.get("keep_prefix_keys", []) or []}
        drop_prefix_keys = {str(value) for value in override_cfg.get("drop_prefix_keys", []) or []}

        for key, value in group_cfg.items():
            if key == "prefixes" and isinstance(value, dict):
                selected_prefixes = []
                for prefix_key, prefix_value in value.items():
                    if keep_prefix_keys and str(prefix_key) not in keep_prefix_keys:
                        continue
                    if str(prefix_key) in drop_prefix_keys:
                        continue
                    selected_prefixes.append(str(prefix_value))
                _append_unique(allowed_prefixes, selected_prefixes)
                continue

            if key.endswith("_fields") and isinstance(value, list):
                selected_fields = []
                for field in value:
                    field_name = str(field)
                    if keep_fields and field_name not in keep_fields:
                        continue
                    if field_name in drop_fields:
                        continue
                    selected_fields.append(field_name)
                _append_unique(allowed_cols, selected_fields)

    return allowed_cols, allowed_prefixes


def resolve_data_config(data_cfg: Dict) -> Dict:
    resolved = deepcopy(data_cfg)
    registry = load_feature_name_registry(resolved.get("feature_names_path"))

    drop_cols = list(resolved.get("drop_cols", []) or [])
    drop_prefixes = list(resolved.get("drop_prefixes", []) or [])
    targets = list(resolved.get("target_cols", []) or [])

    if registry and resolved.get("drop_metadata_fields", True):
        _append_unique(drop_cols, registry.get("metadata", {}).get("fields", []))

    if registry and resolved.get("drop_diagnostic_fields", True):
        _append_unique(drop_cols, registry.get("diagnostics", {}).get("fields", []))

    if registry and resolved.get("drop_unused_y_fields", True):
        y_registry = registry.get("y", {})
        y_exact_groups = (
            y_registry.get("coi_fields", []),
            y_registry.get("bus_frequency_summary_fields", []),
            y_registry.get("bus_voltage_summary_fields", []),
        )
        for group in y_exact_groups:
            unused_fields = [field for field in group if field not in targets]
            _append_unique(drop_cols, unused_fields)

        for prefix_value in y_registry.get("prefixes", {}).values():
            if not _target_uses_prefix(targets, prefix_value):
                _append_unique(drop_prefixes, [prefix_value])

    if registry:
        prefix_map = registry.get("y", {}).get("prefixes", {})
        for prefix_key in resolved.get("drop_prefix_keys", []) or []:
            prefix_value = prefix_map.get(prefix_key)
            if prefix_value:
                _append_unique(drop_prefixes, [prefix_value])

    if registry and resolved.get("use_registry_feature_allowlist", False):
        allowed_cols, allowed_prefixes = _collect_registry_input_allowlist(
            registry,
            include_groups=resolved.get("registry_feature_groups"),
            group_overrides=resolved.get("registry_group_overrides"),
        )
        resolved["allowed_feature_cols"] = allowed_cols
        resolved["allowed_feature_prefixes"] = allowed_prefixes

    resolved["drop_cols"] = drop_cols
    resolved["drop_prefixes"] = drop_prefixes
    return resolved


# -----------------------------
# Naming / checkpoint helpers
# -----------------------------

def format_run_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "+".join(format_run_value(item) for item in value)
    if isinstance(value, dict):
        if "name" in value:
            return format_run_value(value["name"])
        return "dict"
    return str(value)


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def checkpoint_filenames(model_type: str) -> Dict[str, str]:
    model_slug = sanitize_name(model_type).lower()
    return {
        "best_state_dict": f"{model_slug}_state_dict_best.pt",
        "final_state_dict": f"{model_slug}_state_dict.pt",
    }


# -----------------------------
# Model-kwargs resolution
# -----------------------------

def build_model_kwargs(
    model_cfg: Dict,
    feature_cols: Sequence[str],
    *,
    train_cfg: Optional[Dict] = None,
    feature_name_registry: Optional[Dict] = None,
) -> Dict[str, Any]:
    train_cfg = train_cfg or {}
    group_head_indices = parse_group_head_indices(
        model_cfg.get("group_head_indices", train_cfg.get("head_indices"))
    )
    attention_hidden_dim = model_cfg.get("attention_hidden_dim")
    if attention_hidden_dim is not None:
        attention_hidden_dim = int(attention_hidden_dim)

    resolved_model_cfg = deepcopy(model_cfg)
    u_feature_spec = resolved_model_cfg.get("u_feature_spec")
    if u_feature_spec:
        if not feature_name_registry:
            raise ValueError("u_feature_spec requires 'feature_names_path' in the data config.")
        x_sched = feature_name_registry.get("x_sched", {})
        resolved_u_cols: list[str] = []
        if "items" in u_feature_spec:
            scalar_fields = set(x_sched.get("scalar_fields", []))
            prefix_map = x_sched.get("prefixes", {})
            for item in u_feature_spec.get("items", []):
                kind = item.get("kind", "scalar")
                if kind == "scalar":
                    field = item["field"]
                    if field not in scalar_fields:
                        raise KeyError(f"Unknown x_sched scalar field '{field}' in u_feature_spec.")
                    _append_unique(resolved_u_cols, [field])
                elif kind == "prefixed":
                    prefix_name = item["name"]
                    prefix_value = prefix_map.get(prefix_name)
                    if prefix_value is None:
                        raise KeyError(f"Unknown x_sched prefix '{prefix_name}' in u_feature_spec.")
                    _append_unique(resolved_u_cols, [f"{prefix_value}{item['index']}"])
                else:
                    raise ValueError(f"Unsupported u_feature_spec item kind '{kind}'.")
        else:
            _append_unique(resolved_u_cols, u_feature_spec.get("scalar_fields", []))
            prefix_map = x_sched.get("prefixes", {})
            for prefix_entry in u_feature_spec.get("prefixes", []):
                prefix_name = prefix_entry["name"]
                prefix_value = prefix_map.get(prefix_name)
                if prefix_value is None:
                    raise KeyError(f"Unknown x_sched prefix '{prefix_name}' in u_feature_spec.")
                indices = prefix_entry.get("indices")
                if indices:
                    _append_unique(resolved_u_cols, [f"{prefix_value}{idx}" for idx in indices])
                else:
                    matching_cols = [col for col in feature_cols if str(col).startswith(prefix_value)]
                    if not matching_cols:
                        raise KeyError(
                            f"No feature columns matched x_sched prefix '{prefix_name}' ({prefix_value})."
                        )
                    _append_unique(resolved_u_cols, matching_cols)
        resolved_model_cfg["u_feature_cols"] = resolved_u_cols

    return {
        "shared_sizes": parse_size_list(resolved_model_cfg.get("shared_sizes")),
        "head_sizes": parse_size_list(resolved_model_cfg.get("head_sizes")),
        "hidden_sizes": parse_size_list(resolved_model_cfg.get("hidden_sizes")),
        "group_shared_sizes": parse_size_list(resolved_model_cfg.get("group_shared_sizes")),
        "dropout": float(resolved_model_cfg.get("dropout", 0.0)),
        "group_head_indices": group_head_indices,
        "kan_grid_size": int(resolved_model_cfg.get("kan_grid_size", train_cfg.get("kan_grid_size", 8))),
        "kan_grid_min": float(resolved_model_cfg.get("kan_grid_min", train_cfg.get("kan_grid_min", -1.0))),
        "kan_grid_max": float(resolved_model_cfg.get("kan_grid_max", train_cfg.get("kan_grid_max", 1.0))),
        "attention_hidden_dim": attention_hidden_dim,
        "attention_temperature": float(resolved_model_cfg.get("attention_temperature", 1.0)),
        "attention_dropout": float(resolved_model_cfg.get("attention_dropout", 0.0)),
        "u_feature_idx": resolve_feature_indices(
            feature_cols,
            idx_key="u_feature_idx",
            col_key="u_feature_cols",
            cfg=resolved_model_cfg,
        ),
        "v_feature_idx": resolve_feature_indices(
            feature_cols,
            idx_key="v_feature_idx",
            col_key="v_feature_cols",
            cfg=resolved_model_cfg,
        ),
        "activation": str(resolved_model_cfg.get("activation", "relu")),
    }


# -----------------------------
# Retained-model export helpers
# -----------------------------

CORE_ARTIFACTS = [
    "run_config.yaml",
    "model.txt",
    "model_stats.json",
    "artifact_manifest.json",
    "x_scaler.pkl",
    "y_scaler.pkl",
    "metrics_by_target.csv",
    "metrics_summary.json",
    "rmse_results.txt",
    "training_summary.txt",
]


def _parse_filters(values):
    filters = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid filter '{value}'. Expected field=value.")
        key, raw = value.split("=", 1)
        filters.append((key.strip(), raw.strip()))
    return filters


def _coerce_filter_value(series, raw: str):
    import pandas as pd

    if pd.api.types.is_bool_dtype(series):
        return raw.lower() in {"1", "true", "yes", "y"}
    if pd.api.types.is_integer_dtype(series):
        return int(raw)
    if pd.api.types.is_float_dtype(series):
        return float(raw)
    return raw


def _apply_filters(df, filters):
    out = df
    for key, raw in filters:
        if key not in out.columns:
            raise KeyError(f"Unknown filter column '{key}'.")
        out = out[out[key] == _coerce_filter_value(out[key], raw)]
    return out


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        import shutil

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _checkpoint_artifacts(source_run_dir: Path) -> list[str]:
    import json

    manifest_path = source_run_dir / "artifact_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        checkpoint_files = manifest.get("checkpoint_files", {})
        names = [checkpoint_files.get("best_state_dict"), checkpoint_files.get("final_state_dict")]
        return [name for name in names if name]
    return ["vis_mlp_state_dict_best.pt", "vis_mlp_state_dict.pt"]


def export_retained_model_bundle(
    *,
    dest_dir: str | Path,
    source_run_dir: str | Path | None = None,
    summary_csv: str | Path | None = None,
    filters: Optional[Sequence[str]] = None,
    metric: str = "agg_rmse_mean",
    descending: bool = False,
    note: str = "",
) -> Dict[str, Any]:
    import json
    import pandas as pd

    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_payload: Dict[str, Any] = {
        "metric": str(metric),
        "descending": bool(descending),
        "filters": list(filters or []),
        "note": str(note),
    }

    selected_row = None
    if source_run_dir is not None:
        run_dir = Path(source_run_dir)
    else:
        if summary_csv is None:
            raise ValueError("Provide either source_run_dir or summary_csv.")
        df = pd.read_csv(summary_csv)
        df = _apply_filters(df, _parse_filters(filters))
        if df.empty:
            raise ValueError("No runs matched the requested filters.")
        if metric not in df.columns:
            raise KeyError(f"Selection metric '{metric}' not found in {summary_csv}.")
        df = df.sort_values(metric, ascending=not descending)
        selected_row = df.iloc[0].to_dict()
        run_dir = Path(str(selected_row["run_dir"]))
        selection_payload["selected_row"] = selected_row

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in CORE_ARTIFACTS + _checkpoint_artifacts(run_dir):
        _copy_if_exists(run_dir / name, artifacts_dir / name)

    selection_payload["source_run_dir"] = str(run_dir)
    with (out_dir / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2)

    return selection_payload
