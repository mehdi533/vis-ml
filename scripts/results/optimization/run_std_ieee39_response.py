from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _repo_root_from(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path.parents[2]


ROOT = _repo_root_from(Path(__file__).resolve())
DEFAULT_PRESENTATION_CONFIG = ROOT / "configs/presentation/presentation_vis_case.yaml"
DEFAULT_GEN_CONFIG = ROOT / "configs/data_generation/generation.yaml"
STD_IEEE39_CASE = "data_generation/andes_cases/ieee39_full.xlsx"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _format_line_label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    if text.isdigit():
        return f"Line_{int(text)}"
    return text


def _build_std_config(presentation_cfg: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    base_cfg = _load_yaml(DEFAULT_GEN_CONFIG)

    run_cfg = dict(presentation_cfg.get("run", {}) or {})
    system_cfg = dict(presentation_cfg.get("system", {}) or {})

    case_label = str(run_cfg.get("case_label", "vis_case")).strip() or "vis_case"
    output_root = Path(str(run_cfg.get("output_root", "results/presentation_vis")))
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    output_dir = (output_root / case_label / "std_ieee39").resolve()

    disturbance_family = str(system_cfg.get("disturbance_family", "global_mismatch")).strip().lower()
    contingency_mode = str(system_cfg.get("contingency_mode", "none")).strip().lower()

    load_step_enabled = disturbance_family in {"global_mismatch", "zone_mismatch", "mixed"}
    line_outage_enabled = disturbance_family in {"line_outage", "mixed"} or contingency_mode == "single_line"

    load_step_time = float(system_cfg.get("load_step_time_s", 1.0))
    disturbance_scale = float(system_cfg.get("disturbance_scale", 1.2))
    base_load_scale = float(system_cfg.get("base_load_scale", 0.75))

    owners = []
    if disturbance_family == "zone_mismatch":
        zone_id = system_cfg.get("zone_id")
        if zone_id is not None and str(zone_id).strip():
            owners = [str(zone_id).strip()]

    line_label = None
    if line_outage_enabled:
        line_label = _format_line_label(system_cfg.get("contingency_label"))
        if line_label is None:
            raise ValueError("contingency_label is required for line-outage settings.")

    overrides = {
        "case": STD_IEEE39_CASE,
        "output_dir": str(output_dir),
        "output_csv": "simulation_results.csv",
        "seed": int(run_cfg.get("random_seed", 42)),
        "n_sims": 1,
        "workers": 1,
        "stream_level": 50,
        "base_load_scale": {"low": base_load_scale, "high": base_load_scale},
        "contingency": {
            "load_step": {
                "enable": bool(load_step_enabled),
                "time": load_step_time,
                "scale": {"low": disturbance_scale, "high": disturbance_scale},
            },
            "line_n1": {
                "enable": bool(line_outage_enabled),
                "trip_time": load_step_time,
                "line_ids": [line_label] if line_label else [],
                "max_lines": 1,
            },
        },
        "load": {
            "owners": owners,
        },
        "ed": {
            "enable": False,
            "line_limits_enable": False,
        },
        "ibr": {
            "n_ibr": 0,
            "indices": [],
        },
        "debug": {
            "save_coi_traces": True,
            "coi_trace_dir": str(output_dir),
        },
    }

    merged = _deep_merge(base_cfg, overrides)

    config_path = output_dir / "raw" / "generated_configs" / "std_ieee39.yaml"
    return merged, config_path


def _run_command(cmd: list[str], *, cwd: Path) -> None:
    print("[std_ieee39] $", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _probe_python_bin(candidate: str) -> bool:
    try:
        out = subprocess.check_output(
            [candidate, "-c", "import numpy, andes; print(numpy.__version__)"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return False
    version = out.splitlines()[-1].strip()
    major = int(version.split(".", maxsplit=1)[0]) if version else 0
    return major < 2


def _select_python_bin() -> str:
    candidates: list[str] = []
    env_bin = os.environ.get("PYTHON_BIN")
    if env_bin:
        candidates.append(env_bin)
    candidates.append(str(Path(str(sys.executable)).resolve()))
    venv_env = os.environ.get("VIRTUAL_ENV")
    if venv_env:
        candidates.append(str(Path(venv_env) / "bin" / "python"))
    for name in ("python", "python3", "python3.11"):
        path = shutil.which(name)
        if path:
            candidates.append(path)
    for rel in ("../venv/bin/python", "venv/bin/python", ".venv/bin/python"):
        cand = (ROOT / rel).resolve()
        candidates.append(str(cand))

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if Path(cand).exists() and _probe_python_bin(cand):
            return cand

    raise RuntimeError(
        "No compatible python found (requires numpy<2 and andes importable). "
        "Create a clean venv with numpy<2 and andes, then retry."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a standard IEEE39 sync-gen response using presentation VIS settings.")
    parser.add_argument("--config", default=str(DEFAULT_PRESENTATION_CONFIG), help="Path to presentation_vis_case.yaml")
    args = parser.parse_args()

    presentation_cfg = _load_yaml(Path(args.config))
    std_cfg, cfg_path = _build_std_config(presentation_cfg)
    _write_yaml(cfg_path, std_cfg)

    python_bin = _select_python_bin()

    print(f"[std_ieee39] Using python: {python_bin}")
    cmd = [python_bin, str(ROOT / "data_generation" / "run_sims.py"), "--config", str(cfg_path)]
    _run_command(cmd, cwd=ROOT)

    print(f"[std_ieee39] Generated config: {cfg_path}")
    print(f"[std_ieee39] Output dir: {std_cfg['output_dir']}")


if __name__ == "__main__":
    main()
