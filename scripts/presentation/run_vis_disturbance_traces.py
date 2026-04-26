from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import andes

ROOT = Path(__file__).resolve().parents[2]

# Ensure repo root for local imports
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

from scheduling.replay_validation import (  # type: ignore  # noqa: E402
    _add_line_trip_contingency,
    _add_load_step_contingency,
    _apply_base_scale,
    _apply_dispatch,
    _apply_md,
    _configure_tds,
    _setup_pq_model,
)

FORMULATION_ID_BY_CODE = {
    "A": "ed",
    "B": "ed_line",
    "C": "ed_line_n1",
    "D": "ed_surrogate",
    "E": "ed_line_n1_surrogate",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _parse_line_uid(label: Any) -> int:
    if label is None:
        raise ValueError("contingency_label is required for line outage.")
    text = str(label).strip()
    if text.isdigit():
        return int(text)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse line uid from '{text}'.")
    return int(digits)


def _select_pq_targets(ss, zone_id: str | None) -> list[str] | None:
    if not zone_id:
        return None
    pq_names = list(ss.PQ.name.v) if ss.PQ.n else []
    owners = [str(o) for o in list(ss.PQ.owner.v)] if ss.PQ.n else []
    targets = [str(name) for name, owner in zip(pq_names, owners) if str(owner) == str(zone_id)]
    return targets or None


def _extract_frequency_trace(ss) -> np.ndarray:
    ss.TDS.load_plotter()
    plotter = ss.TDS.plotter
    time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
    coi_indices = list(plotter.find("omega COI", idx_only=True))
    if not coi_indices:
        raise RuntimeError("Could not find 'omega COI' trace in the ANDES plotter.")
    f0 = float(getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0)
    f_coi = np.asarray(plotter.get_values([int(coi_indices[0])]), dtype=float).reshape(-1) * f0
    rocof = np.gradient(f_coi, time, axis=0)
    delta_f = f_coi - f0
    return np.column_stack([time, f_coi, delta_f, rocof])


def _load_suite_row(case_root: Path, bounds_case: str, formulation_id: str) -> dict[str, Any]:
    suite_path = case_root / bounds_case / "raw" / "suite_summary.json"
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite summary not found: {suite_path}")
    payload = _load_json(suite_path)
    rows = list(payload.get("rows") or [])
    for row in rows:
        if str(row.get("formulation_id", "")).strip() == formulation_id:
            return row
    raise ValueError(f"Formulation id '{formulation_id}' not found in {suite_path}")


def _run_trace_for_case(*, opt_cfg: dict[str, Any], summary: dict[str, Any],
                        contingency: dict[str, Any], tds_cfg: dict[str, Any],
                        pq_targets: list[str] | None) -> np.ndarray:
    ss = andes.load(str(opt_cfg["system"]["case"]), setup=False)
    ss.config.freq = float(opt_cfg.get("system", {}).get("frequency_hz", 50.0))
    _apply_base_scale(ss, float(opt_cfg["scenario"]["base_scale"]))
    _apply_dispatch(ss, np.asarray(summary.get("dispatch_summary", {}).get("pg_opt", []), dtype=float))
    _apply_md(
        ss,
        np.asarray(summary.get("dispatch_summary", {}).get("m_opt", []), dtype=float),
        np.asarray(summary.get("dispatch_summary", {}).get("d_opt", []), dtype=float),
    )
    _setup_pq_model(ss)

    if contingency.get("type") == "load_step":
        _add_load_step_contingency(
            ss,
            time=float(contingency["time"]),
            scale=float(contingency["scale"]),
            pq_targets=pq_targets,
        )
    if contingency.get("type") == "line_trip":
        _add_line_trip_contingency(
            ss,
            time=float(contingency["time"]),
            line_uid=int(contingency["line_uid"]),
        )
    if contingency.get("type") == "mixed":
        _add_load_step_contingency(
            ss,
            time=float(contingency["time"]),
            scale=float(contingency["scale"]),
            pq_targets=pq_targets,
        )
        _add_line_trip_contingency(
            ss,
            time=float(contingency["time"]),
            line_uid=int(contingency["line_uid"]),
        )

    ss.setup()
    ss.PFlow.run()
    _configure_tds(ss, tds_cfg)
    ss.TDS.init()
    ss.TDS.run()
    return _extract_frequency_trace(ss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export VIS frequency traces for three disturbance types.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "presentation_vis_case.yaml"), help="presentation_vis_case.yaml")
    parser.add_argument("--bounds-case", default="default", help="Bounds case folder (default/tight)")
    parser.add_argument("--formulation", default="E", help="Formulation code (A-E) or formulation id")
    parser.add_argument("--out", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    run_cfg = dict(cfg.get("run", {}) or {})
    system_cfg = dict(cfg.get("system", {}) or {})
    logging_cfg = dict(cfg.get("logging", {}) or {})

    case_label = str(run_cfg.get("case_label", "vis_case")).strip() or "vis_case"
    output_root = Path(str(run_cfg.get("output_root", "results/presentation_vis")))
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    case_root = output_root / case_label

    formulation = str(args.formulation).strip()
    formulation_id = FORMULATION_ID_BY_CODE.get(formulation.upper(), formulation)

    row = _load_suite_row(case_root, args.bounds_case, formulation_id)
    summary_path = _resolve(str(row.get("summary_json")))
    summary = _load_json(summary_path)

    opt_cfg_path = summary.get("config_path") or row.get("config_path")
    if not opt_cfg_path:
        raise ValueError("Missing optimization config path in suite summary.")
    opt_cfg = _load_yaml(_resolve(str(opt_cfg_path)))

    load_step_time = float(system_cfg.get("load_step_time_s", opt_cfg.get("scenario", {}).get("load_step_time", 1.0)))
    load_step_scale = float(system_cfg.get("disturbance_scale", opt_cfg.get("scenario", {}).get("step_scale", 1.2)))
    line_uid = _parse_line_uid(system_cfg.get("contingency_label"))

    zone_id = system_cfg.get("zone_id") if str(system_cfg.get("disturbance_family", "global_mismatch")).strip().lower() == "zone_mismatch" else None

    t_window = logging_cfg.get("trajectory_time_window_s", [0.0, 20.0])
    t_end = float(max(t_window)) if isinstance(t_window, (list, tuple)) and len(t_window) >= 2 else 20.0
    t_step = float(logging_cfg.get("trajectory_sampling_step", 0.05))
    tds_cfg = {
        "t_end": t_end,
        "t_step": t_step,
        "method": "backeuler",
        "tol": 1e-3,
        "criteria": 0,
        "honest": 1,
        "max_iter": 35,
        "shrinkt": 1,
        "no_tqdm": True,
    }

    out_dir = Path(args.out) if args.out else (case_root / args.bounds_case / "raw" / "formulations" / case_label / "vis_disturbance_traces")
    out_dir.mkdir(parents=True, exist_ok=True)

    contingencies = [
        ("load_step", {"type": "load_step", "time": load_step_time, "scale": load_step_scale}),
        ("line_trip", {"type": "line_trip", "time": load_step_time, "line_uid": line_uid}),
        ("mixed", {"type": "mixed", "time": load_step_time, "scale": load_step_scale, "line_uid": line_uid}),
    ]

    meta = {
        "formulation_id": formulation_id,
        "summary_json": str(summary_path),
        "opt_config": str(opt_cfg_path),
        "out_dir": str(out_dir),
        "contingencies": [],
    }

    for name, cont in contingencies:
        pq_targets = None
        if cont["type"] in {"load_step", "mixed"} and zone_id is not None:
            ss_tmp = andes.load(str(opt_cfg["system"]["case"]), setup=False)
            pq_targets = _select_pq_targets(ss_tmp, str(zone_id))
        trace = _run_trace_for_case(
            opt_cfg=opt_cfg,
            summary=summary,
            contingency=cont,
            tds_cfg=tds_cfg,
            pq_targets=pq_targets,
        )
        out_path = out_dir / f"vis_trace_{name}.csv"
        header = "time_s,f_coi_hz,delta_f_coi_hz,rocof_coi_hz_per_s"
        np.savetxt(out_path, trace, delimiter=",", header=header, comments="")
        meta["contingencies"].append({"name": name, **cont, "trace_csv": str(out_path)})
        print(f"[vis_traces] Wrote {out_path}")

    with (out_dir / "trace_summary.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
