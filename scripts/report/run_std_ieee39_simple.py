from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
import andes

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_generation.run_sims import run_ed_dispatch

ROOT = Path(__file__).resolve().parents[2]
# CASE_PATH = ROOT / "data_generation" / "andes_cases" / "ieee39_full.xlsx"
# import andes
CASE_PATH = andes.get_case("ieee39/ieee39_full.xlsx")
DEFAULT_CONFIG = ROOT / "configs" / "presentation_vis_case.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_base_operating_point_scale(ss, base_scale: float, *, scale_pv: bool = True) -> None:
    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    if not scale_pv:
        return
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale
    if ss.Slack.n:
        ss.Slack.p0.v[0] = ss.Slack.p0.v[0] * base_scale
        ss.Slack.q0.v[0] = ss.Slack.q0.v[0] * base_scale


def _configure_pq_model(ss) -> None:
    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.config.p2i = 0
    ss.PQ.config.q2i = 0
    ss.PQ.config.pq2z = 0
    ss.PV.config.allow_adjust = 0


def _select_pq_targets(ss, *, zone_id: str | None) -> list[str]:
    pq_names = list(ss.PQ.name.v) if ss.PQ.n else []
    if not zone_id:
        return [str(name) for name in pq_names]
    owners = [str(o) for o in list(ss.PQ.owner.v)] if ss.PQ.n else []
    out = [str(name) for name, owner in zip(pq_names, owners) if str(owner) == str(zone_id)]
    return out


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


def _line_idx_from_uid(ss, uid: int) -> int:
    idx_vals = list(getattr(getattr(ss.Line, "idx", None), "v", []) or [])
    for pos, idx_val in enumerate(idx_vals):
        if int(idx_val) == int(uid):
            return pos
    raise ValueError(f"Line uid {uid} not found in case.")


def _configure_tds(ss, *, t_end: float, t_step: float) -> None:
    ss.TDS.config.no_tqdm = True
    ss.TDS.config.criteria = 0
    ss.TDS.config.tol = 1e-3
    ss.TDS.config.tf = float(t_end)
    ss.TDS.config.tstep = float(t_step)
    ss.TDS.config.fixt = 0
    ss.TDS.config.method = "backeuler"
    ss.TDS.config.honest = 0
    ss.TDS.config.max_iter = 80
    ss.TDS.config.shrinkt = 2


def _extract_coi(ss) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ss.TDS.load_plotter()
    plotter = ss.TDS.plotter
    time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
    coi_indices = list(plotter.find("omega COI", idx_only=True))
    if not coi_indices:
        raise RuntimeError("COI frequency channel not found in plotter.")
    f0 = float(getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0)
    f_coi_hz = np.asarray(plotter.get_values([int(coi_indices[0])]), dtype=float).reshape(-1) * f0
    if f_coi_hz.size != time.size:
        raise RuntimeError("COI frequency series length mismatch.")
    rocof = np.gradient(f_coi_hz, time, edge_order=2 if time.size > 2 else 1)
    return time, f_coi_hz, rocof


def _run_simulation(
    *,
    disturbance_family: str,
    contingency_mode: str,
    load_step_time: float,
    disturbance_scale: float,
    base_load_scale: float,
    t_end: float,
    t_step: float,
    zone_id: str | None,
    contingency_label: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict[str, Any]]:
    ss = andes.load(str(CASE_PATH), setup=False)
    ss.config.freq = float(50)

    _apply_base_operating_point_scale(ss, base_load_scale, scale_pv=True)

    ed_cfg = {
        "enable": True,
        "solver": "Gurobi",
        "verbose": False,
        "line_limits_enable": False,
        "cost_table_path": "configs/shared/ieee39_regcv1_dispatch_costs.yaml",
    }
    try:
        run_ed_dispatch(ss, ed_cfg, ibr_idx=[])
    except Exception as exc:
        print(f"[std_ieee39_simple] Warning: ED failed ({exc}); using base setpoints.")

    _configure_pq_model(ss)

    load_step_enabled = disturbance_family in {"global_mismatch", "zone_mismatch", "mixed"}
    line_outage_enabled = disturbance_family in {"line_outage", "mixed"} or contingency_mode == "single_line"

    if load_step_enabled:
        targets = _select_pq_targets(ss, zone_id=zone_id)
        for dev in targets:
            ss.add(model="Alter", param_dict=dict(t=load_step_time, model="PQ", dev=dev, src="Ppf", attr="v", method="*", amount=disturbance_scale))
            ss.add(model="Alter", param_dict=dict(t=load_step_time, model="PQ", dev=dev, src="Qpf", attr="v", method="*", amount=disturbance_scale))

    if line_outage_enabled:
        line_uid = _parse_line_uid(contingency_label)
        line_idx = _line_idx_from_uid(ss, line_uid)
        ss.add(model="Toggle", param_dict={"t": load_step_time, "model": "Line", "dev": line_idx})

    ss.setup()
    ss.PFlow.run()

    _configure_tds(ss, t_end=t_end, t_step=t_step)
    ss.TDS.init()
    success = bool(ss.TDS.run())
    time_s, f_coi_hz, rocof = _extract_coi(ss)
    tds_info = {
        "success": success,
        "err_msg": str(getattr(ss.TDS, "err_msg", "")),
        "last_converged": float(getattr(ss.TDS, "last_converged", 0.0) or 0.0),
        "niter": int(getattr(ss.TDS, "niter", 0) or 0),
        "busted": bool(getattr(ss.TDS, "busted", False)),
    }
    return time_s, f_coi_hz, rocof, success, tds_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IEEE39 standard case with presentation disturbance and export COI trace.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to presentation_vis_case.yaml")
    parser.add_argument("--out", default=None, help="Optional output directory override")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    run_cfg = dict(cfg.get("run", {}) or {})
    system_cfg = dict(cfg.get("system", {}) or {})
    logging_cfg = dict(cfg.get("logging", {}) or {})

    case_label = str(run_cfg.get("case_label", "vis_case")).strip() or "vis_case"
    output_root = Path(str(run_cfg.get("output_root", "results/presentation_vis")))
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    out_dir = Path(args.out) if args.out else (output_root / case_label / "std_ieee39_simple")
    out_dir.mkdir(parents=True, exist_ok=True)

    disturbance_family = str(system_cfg.get("disturbance_family", "global_mismatch")).strip().lower()
    contingency_mode = str(system_cfg.get("contingency_mode", "none")).strip().lower()
    load_step_time = float(system_cfg.get("load_step_time_s", 1.0))
    disturbance_scale = float(system_cfg.get("disturbance_scale", 1.2))
    base_load_scale = float(system_cfg.get("base_load_scale", 0.75))

    zone_id = None
    if disturbance_family == "zone_mismatch":
        zone_val = system_cfg.get("zone_id")
        if zone_val is not None:
            zone_id = str(zone_val)

    t_window = logging_cfg.get("trajectory_time_window_s", [0.0, 20.0])
    if isinstance(t_window, (list, tuple)) and len(t_window) >= 2:
        t_end = float(max(t_window))
    else:
        t_end = 20.0
    base_step = float(logging_cfg.get("trajectory_sampling_step", 0.05))

    time_s, f_coi_hz, rocof, last_success, last_info = _run_simulation(
        disturbance_family=disturbance_family,
        contingency_mode=contingency_mode,
        load_step_time=load_step_time,
        disturbance_scale=disturbance_scale,
        base_load_scale=base_load_scale,
        t_end=t_end,
        t_step=base_step,
        zone_id=zone_id,
        contingency_label=system_cfg.get("contingency_label"),
    )
    last_time = float(time_s[-1]) if time_s.size else 0.0
    if last_time < t_end - base_step:
        print(
            f"[std_ieee39_simple] Warning: simulation stopped at t={last_time:.3f}s "
            f"with t_step={base_step} (success={last_success})."
        )
        if last_info:
            print(
                f"[std_ieee39_simple] TDS info: last_converged={last_info.get('last_converged')}, "
                f"niter={last_info.get('niter')}, busted={last_info.get('busted')}, err_msg={last_info.get('err_msg')}"
            )

    if time_s is None or f_coi_hz is None or rocof is None:
        raise RuntimeError("Simulation did not produce COI traces.")

    if np.allclose(f_coi_hz, f_coi_hz[0], atol=1e-6):
        print("[std_ieee39_simple] Warning: COI frequency did not deviate from the initial value.")
    if last_time < t_end - base_step:
        print(f"[std_ieee39_simple] Warning: simulation ended early at t={last_time:.3f}s (target {t_end:.3f}s).")
    if not last_success:
        print("[std_ieee39_simple] Warning: TDS did not converge; exported available trace.")

    out_path = out_dir / "coi_trace_ieee39_full.csv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("time_s,f_coi_hz,rocof_coi_hz_per_s\n")
        for t, f_hz, r in zip(time_s, f_coi_hz, rocof):
            f.write(f"{t},{f_hz},{r}\n")

    print(f"[std_ieee39_simple] Wrote {out_path}")


if __name__ == "__main__":
    main()
