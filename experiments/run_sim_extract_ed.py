from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import andes
import numpy as np
import yaml

from data_generation.extract_metrics import export_plotter_all, extract_simulation_row
from scheduling.economic_dispatch import ed_calculation


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _repeat_or_validate(values: Sequence[float], n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"{name} must have length 1 or {n} (got {arr.size}).")
    return arr


def add_measurement_devices(ss):
    """Add BusROCOF + PMU at every bus (matches analyze_sim notebook)."""
    for bus in ss.Bus.as_df().idx.values:
        ss.add(
            model="BusROCOF",
            idx=f"BusROCOF_{bus}",
            name=f"BusROCOF {bus}",
            param_dict=dict(bus=bus, Tr=0.02, Tw=0.1, Tf=0.02),
        )

    existing = list(ss.PMU.as_df().bus.values) if ss.PMU.n > 0 else []
    for bus in ss.Bus.as_df().idx.values:
        if bus not in existing:
            ss.add(model="PMU", param_dict=dict(bus=bus))


def _run_single_sim(
    cfg: Dict,
    *,
    rng: np.random.Generator,
    case_path: str,
    base_scale: float,
    step_scale: float,
    cost_cfg: Dict,
    export_plotter: bool,
    plotter_dir: Path | None,
) -> Tuple[Dict, np.ndarray, float, float | None]:
    ss = andes.load(case_path, setup=False)
    ss.config.freq = float(50)
    add_measurement_devices(ss)

    regcv1_ids = ss.REGCV1.name.v
    M_vec = rng.uniform(cfg["ibr"]["M_range"][0], cfg["ibr"]["M_range"][1], size=len(regcv1_ids))
    D_vec = rng.uniform(cfg["ibr"]["D_range"][0], cfg["ibr"]["D_range"][1], size=len(regcv1_ids))

    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale

    ss.REGCV1.M.v, ss.REGCV1.D.v = M_vec, D_vec

    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.config.p2i = 0
    ss.PQ.config.q2i = 0
    ss.PQ.config.pq2z = 0

    pq_p_before = np.asarray(ss.PQ.p0.v, dtype=float).copy()
    pq_q_before = np.asarray(ss.PQ.q0.v, dtype=float).copy()

    a = _repeat_or_validate(cost_cfg["a"], ss.PV.n + ss.Slack.n, "ed_costs.a")
    b = _repeat_or_validate(cost_cfg["b"], ss.PV.n + ss.Slack.n, "ed_costs.b")
    c = _repeat_or_validate(cost_cfg["c"], ss.PV.n + ss.Slack.n, "ed_costs.c")
    Pg_opt, ed_cost, ed_lam = ed_calculation(ss, a=a, b=b, c=c)

    for gen_idx, Pg in enumerate(Pg_opt, start=1):
        if gen_idx in ss.PV.idx.v:
            ss.PV.p0.v[gen_idx-1] = Pg
        elif gen_idx in ss.Slack.idx.v:
            ss.Slack.p0.v[0] = Pg
        else:
            raise ValueError(f"Generator index {gen_idx} not found in PV or Slack.")

    ss.setup()
    ss.PFlow.run()

    ss.TDS.config.no_tqdm = bool(cfg["tds"].get("no_tqdm", True))
    ss.TDS.config.criteria = int(cfg["tds"].get("criteria", 0))
    ss.TDS.config.tol = float(cfg["tds"].get("tol", 1e-6))
    ss.TDS.config.tf = float(cfg["tds"]["t_end"])
    ss.TDS.config.tstep = float(cfg["tds"]["t_step"])
    ss.TDS.config.fixt = int(cfg["tds"].get("fixt", 0))
    ss.TDS.config.method = str(cfg["tds"].get("method", "backeuler"))
    ss.TDS.config.honest = int(cfg["tds"].get("honest", 0))
    ss.TDS.config.max_iter = int(cfg["tds"].get("max_iter", 35))
    ss.TDS.config.shrinkt = int(cfg["tds"].get("shrinkt", 1))

    ss.TDS.init()

    for uid in range(ss.PQ.n):
        p = ss.PQ.p0.v[uid] * step_scale
        q = ss.PQ.q0.v[uid] * step_scale
        ss.PQ.p0.v[uid], ss.PQ.Ppf.v[uid] = p, p
        ss.PQ.q0.v[uid], ss.PQ.Qpf.v[uid] = q, q

    success = bool(ss.TDS.run())
    ss.TDS.load_plotter()

    pq_p_after = np.asarray(ss.PQ.Ppf.v, dtype=float).copy()
    pq_q_after = np.asarray(ss.PQ.Qpf.v, dtype=float).copy()

    plotter_csv = None
    if export_plotter and plotter_dir is not None:
        plotter_dir.mkdir(parents=True, exist_ok=True)
        plotter_csv = str(plotter_dir / "plotter_single.csv")
        export_plotter_all(ss.TDS.plotter, plotter_csv)

    row = extract_simulation_row(
        ss=ss,
        base_load_scale=base_scale,
        load_step_scale=step_scale,
        load_step_time=float(cfg["tds"]["load_step_time"]),
        pq_names=list(ss.PQ.name.v) if ss.PQ.n else [],
        pq_p_before=pq_p_before,
        pq_q_before=pq_q_before,
        pq_p_after=pq_p_after,
        pq_q_after=pq_q_after,
        M_vec=M_vec,
        D_vec=D_vec,
        success=success,
        plotter_csv=plotter_csv,
    )

    return row, Pg_opt, float(ed_cost), ed_lam


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single sim + extract features + ED.")
    parser.add_argument("--config", default="experiments/generation.yaml", help="Path to sim YAML.")
    parser.add_argument("--cost-config", default="scheduling/mtlsh_convex.yaml", help="Path to cost YAML.")
    parser.add_argument("--base-scale", type=float, default=1.0, help="Base load scale.")
    parser.add_argument("--step-scale", type=float, default=.9, help="Load step scale.")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    cost_cfg = _load_yaml(Path(args.cost_config))
    if "ed_costs" not in cost_cfg:
        raise KeyError("Missing ed_costs in cost-config YAML.")

    andes.config_logger(stream_level=int(cfg.get("stream_level", 30)))
    case_path = cfg["case"]

    output_dir = Path(cfg.get("output_dir", "experiments"))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / cfg.get("output_csv", "simulation_results.csv")

    plotter_cfg = cfg.get("plotter", {})
    export_plotter = bool(plotter_cfg.get("export", False))
    plotter_dir = None
    if export_plotter:
        plotter_dir = output_dir / plotter_cfg.get("subdir", "plotter")

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    row, Pg_opt, ed_cost, ed_lam = _run_single_sim(
        cfg,
        rng=rng,
        case_path=case_path,
        base_scale=float(args.base_scale),
        step_scale=float(args.step_scale),
        cost_cfg=cost_cfg["ed_costs"],
        export_plotter=export_plotter,
        plotter_dir=plotter_dir,
    )

    row["sim_id"] = 0
    row["seed"] = int(cfg.get("seed", 42))
    row["ed_cost"] = float(ed_cost)
    row["ed_lambda"] = float(ed_lam) if ed_lam is not None else np.nan
    for i, val in enumerate(Pg_opt, start=1):
        row[f"ed_Pg_{i}"] = float(val)

    fieldnames = list(row.keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"Wrote results to {csv_path}")


if __name__ == "__main__":
    main()
