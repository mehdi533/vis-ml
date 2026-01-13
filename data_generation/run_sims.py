import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
import yaml
import andes

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_metrics import export_plotter_all, extract_simulation_row


def _resolve_case_path(case: str) -> str:
    case_path = andes.get_case(case)
    if os.path.exists(case_path):
        return case_path
    if os.path.exists(case):
        return case
    raise FileNotFoundError(f"Case not found: {case}")


def _as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _build_fieldnames(pq_names: Sequence[str], n_ibr: int, include_plotter: bool) -> List[str]:
    fieldnames = [
        "sim_id",
        "seed",
        "success",
        "base_load_scale",
        "load_step_scale",
        "load_step_time",
        "base_load_p_total",
        "base_load_q_total",
        "DELTA_PQ_tot",
        "M_agg",
        "D_agg",
        "time_max_dev",
        "rocof_max_COI",
        "rocof_min_COI",
        "devdown_COI",
        "devup_COI",
    ]
    for i in range(1, n_ibr + 1):
        fieldnames.append(f"M_{i}")
        fieldnames.append(f"D_{i}")
    for name in pq_names:
        fieldnames.append(f"DELTA_P_{name}")
        fieldnames.append(f"DELTA_Q_{name}")
    for i in range(1, n_ibr + 1):
        fieldnames.append(f"Delta_P_IBR_{i}")
    if include_plotter:
        fieldnames.append("plotter_csv")
    return fieldnames


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_sim(cfg: Dict, sim_id: int, rng: np.random.Generator, case_path: str, pq_names: Sequence[str], regcv1_ids: Sequence[str], plotter_dir: Path):
    base_scale = rng.uniform(cfg["base_load_scale"]["low"], cfg["base_load_scale"]["high"])
    step_scale = rng.uniform(cfg["load_step_scale"]["low"], cfg["load_step_scale"]["high"])
    M_vec = rng.uniform(cfg["ibr"]["M_range"][0], cfg["ibr"]["M_range"][1], size=len(regcv1_ids))
    D_vec = rng.uniform(cfg["ibr"]["D_range"][0], cfg["ibr"]["D_range"][1], size=len(regcv1_ids))

    ss = andes.load(case_path, setup=False)
    ss.config.freq = float(50)

    # Apply base load scale
    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale
    
    ss.REGCV1.M.v, ss.REGCV1.D.v = M_vec, D_vec

    ss.PQ.config.p2p = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2q = 1
    ss.PQ.config.q2z = 0
    ss.PQ.config.pq2z = 0

    pq_p_before, pq_q_before = ss.PQ.p0.v, ss.PQ.q0.v  
    
    for dev in _as_list(cfg["load"].get("pq_names")):
        ss.add(model='Alter', param_dict=dict(t=cfg["tds"]["load_step_time"], model='PQ', dev=dev, src='Ppf', attr='v', method='*', amount=step_scale))
        ss.add(model='Alter', param_dict=dict(t=cfg["tds"]["load_step_time"], model='PQ', dev=dev, src='Qpf', attr='v', method='*', amount=step_scale))

    ss.setup()
    ss.PFlow.run()

    ss.TDS.config.no_tqdm = bool(cfg["tds"].get("no_tqdm", True))
    ss.TDS.config.criteria = int(cfg["tds"].get("criteria", 0))
    ss.TDS.config.tol = float(cfg["tds"].get("tol", 1e-3))
    ss.TDS.config.tf = float(cfg["tds"]["t_end"])
    ss.TDS.config.tstep = float(cfg["tds"]["t_step"])
    ss.TDS.config.fixt = int(cfg["tds"].get("fixt", 0))
    ss.TDS.config.method = str(cfg["tds"].get("method", "backeuler"))
    ss.TDS.config.honest = int(cfg["tds"].get("honest", 1))
    ss.TDS.config.max_iter = int(cfg["tds"].get("max_iter", 35))
    ss.TDS.config.shrinkt = int(cfg["tds"].get("shrinkt", 1))

    ss.TDS.init()
    success = bool(ss.TDS.run())
    ss.TDS.load_plotter()

    pq_p_after, pq_q_after = ss.PQ.Ppf.v, ss.PQ.Qpf.v  

    plotter_csv = None
    if plotter_dir is not None:
        plotter_csv = str(plotter_dir / f"plotter_{sim_id:05d}.csv")
        export_plotter_all(ss.TDS.plotter, plotter_csv)

    row = extract_simulation_row(
        ss=ss,
        base_load_scale=base_scale,
        load_step_scale=step_scale,
        load_step_time=float(cfg["tds"]["load_step_time"]),
        pq_names=pq_names,
        pq_p_before=pq_p_before,
        pq_q_before=pq_q_before,
        pq_p_after=pq_p_after,
        pq_q_after=pq_q_after,
        M_vec=M_vec,
        D_vec=D_vec,
        success=success,
        plotter_csv=plotter_csv,
    )

    row["sim_id"] = sim_id
    row["seed"] = int(cfg["seed"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ANDES sims and extract metrics.")
    parser.add_argument("--config", default="data_generation/generation.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    print("Loading config from ", args.config)
    cfg = load_config(args.config)
    andes.config_logger(stream_level=int(cfg.get("stream_level", 30)))

    rng = np.random.default_rng(int(cfg["seed"]))
    case_path = _resolve_case_path(cfg["case"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / cfg.get("output_csv", "simulation_results.csv")

    plotter_cfg = cfg.get("plotter", {})
    export_plotter = bool(plotter_cfg.get("export", True))
    plotter_dir = None
    if export_plotter:
        plotter_dir = output_dir / plotter_cfg.get("subdir", "plotter")
        plotter_dir.mkdir(parents=True, exist_ok=True)

    base_ss = andes.load(case_path, setup=False)
    pq_names = list(base_ss.PQ.name.v) if base_ss.PQ.n else []
    if cfg["load"].get("pq_names"):
        pq_names = [name for name in pq_names if name in cfg["load"]["pq_names"]]

    regcv1_ids = cfg["ibr"].get("indices") or []
    if not regcv1_ids:
        n_ibr = int(cfg["ibr"].get("n_ibr", 4))
        regcv1_ids = list(base_ss.REGCV1.idx.v)[:n_ibr]
    if not regcv1_ids:
        raise ValueError("No REGCV1 entries found to assign M/D parameters.")

    fieldnames = _build_fieldnames(pq_names, len(regcv1_ids), export_plotter)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for sim_id in range(int(cfg["n_sims"])):
            row = run_single_sim(cfg, sim_id, rng, case_path, pq_names, regcv1_ids, plotter_dir)
            for name in fieldnames:
                row.setdefault(name, np.nan)
            writer.writerow(row)


if __name__ == "__main__":
    main()
