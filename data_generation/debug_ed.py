from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import yaml

import andes

from run_sims import (
    _apply_base_operating_point_scale,
    _assign_vis_coefficients,
    _dispatch_ibr_indices,
    _run_ed_dispatch,
    _table_3_1_dispatch_cost_arrays,
)


def _format_row(
    *,
    idx: int,
    bus: int,
    label: str,
    kind: str,
    pg: float,
    pmin: float,
    pmax: float,
    a: float,
    b: float,
    c: float,
    b_r: float,
) -> str:
    reserve = pmax - pg
    return (
        f"{idx:>2}  bus={bus:<2}  {label:<5}  type={kind:<5}  "
        f"Pg={pg:>8.5f}  [{pmin:>7.5f}, {pmax:>7.5f}]  reserve={reserve:>7.5f}  "
        f"cost=(a={a:>7.4f}, b={b:>6.2f}, c={c:>7.2f}, b_r={b_r:>6.2f})"
    )


def _generator_labels(ss) -> tuple[list[int], list[str], list[str]]:
    gen_buses = [int(v) for v in list(ss.PV.bus.v) + list(ss.Slack.bus.v)]
    labels: list[str] = []
    kinds: list[str] = []
    ibr_idx = set(_dispatch_ibr_indices(ss, fallback=[]))
    for idx, _ in enumerate(gen_buses):
        label = f"G{idx + 1}"
        if idx < ss.PV.n:
            try:
                label = str(ss.PV.name.v[idx])
            except Exception:
                pass
        else:
            slack_idx = idx - ss.PV.n
            try:
                label = str(ss.Slack.name.v[slack_idx])
            except Exception:
                pass
        labels.append(label)
        kinds.append("IBR" if idx in ibr_idx else "GEN")
    return gen_buses, labels, kinds


def _print_summary(ss, *, pg: np.ndarray, ed_meta: dict[str, float | str], show_mapping: bool) -> None:
    gen_buses, labels, kinds = _generator_labels(ss)
    a, b, c, b_r = _table_3_1_dispatch_cost_arrays(ss, {"cost_table_path": "configs/table_3_1_dispatch_costs.yaml"})
    pmin = np.asarray(ss.PV.pmin.v + ss.Slack.pmin.v, dtype=float)
    pmax = np.asarray(ss.PV.pmax.v + ss.Slack.pmax.v, dtype=float)

    print("Dispatch summary")
    print(f"  solver={ed_meta['ed_solver']}  status={ed_meta['ed_status']}")
    print(f"  total load Pd={float(np.sum(ss.PQ.p0.v)):.6f} pu")
    print(f"  total dispatch Pg={float(np.sum(pg)):.6f} pu")
    print(
        "  objective parts: "
        f"constant={float(ed_meta['ed_constant_cost']):.6f}, "
        f"energy={float(ed_meta['ed_energy_cost']):.6f}, "
        f"quadratic={float(ed_meta['ed_quadratic_cost']):.6f}, "
        f"reserve_reported={float(ed_meta['ed_reserve_cost']):.6f}, "
        f"total_reported={float(ed_meta['ed_total_cost']):.6f}"
    )
    print("  model objective solved in data generation: sum(c + b*Pg + a*Pg^2)")
    print("  reserve_reported is diagnostic only and is not included in total_reported.")
    print("")
    print("Per-generator dispatch")
    for idx, (bus, label, kind, pg_i, pmin_i, pmax_i, a_i, b_i, c_i, br_i) in enumerate(
        zip(gen_buses, labels, kinds, pg, pmin, pmax, a, b, c, b_r)
    ):
        print(
            _format_row(
                idx=idx,
                bus=bus,
                label=label,
                kind=kind,
                pg=float(pg_i),
                pmin=float(pmin_i),
                pmax=float(pmax_i),
                a=float(a_i),
                b=float(b_i),
                c=float(c_i),
                b_r=float(br_i),
            )
        )

    if not show_mapping:
        return

    print("")
    print("Feature mapping")
    print(f"  dispatch vector order (PV + Slack buses): {gen_buses}")
    mapped = _dispatch_ibr_indices(ss, fallback=[])
    print(f"  REGCV1 -> dispatch positions from REGCV1.gen: {mapped}")
    print(
        "  exported data columns use [P_GENROU_1..n, P_REGCV1_1..n]. "
        "The REGCV1 columns are filled from the mapped dispatch positions."
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run and print one data-generation ED solve.")
    parser.add_argument(
        "--config",
        default="data_generation/generation.yaml",
        help="Path to a data-generation YAML config.",
    )
    parser.add_argument(
        "--base-scale",
        type=float,
        default=0.8,
        help="Base operating-point scale applied before ED.",
    )
    parser.add_argument(
        "--m-value",
        type=float,
        default=4.0,
        help="Uniform REGCV1 inertia value used for this debug run.",
    )
    parser.add_argument(
        "--d-value",
        type=float,
        default=3.0,
        help="Uniform REGCV1 damping value used for this debug run.",
    )
    parser.add_argument(
        "--disable-line-limits",
        action="store_true",
        help="Disable PTDF line limits for the debug ED solve.",
    )
    parser.add_argument(
        "--show-mapping",
        action="store_true",
        help="Print how the dispatch vector maps into exported REGCV1 features.",
    )
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ss = andes.load(cfg["case"], setup=False)
    ss.config.freq = 50.0

    _apply_base_operating_point_scale(
        ss,
        float(args.base_scale),
        scale_pv=bool(cfg.get("load", {}).get("scale_pv", False)),
    )
    _assign_vis_coefficients(
        ss,
        np.full(ss.REGCV1.n, float(args.m_value), dtype=float),
        np.full(ss.REGCV1.n, float(args.d_value), dtype=float),
    )

    ed_cfg = dict(cfg.get("ed", {}) or {})
    if args.disable_line_limits:
        ed_cfg["line_limits_enable"] = False

    pg, ed_meta = _run_ed_dispatch(
        ss,
        ed_cfg,
        ibr_idx=_dispatch_ibr_indices(ss, fallback=ed_cfg.get("ibr_idx") or []),
    )
    _print_summary(ss, pg=pg, ed_meta=ed_meta, show_mapping=bool(args.show_mapping))


if __name__ == "__main__":
    main()
