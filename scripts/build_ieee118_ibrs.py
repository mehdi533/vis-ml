#!/usr/bin/env python3
"""Build a persistent IEEE 118-bus case with dynamics + REGCV1 grid-forming IBRs.

Loads the ANDES-bundled power-flow-only IEEE 118 case, attaches GENROU+TGOV1N to
the synchronous generators, replaces a spread of generators with REGCV1
grid-forming converters, and saves the assembled case to
data_generation/andes_cases/ieee118_ibrs.xlsx (mirroring ieee39_full_ibrs.xlsx).

Prints a JSON summary and the chosen IBR generator idxs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import andes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.systems.dynamify import dynamify_case  # noqa: E402
from research.systems.registry import augment_with_grid_forming_ibrs  # noqa: E402

N_IBR = 4
NOMINAL_M = 6.0
NOMINAL_D = 3.0
# Lower inertia than the round-rotor default (H=4) so the 118 grid actually
# swings -- modelling the low-inertia condition the thesis targets.
DEFAULT_H = 2.5
OUT = ROOT / "data_generation" / "andes_cases" / "ieee118_ibrs.xlsx"


def synthesize_line_limits(ss, factor: float = 1.4, floor_mva: float = 40.0):
    """Set Line.rate_a from a DC base-case flow (matpower case118 ships with none).

    rate_a = max(floor, factor * |DC flow|), so lines sit at ~1/factor loading at
    base and N-1 contingencies can bind meaningfully. Returns the ratings (MVA).
    """
    import numpy as np

    buses = list(ss.Bus.idx.v)
    bidx = {b: i for i, b in enumerate(buses)}
    nb = len(buses)
    x = np.asarray(ss.Line.x.v, dtype=float)
    f = [bidx[b] for b in ss.Line.bus1.v]
    t = [bidx[b] for b in ss.Line.bus2.v]
    inj = np.zeros(nb)
    for mdl in ("PV", "Slack"):
        m = getattr(ss, mdl, None)
        if m and m.n:
            for k in range(m.n):
                inj[bidx[m.bus.v[k]]] += float(m.p0.v[k])
    for k in range(ss.PQ.n):
        inj[bidx[ss.PQ.bus.v[k]]] -= float(ss.PQ.p0.v[k])
    slack = bidx[ss.Slack.bus.v[0]]
    B = np.zeros((nb, nb))
    for i in range(len(x)):
        if x[i] <= 0 or not np.isfinite(x[i]):
            continue
        b = 1.0 / x[i]
        B[f[i], f[i]] += b; B[t[i], t[i]] += b
        B[f[i], t[i]] -= b; B[t[i], f[i]] -= b
    keep = [i for i in range(nb) if i != slack]
    theta = np.zeros(nb)
    theta[keep] = np.linalg.solve(B[np.ix_(keep, keep)], inj[keep])
    flow = np.array([(theta[f[i]] - theta[t[i]]) / x[i] if x[i] > 0 else 0.0 for i in range(len(x))])
    rate = np.maximum(floor_mva, factor * np.abs(flow) * 100.0)
    ss.Line.rate_a.v = rate
    return rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-H", type=float, default=DEFAULT_H)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--line-limits", action="store_true", help="Synthesize DC-flow line ratings.")
    args = ap.parse_args()
    out_path = Path(args.out)

    warnings.filterwarnings("ignore")
    andes.config_logger(stream_level=40)
    case = os.path.join(os.path.dirname(andes.__file__), "cases", "matpower", "case118.m")

    ss = andes.load(case, setup=False, no_output=True)
    pv_idxs = list(ss.PV.idx.v)
    # Spread the IBRs across the generator set.
    step = max(1, len(pv_idxs) // N_IBR)
    ibr_gens = pv_idxs[::step][:N_IBR]

    # Add a Center-of-Inertia device so the COI frequency (rocof_COI, dev_COI) is
    # produced by TDS; the synchronous machines reference it.
    ss.add("COI", param_dict={"idx": "COI_1", "name": "COI_1"})
    dynamify_case(ss, target_H=args.target_H, exclude_gen_idxs=ibr_gens, coi_idx="COI_1")
    augment_with_grid_forming_ibrs(
        ss, gen_idxs=ibr_gens,
        m_values=[NOMINAL_M] * len(ibr_gens),
        d_values=[NOMINAL_D] * len(ibr_gens),
    )
    ss.setup()
    ss.config.freq = 50.0  # European 50 Hz base (ANDES defaults to 60)
    pf = bool(ss.PFlow.run())

    if args.line_limits:
        rates = synthesize_line_limits(ss)
        n_rated = int((rates > 40.0).sum())
    else:
        n_rated = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    andes.io.xlsx.write(ss, str(out_path), overwrite=True)

    # Reload the saved artifact fresh and confirm it is TDS-ready.
    ss2 = andes.load(str(out_path), setup=True, no_output=True)
    ss2.config.freq = 50.0  # European 50 Hz base
    pf2 = bool(ss2.PFlow.run())
    ss2.TDS.config.tf = 1.0
    ss2.TDS.config.no_tqdm = 1
    ss2.TDS.run()

    summary = {
        "output_case": str(out_path.relative_to(ROOT)),
        "target_H": args.target_H,
        "n_lines_rated": n_rated,
        "n_buses": int(ss2.Bus.n),
        "n_lines": int(ss2.Line.n),
        "n_genrou": int(ss2.GENROU.n),
        "n_regcv1": int(ss2.REGCV1.n),
        "ibr_gen_idxs": [int(x) if str(x).isdigit() else x for x in ibr_gens],
        "pflow_built": pf,
        "pflow_reloaded": pf2,
        "tds_reloaded_exit_code": int(ss2.exit_code),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
