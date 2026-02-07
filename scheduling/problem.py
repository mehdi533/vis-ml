# Write this file that is based on the other ones but make sure that the pb is written correctly. Specifically for a small trained MLP (2 layers)

"""
Write the MILP end-to-end:
- equality constraints on input features (pre-defined)
- inequality constraints on the output features (pre-defined)
- inequality constraints on the line flows (see convex opt. line flow class + smart grids technology class)
- ineq. / eq. constraints on layer by layer output (with small MLP)
- eq. constraints power dispatch and prod. of REGCV1S


Takes too long to solve, was correct in the other
"""

import cvxpy as cp
import numpy as np
import torch
import andes
import argparse
from pathlib import Path
import yaml
import joblib

from typing import Dict, List, Sequence, Tuple

from andes.interop.pandapower import to_pandapower
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
from pandapower import auxiliary as aux

from models.models import MTLSharedHeads

from scheduling.utils import (
    build_features,
    compute_y_bounds,
    add_measurement_devices,
    _extract_linear_layers,
    activation_masks_mtlshared,
)


def input_features_constraints(x_val_sc: np.ndarray, 
                               x_features: list[str], 
                               m_idx: list[int], 
                               d_idx: list[int],
                               m_min_sc: float,
                               m_max_sc: float,
                               d_min_sc: float,
                               d_max_sc: float):
    
    # TODO: maybe I should redefine the variables because I only need the M and Ds to be changing, not the rest no matter if I add limits
    x = cp.Variable(len(x_val_sc), name="features")
    
    eq_idx = [i for i, feat in enumerate(x_features) if feat not in sorted(set(m_idx) | set(d_idx))] # provides list of indices
    print(eq_idx)
    print(sorted(set(m_idx) | set(d_idx)))
    constraints = [x[eq_idx] == x_val_sc[eq_idx]]
    constraints += [x[m_idx] >= m_min_sc, x[m_idx] <= m_max_sc]
    constraints += [x[d_idx] >= d_min_sc, x[d_idx] <= d_max_sc]

    return x, constraints


def output_features_constraints(y_min_sc: np.ndarray, 
                                y_max_sc: np.ndarray,
                                dp_ibr_sc: np.ndarray): # TODO not sure if p_scaled is np.darray
    
    y = cp.Variable(len(y_min_sc), name="output_features")

    constraints = []

    # Constraints on min and max value for freq. values and /\P_IBR values
    constraints += [y >= y_min_sc]
    constraints += [y <= y_max_sc]

    # Constraints on (/\P_IBR)_sc == y[4:8]
    constraints += [y[4:8] == dp_ibr_sc]

    return y, constraints


def inequality_line_flows(ss: andes.System, 
                          pg: cp.Expression | np.ndarray,
                          pd: cp.Expression | np.ndarray):

    # Convert andes system to pandapower format to compute line flow limits constraints
    pp_net = to_pandapower(ss, verify=False) # TODO what if i put true here? 

    # Compute PTDF matrix
    pp_net._options = {}
    aux._add_ppc_options(
        pp_net,
        calculate_voltage_angles=True,
        trafo_model="pi",
        check_connectivity=False,
        mode="opf",
        switch_rx_ratio=2,
        enforce_q_lims=False,
        recycle=None,
    )
    _, ppci = _pd2ppc(pp_net)
    branch = ppci["branch"]
    fmax = np.asarray(branch[:, 5], dtype=float)  # RATE_A TODO why is this up to 5? 
    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"], using_sparse_solver=False) # TODO what if i put true here? 
    
    bus_ids = pp_net.bus.index.to_numpy()
    line_ids = pp_net.line.index.to_numpy() # TODO: why is this not used? 

    # Compute Cd and Cg matrices to map generator and load injections to bus power injections
    bus_df = ss.Bus.as_df()[["idx"]]
    bus_nums = bus_df["idx"].to_numpy(dtype=int)
    bus_uids = bus_df.index.to_numpy(dtype=int)
    bus_pos = {int(bus): i for i, bus in enumerate(bus_ids)}
    bus_num_to_pos = {int(num): bus_pos[int(uid)] for num, uid in zip(bus_nums, bus_uids)}

    gen_buses = np.concatenate([np.asarray(ss.PV.bus.v, dtype=int), np.asarray(ss.Slack.bus.v, dtype=int)])
    load_buses = np.asarray(ss.PQ.bus.v, dtype=int)

    gen_rows = np.fromiter((bus_num_to_pos[int(bus)] for bus in gen_buses), dtype=int, count=len(gen_buses))
    load_rows = np.fromiter((bus_num_to_pos[int(bus)] for bus in load_buses), dtype=int, count=len(load_buses))

    Cg = np.zeros((len(bus_ids), len(gen_buses)), dtype=float)
    Cd = np.zeros((len(bus_ids), len(load_buses)), dtype=float)
    Cg[gen_rows, np.arange(len(gen_buses))] = 1.0
    Cd[load_rows, np.arange(len(load_buses))] = 1.0

    # Compute net injections (Cg, pg_var, Cd, pd_vec)
    injections = Cg @ pg - Cd @ pd
    
    """
    if shunt: # TODO: double check that
        p = p - shunt
    if bus_inj:
        p = p - bus_inj
    """

    # Compute line flows (ptdf, injections)
    flows = ptdf @ injections

    # Build constraints (flows, fmax)
    constraints = [flows <= fmax, flows >= -fmax]

    return flows, constraints


def masks_equal(masks_a, masks_b): # Can i replace this????
    if masks_a is masks_b:
        return True
    if masks_a is None or masks_b is None:
        return False
    if len(masks_a) != len(masks_b):
        return False
    for part_a, part_b in zip(masks_a, masks_b):
        if len(part_a) != len(part_b):
            return False
        for a, b in zip(part_a, part_b):
            if a.shape != b.shape or not np.array_equal(a, b):
                return False
    return True


def solve_fixed_pattern_iter(
        model,
        x,
        x_val_sc,
        y,
        max_iter=10,
        base_constraints=None,
        objective=None,
        solver="GUROBI",
        ):
    # if x_init is None:
    #     x_init = x_var.value
    #     if x_init is None:
    #         raise ValueError("x_init must be provided or x_var must have a value.")
    masks = activation_masks_mtlshared(model, x_val_sc)
    base_constraints = list(base_constraints) if base_constraints is not None else []
    objective = objective if objective is not None else cp.Minimize(0)

    for it in range(max_iter):
        constraints_nn = surrogate_constraints_fixed_activation(model, x, x_val_sc, y, masks)
        prob = cp.Problem(objective, base_constraints + constraints_nn)
        prob.solve(solver=solver)

        x_sol = x.value
        new_masks = activation_masks_mtlshared(model, np.asarray(x_sol, dtype=float))

        if masks_equal(new_masks, masks):
        # if masks is new_masks:
            print(f"Did {it} iterations to find consistent pattern given the values")
            return constraints_nn  # consistent pattern

        masks = new_masks

    print(f"Couldn't reach stable result after {max_iter} iterations")

    return constraints_nn  # best effort


def surrogate_constraints_fixed_activation(
        model, 
        x,
        x_val_sc,
        # x_min_sc,
        # x_max_sc,
        y, # TODO: add type
        masks=None,
        ):
    constraints = []
    h = x
    
    # Extracting masks based on activation
    if masks is None:
        shared_masks, head_masks = activation_masks_mtlshared(model, x_val_sc) 
    else:
        shared_masks, head_masks = masks

    # === Shared layers ====

    shared_layers = _extract_linear_layers(model.shared)

    for idx, (W, b) in enumerate(shared_layers):
        z = W @ h + b
        out = cp.Variable(len(b), name=f"shared_{idx}")
        mask = shared_masks[idx]
        if np.all(mask):
            constraints += [z >= 0, out == z]
        elif np.all(~mask):
            constraints += [z <= 0, out == 0]
        else:
            constraints += [
                z[mask] >= 0,
                out[mask] == z[mask],
                z[~mask] <= 0,
                out[~mask] == 0,
            ]
        h = out

    # === Heads (label specific) ====

    head_relu_idx = 0
    for i, head in enumerate(model.heads):
        head_layers = _extract_linear_layers(head)
        hh = h
        for idx, (W, b) in enumerate(head_layers):
            z = W @ hh + b

            if idx == len(head_layers) - 1: # last layer of the head, we want to constrain it to be equal to the output variable y
                constraints.append(y[i] == z[0])
                continue

            out = cp.Variable(len(b), name=f"head{i}_{idx}")
            
            mask = head_masks[head_relu_idx]
            head_relu_idx += 1
            if np.all(mask):
                constraints += [z >= 0, out == z]
            elif np.all(~mask):
                constraints += [z <= 0, out == 0]
            else:
                constraints += [
                    z[mask] >= 0,
                    out[mask] == z[mask],
                    z[~mask] <= 0,
                    out[~mask] == 0,
                ]
            hh = out

    return constraints


def surrogate_constraints_milp(model, 
                               x,
                               x_min_sc,
                               x_max_sc,
                               y, # TODO: add type
                               ):
    
    constraints = []

    # defining the input to the first layer as the variable x (features)
    h, h_min, h_max = x, x_min_sc.copy(), x_max_sc.copy() 

    # === Shared layers ====

    shared_layers = _extract_linear_layers(model.shared) # Shared layers are the same for all outputs

    for idx, (W, b) in enumerate(shared_layers):
        z = W @ h + b
        z_min = W.clip(min=0) @ h_min + W.clip(max=0) @ h_max + b
        z_max = W.clip(min=0) @ h_max + W.clip(max=0) @ h_min + b
        out = cp.Variable(len(b), name=f"shared_{idx}")

        if np.any(z_min >= 0): # always active
            idx = np.flatnonzero(z_min >= 0)
            constraints += [out[idx] == z[idx], z[idx] >= 0]
        if np.any(z_max <= 0): # always inactive
            idx = np.flatnonzero(z_max <= 0)
            constraints += [out[idx] == 0, z[idx] <= 0]
        if np.any( ~((z_min >= 0) | (z_max <= 0)) ): # uncertain if z_min < 0 < z_max
            idx = np.flatnonzero( ~((z_min >= 0) | (z_max <= 0)) )
            a = cp.Variable(len(idx), boolean=True, name=f"mlp_{idx}_bin")
            constraints += [
                            out[idx] >= 0,
                            out[idx] >= z[idx],
                            out[idx] <= z[idx] - cp.multiply(z_min[idx], 1 - a), # loose upper bound since z_min is negative
                            out[idx] <= cp.multiply(z_max[idx], a)
                    ]
            
        h, h_min, h_max = out, np.maximum(0, z_min), np.maximum(0, z_max)
    

    # === Heads (label specific) ====

    for i, head in enumerate(model.heads): # Heads are unique for all outputs so we need to treat them separately
        head_layers = _extract_linear_layers(head)
        hh, hh_min, hh_max = h, h_min.copy(), h_max.copy() # input to the head is the output of the shared layers
        for idx, (W, b) in enumerate(head_layers):
            z = W @ hh + b
            
            if idx == len(head_layers) - 1: # last layer of the head, we want to constrain it to be equal to the output variable y
                constraints.append(y[i] == z)
                continue 
            
            # TODO: rewrite the rest as an inner func. since it's used multiple times and it's the same as for the shared layers
            z_min = W.clip(min=0) @ hh_min + W.clip(max=0) @ hh_max + b
            z_max = W.clip(min=0) @ hh_max + W.clip(max=0) @ hh_min + b
            out = cp.Variable(len(b), name=f"head{i}_{idx}")
            
            if np.any(z_min >= 0): # always active
                idx = np.flatnonzero(z_min >= 0)
                constraints += [out[idx] == z[idx], z[idx] >= 0]
            if np.any(z_max <= 0): # always inactive
                idx = np.flatnonzero(z_max <= 0)
                constraints += [out[idx] == 0, z[idx] <= 0]
            if np.any( ~((z_min >= 0) | (z_max <= 0)) ): # uncertain if z_min < 0 < z_max
                idx = np.flatnonzero( ~((z_min >= 0) | (z_max <= 0)) )
                a = cp.Variable(len(idx), boolean=True, name=f"mlp_{idx}_bin")
                constraints += [
                                out[idx] >= 0,
                                out[idx] >= z[idx],
                                out[idx] <= z[idx] - cp.multiply(z_min[idx], 1 - a), 
                                out[idx] <= cp.multiply(z_max[idx], a)
                        ]
            
            hh, hh_min, hh_max = out, np.maximum(0, z_min), np.maximum(0, z_max)
    
    return constraints


def define_rted_vis(ss: andes.System, model,
                    x_val_sc: np.ndarray,
                    m_idx: list[int], d_idx: list[int],
                    m_min_sc: float | np.ndarray, m_max_sc: float | np.ndarray,
                    d_min_sc: float | np.ndarray, d_max_sc: float | np.ndarray,
                    x_min_sc: np.ndarray, x_max_sc: np.ndarray,
                    y_min_sc: np.ndarray, y_max_sc: np.ndarray,
                    p_scaled: np.ndarray,
                    a: np.ndarray, b: np.ndarray, c: np.ndarray,
                    pg: cp.Variable, pg_min: np.ndarray, pg_max: np.ndarray,
                    pd: np.ndarray):
    """
    function to define the problem end to end, it should 1. test constraints one by one to see if feasible. 2. provide the entire problem with constraints... 
    Define obj. func. (minimize cost, load from yaml file) and constraints (input features, output features, line flows, surrogate model) and solve the problem with a MILP solver (e.g., Gurobi).
    """

    if pd is None:
        pd = np.asarray(ss.PQ.p0.v, dtype=float) if ss.PQ.n > 0 else np.zeros(0, dtype=float)

    if pg_min is None or pg_max is None:
        pg_min = np.asarray([ss.PV.pmin.v[i] for i in range(ss.PV.n)] + [ss.Slack.pmin.v[i] for i in range(ss.Slack.n)], dtype=float)
        pg_max = np.asarray([ss.PV.pmax.v[i] for i in range(ss.PV.n)] + [ss.Slack.pmax.v[i] for i in range(ss.Slack.n)], dtype=float)

    constraints = []

    x, constraints_x = input_features_constraints(x_val_sc, np.arange(len(x_val_sc)), m_idx, d_idx, m_min_sc, m_max_sc, d_min_sc, d_max_sc)
    y, constraints_y = output_features_constraints(y_min_sc, y_max_sc, p_scaled)
    # constraints_nn = surrogate_constraints_milp(model, x, x_min_sc, x_max_sc, y)
    flows, constraints_line = inequality_line_flows(ss, pg, pd)
    constraints_ed = [pg >= pg_min, pg <= pg_max, cp.sum(pg) == float(np.sum(pd))]

    objective = cp.Minimize(cp.sum(a + cp.multiply(b, pg) + cp.multiply(c, cp.square(pg))))
    base_constraints = constraints_x + constraints_y + constraints_line + constraints_ed
    constraints_nn = solve_fixed_pattern_iter(
        model,
        x,
        x_val_sc,
        y,
        max_iter=100,
        base_constraints=base_constraints,
        objective=objective,
    )

    constraints += constraints_x # if isinstance(constraints_x, list) else [constraints_x]
    constraints += constraints_y
    constraints += constraints_nn
    constraints += constraints_line
    constraints += constraints_ed # Constraints on generator outputs and ED cosntraint

    feasibility = {}

    checks = {
        "input": constraints_x + constraints_ed, # 69 (eq. and ineq.)
        "output": constraints_y + constraints_ed, # 20 (ineq.)
        "nn": constraints_nn + constraints_ed, # 6560 (256, 128)  (128, 64) * 8 ==> if all big M constraints would be = 4 * (256 + 128) + 4 * (128 + 64) + 8 = 7688 ==> pruning is effective for (7680 - 6560) / 2 = 560 neurons
        "line_flows": constraints_line + constraints_ed, # 92 (ineq. ==> 2 per line)
        "combined": constraints, # 6762 constraints (21 constraints for sum(pg) = sum(pd) (1), pg_max and pg_min ineq. (10 each)
    }

    solver_kwargs = {"solver": "GUROBI", "verbose": True, "reoptimize": False}

    for name, cons in checks.items():
        cons_list = cons if isinstance(cons, list) else [cons]
        prob_check = cp.Problem(objective, cons_list)
        prob_check.solve(**(solver_kwargs))
        feasibility[name] = prob_check.status

    prob = cp.Problem(objective, constraints)

    prob.solve(**(solver_kwargs))

    return prob, {"x": x, "y": y, "flows": flows, "pg": pg, "constraints": constraints, "feasibility": feasibility}


def main():
    """
    define and run rted vis
    """

    parser = argparse.ArgumentParser(description="Scan step_scale to find ED differences with NN convex constraints.")
    parser.add_argument("--config", default="experiments/generation.yaml", help="Path to sim YAML.")
    parser.add_argument("--cost-config", default="scheduling/mtlsh_convex.yaml", help="Path to cost YAML.")
    parser.add_argument("--base-scale", type=float, default=1.0, help="Base load scale.")
    parser.add_argument("--step-scale", type=float, default=0.95, help="Load step scale.")
    # parser.add_argument("--plot-dir", type=str, default="experiments", help="Directory to save plots.")
    args = parser.parse_args()

    # ==== load configs =======

    def load_yaml(path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    cfg = load_yaml(Path(args.config))
    cost_cfg = load_yaml(Path(args.cost_config))

    # ==== load model =======

    model_cfg = cost_cfg.get("model", {})
    state_path = Path(model_cfg.get("state_dict", "")) / "vis_mlp_state_dict.pt"
    
    if not state_path.exists():
        raise FileNotFoundError(f"Model state_dict not found at {state_path}")

    model = MTLSharedHeads(
                in_dim=int(model_cfg["in_dim"]),
                n_tasks=int(model_cfg["n_tasks"]),
                shared_sizes=model_cfg.get("shared_sizes"),
                head_sizes=model_cfg.get("head_sizes"),
                dropout=float(model_cfg.get("dropout", 0.0)),
            )
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    # ==== load scalers =======

    scalers_cfg = cost_cfg.get("scalers", {})
    x_scaler = joblib.load(scalers_cfg.get("x_scaler_path"))
    y_scaler = joblib.load(scalers_cfg.get("y_scaler_path"))

    # ==== andes system =======

    ss = andes.load(cfg["case"], setup=False)
    ss.config.freq = float(50)
    add_measurement_devices(ss)

    m_vec = np.full(ss.REGCV1.n, 4.0)
    d_vec = np.full(ss.REGCV1.n, 1.0)

    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * args.base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * args.base_scale
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * args.base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * args.base_scale

    ss.REGCV1.M.v, ss.REGCV1.D.v = m_vec, d_vec

    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.config.p2i = 0
    ss.PQ.config.q2i = 0
    ss.PQ.config.pq2z = 0

    ss.setup()
    
    # ==== build constraints from cfg file ======

    pg_min = np.asarray(ss.PV.pmin.v.tolist() + ss.Slack.pmin.v.tolist(), dtype=float)
    pg_max = np.asarray(ss.PV.pmax.v.tolist() + ss.Slack.pmax.v.tolist(), dtype=float)

    x_features = cost_cfg["features"]["x_features"]

    a = np.asarray(cost_cfg["ed_costs"]["a"], dtype=float)
    b = np.asarray(cost_cfg["ed_costs"]["b"], dtype=float)
    c = np.asarray(cost_cfg["ed_costs"]["c"], dtype=float)

    bounds_cfg = cost_cfg.get("bounds", {})
    m_min, m_max = bounds_cfg["M_bounds"]
    d_min, d_max = bounds_cfg["D_bounds"]
    m_names = [f"M_{i+1}" for i in range(ss.REGCV1.n)]
    d_names = [f"D_{i+1}" for i in range(ss.REGCV1.n)]
    name_to_idx = {name: i for i, name in enumerate(x_features)}
    m_idx = [name_to_idx[n] for n in m_names]
    d_idx = [name_to_idx[n] for n in d_names]

    def scale_minmax(values, scaler, idx):
        idx = np.asarray(idx, dtype=int)
        return values * scaler.scale_[idx] + scaler.min_[idx]

    def unscale_minmax(values, scaler, idx):
        idx = np.asarray(idx, dtype=int)
        return (values - scaler.min_[idx]) / scaler.scale_[idx]

    m_min_sc = scale_minmax(m_min, x_scaler, m_idx)
    m_max_sc = scale_minmax(m_max, x_scaler, m_idx)
    d_min_sc = scale_minmax(d_min, x_scaler, d_idx)
    d_max_sc = scale_minmax(d_max, x_scaler, d_idx)

    y_min_raw = np.asarray(bounds_cfg.get("y_min", []), dtype=float).reshape(-1)
    y_max_raw = np.asarray(bounds_cfg.get("y_max", []), dtype=float).reshape(-1)
    y_min_sc = scale_minmax(y_min_raw, y_scaler, np.arange(len(y_min_raw)))
    y_max_sc = scale_minmax(y_max_raw, y_scaler, np.arange(len(y_min_raw)))

    pg = cp.Variable(ss.PV.n + ss.Slack.n, name="pg")  # generator outputs
    pg_base = np.array(ss.PV.p0.v.tolist() + ss.Slack.p0.v.tolist())
    
    ibr_idx = np.asarray(cost_cfg.get("ibr_idx", []), dtype=int)
    dp_ibr = pg[ibr_idx] - pg_base[ibr_idx]
    ibr_out_idx = np.arange(4,8)
    dp_ibr_sc = cp.multiply(dp_ibr, y_scaler.scale_[ibr_out_idx]) + y_scaler.min_[ibr_out_idx] 

    pd = np.asarray(ss.PQ.p0.v, dtype=float) * args.step_scale

    # ==== extract features ======

    x_val = build_features(
        ss,
        base_scale=args.base_scale,
        step_scale=args.step_scale,
        load_step_time=float(cfg["tds"]["load_step_time"]),
        M_vec=m_vec,
        D_vec=d_vec,
    )
    
    feat_vec = np.array([x_val[name] for name in x_features], dtype=float).reshape(-1)

    # ==== define features bounds (big M for the MILP) ======
    x_val_sc = scale_minmax(feat_vec, x_scaler, np.arange(len(feat_vec)))

    x_min_sc, x_max_sc = x_val_sc.copy(), x_val_sc.copy()
    x_min_sc[m_idx], x_max_sc[m_idx] = m_min_sc, m_max_sc
    x_min_sc[d_idx], x_max_sc[d_idx] = d_min_sc, d_max_sc

    # === continue with pb definition and solving ====
    prob, values = define_rted_vis(
        ss, model, x_val_sc,
        m_idx, d_idx, m_min_sc, m_max_sc, d_min_sc, d_max_sc,
        x_min_sc, x_max_sc, y_min_sc, y_max_sc, dp_ibr_sc,
        a, b, c, pg, pg_min, pg_max, pd
    )

    for key, val in values.items():
        if hasattr(val, "value"):
            print(f"{key}:\n{val.value}\n")
        elif isinstance(val, list):
            print(f"{key}: {len(val)}")
        else:
            print(f"{key}: {val}")

    print("m: ", unscale_minmax(values["x"][m_idx].value, x_scaler, m_idx))
    print("d: ", unscale_minmax(values["x"][d_idx].value, x_scaler, d_idx))


    return prob, values


if __name__ == "__main__":
    main()
