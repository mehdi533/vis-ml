# line_utils.py
# Line-level metrics and DC proxy helpers used by feature extraction.

import io
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_WARNED_MESSAGES: set[str] = set()


@contextmanager
def _suppress_pandapower_interop_noise():
    """Helper to suppress pandapower interop noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"andes\.interop\.pandapower",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"pandas\..*",
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield


def _to_float_or_nan(value) -> float:
    """Helper to float or nan."""
    try:
        if value is None:
            return float("nan")
        out = float(value)
        return out if np.isfinite(out) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_neg_one(value) -> float:
    """Helper to finite or neg one."""
    out = _to_float_or_nan(value)
    return out if np.isfinite(out) else -1.0


def _warn_once(msg: str) -> None:
    """Helper to warn once."""
    if msg in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(msg)
    print(f"[line_metrics] {msg}")


def _valid_rating(value) -> float:
    """Helper to valid rating."""
    val = _to_float_or_nan(value)
    return val if np.isfinite(val) and val > 0 else np.nan


def _rating_to_pu(rating_raw: float, base_mva: float) -> float:
    """Helper to rating to pu."""
    r = _valid_rating(rating_raw)
    if not np.isfinite(r):
        return np.nan
    if np.isfinite(base_mva) and base_mva > 0:
        return float(r / base_mva)
    return r


def line_extra_fieldnames(line_uids: Sequence[int]) -> List[str]:
    """Build line-related extra fieldnames."""
    fields = [
        "line_fn",
        "line_Vn1",
        "line_Vn2",
        "line_r",
        "line_x",
        "line_b",
        "line_g",
        "line_b1",
        "line_g1",
        "line_b2",
        "line_g2",
        "line_trans",
        "line_tap",
        "line_phi",
        "line_x_over_r",
        "pre_p_from",
        "pre_p_to",
        "pre_loading_from",
        "pre_loading_to",
        "pre_flow_direction_p",
        "pre_v_from",
        "pre_v_to",
        "pre_theta_from",
        "pre_theta_to",
        "pre_delta_theta",
        "bus_degree_from",
        "bus_degree_to",
        "is_bridge",
        "n_components_after_trip",
        "largest_component_fraction_after_trip",
        "total_load_p_prefault",
        "total_gen_p_prefault",
        "reserve_proxy_prefault",
        "system_max_loading_prefault",
        "system_mean_loading_prefault",
        "system_top5_loading_mean_prefault",
        "ptdf_l1_norm_outaged_line",
        "max_abs_lodf_row",
        "predicted_max_post_cont_loading_dc",
    ]
    fields.extend([f"line_oh_uid_{int(uid)}" for uid in line_uids])
    return fields


def _line_rating_from_ss(ss, uid: int) -> float:
    """Helper to line rating from ss."""
    for attr in ("rate_a", "rateA", "RATE_A"):
        obj = getattr(ss.Line, attr, None)
        if obj is None:
            continue
        vals = getattr(obj, "v", None)
        if vals is None or uid >= len(vals):
            continue
        out = _valid_rating(vals[uid])
        if np.isfinite(out):
            return out
    return float("nan")


def _line_ratings_from_pandapower(ss) -> Optional[np.ndarray]:
    """Helper to line ratings from pandapower."""
    try:
        from andes.interop import pandapower as ap
        from pandapower.pd2ppc import _pd2ppc
        from pandapower import auxiliary as aux
    except Exception:
        return None
    try:
        with _suppress_pandapower_interop_noise():
            pp_net = ap.to_pandapower(ss, verify=False)
            if not hasattr(pp_net, "_options") or not isinstance(pp_net._options, dict):
                pp_net._options = {}
            if "mode" not in pp_net._options:
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
        return np.asarray(np.real(ppci["branch"][:, 5]), dtype=float)
    except Exception:
        return None


def _line_records(ss) -> List[Dict[str, float | int | str]]:
    """Helper to line records."""
    records: List[Dict[str, float | int | str]] = []
    n_line = int(getattr(ss.Line, "n", 0))
    vals = {
        "idx": list(getattr(getattr(ss.Line, "idx", None), "v", [])),
        "name": list(getattr(getattr(ss.Line, "name", None), "v", [])),
        "bus1": list(getattr(getattr(ss.Line, "bus1", None), "v", [])),
        "bus2": list(getattr(getattr(ss.Line, "bus2", None), "v", [])),
        "u": list(getattr(getattr(ss.Line, "u", None), "v", [])),
        "Sn": list(getattr(getattr(ss.Line, "Sn", None), "v", [])),
        "fn": list(getattr(getattr(ss.Line, "fn", None), "v", [])),
        "Vn1": list(getattr(getattr(ss.Line, "Vn1", None), "v", [])),
        "Vn2": list(getattr(getattr(ss.Line, "Vn2", None), "v", [])),
        "r": list(getattr(getattr(ss.Line, "r", None), "v", [])),
        "x": list(getattr(getattr(ss.Line, "x", None), "v", [])),
        "b": list(getattr(getattr(ss.Line, "b", None), "v", [])),
        "g": list(getattr(getattr(ss.Line, "g", None), "v", [])),
        "b1": list(getattr(getattr(ss.Line, "b1", None), "v", [])),
        "g1": list(getattr(getattr(ss.Line, "g1", None), "v", [])),
        "b2": list(getattr(getattr(ss.Line, "b2", None), "v", [])),
        "g2": list(getattr(getattr(ss.Line, "g2", None), "v", [])),
        "trans": list(getattr(getattr(ss.Line, "trans", None), "v", [])),
        "tap": list(getattr(getattr(ss.Line, "tap", None), "v", [])),
        "phi": list(getattr(getattr(ss.Line, "phi", None), "v", [])),
    }
    for uid in range(n_line):
        idx_val = vals["idx"][uid] if uid < len(vals["idx"]) else uid + 1
        name_val = vals["name"][uid] if uid < len(vals["name"]) else f"Line_{idx_val}"
        in_service = bool(vals["u"][uid]) if uid < len(vals["u"]) else True
        rec: Dict[str, float | int | str] = {
            "uid": uid,
            "idx": idx_val,
            "name": str(name_val),
            "bus1": _to_float_or_nan(vals["bus1"][uid]) if uid < len(vals["bus1"]) else np.nan,
            "bus2": _to_float_or_nan(vals["bus2"][uid]) if uid < len(vals["bus2"]) else np.nan,
            "rating": _line_rating_from_ss(ss, uid),
            "in_service": in_service,
        }
        for k in ("u", "Sn", "fn", "Vn1", "Vn2", "r", "x", "b", "g", "b1", "g1", "b2", "g2", "trans", "tap", "phi"):
            rec[k] = _to_float_or_nan(vals[k][uid]) if uid < len(vals[k]) else np.nan
        records.append(rec)
    return records


def _line_flow_component(ss, uid: int, candidates: Sequence[object]) -> float:
    """Helper to line flow component."""
    for cand in candidates:
        if isinstance(cand, tuple) and len(cand) == 2:
            attr, preferred_sub = str(cand[0]), str(cand[1])
        else:
            attr, preferred_sub = str(cand), None
        obj = getattr(ss.Line, attr, None)
        if obj is None:
            continue
        vals = None
        if preferred_sub is not None:
            vals = getattr(obj, preferred_sub, None)
        if vals is None:
            vals = getattr(obj, "v", None)
        if vals is None:
            vals = getattr(obj, "e", None)
        if vals is None or uid >= len(vals):
            continue
        return _to_float_or_nan(vals[uid])
    return float("nan")


def _line_prefault_flows(ss, uid: int) -> Dict[str, float]:
    """Helper to line prefault flows."""
    p_from = _line_flow_component(ss, uid, (("Pij", "v"), ("p1", "v"), ("P1", "v"), ("pf", "v"), ("a1", "e")))
    p_to = _line_flow_component(ss, uid, (("Pji", "v"), ("p2", "v"), ("P2", "v"), ("pt", "v"), ("a2", "e")))
    return {
        "pre_p_from": _to_float_or_nan(p_from),
        "pre_p_to": _to_float_or_nan(p_to),
    }


def _bus_state(ss, bus_number: float) -> Tuple[float, float]:
    """Helper to bus state."""
    if not np.isfinite(bus_number):
        return np.nan, np.nan
    bus_idx = list(getattr(getattr(ss.Bus, "idx", None), "v", []))
    if not bus_idx:
        return np.nan, np.nan
    uid = {int(v): i for i, v in enumerate(bus_idx)}.get(int(bus_number))
    if uid is None:
        return np.nan, np.nan
    v_vals = getattr(getattr(ss.Bus, "v", None), "v", None)
    a_vals = getattr(getattr(ss.Bus, "a", None), "v", None)
    v = _to_float_or_nan(v_vals[uid]) if v_vals is not None and uid < len(v_vals) else np.nan
    a = _to_float_or_nan(a_vals[uid]) if a_vals is not None and uid < len(a_vals) else np.nan
    return v, a


def _line_parameters(record: Dict[str, float | int | str]) -> Dict[str, float]:
    """Helper to line parameters."""
    r = _to_float_or_nan(record.get("r"))
    x = _to_float_or_nan(record.get("x"))
    x_over_r = x / r if np.isfinite(r) and np.isfinite(x) and abs(r) > 0 else np.nan
    return {
        "line_fn": _to_float_or_nan(record.get("fn")),
        "line_Vn1": _to_float_or_nan(record.get("Vn1")),
        "line_Vn2": _to_float_or_nan(record.get("Vn2")),
        "line_r": r,
        "line_x": x,
        "line_b": _to_float_or_nan(record.get("b")),
        "line_g": _to_float_or_nan(record.get("g")),
        "line_b1": _to_float_or_nan(record.get("b1")),
        "line_g1": _to_float_or_nan(record.get("g1")),
        "line_b2": _to_float_or_nan(record.get("b2")),
        "line_g2": _to_float_or_nan(record.get("g2")),
        "line_trans": _to_float_or_nan(record.get("trans")),
        "line_tap": _to_float_or_nan(record.get("tap")),
        "line_phi": _to_float_or_nan(record.get("phi")),
        "line_x_over_r": _finite_or_neg_one(x_over_r),
    }


def _identity_one_hot(line_uids: Sequence[int], cont_uid: Optional[int]) -> Dict[str, int]:
    """Helper to identity one hot."""
    return {f"line_oh_uid_{int(uid)}": 1 if (cont_uid is not None and int(uid) == int(cont_uid)) else 0 for uid in line_uids}


def _graph(records: Sequence[Dict[str, float | int | str]], skip_uid: Optional[int] = None) -> Tuple[Dict[int, set], set]:
    """Helper to graph."""
    adj: Dict[int, set] = {}
    nodes = set()
    for rec in records:
        if not bool(rec.get("in_service", True)):
            continue
        b1 = _to_float_or_nan(rec.get("bus1"))
        b2 = _to_float_or_nan(rec.get("bus2"))
        if not (np.isfinite(b1) and np.isfinite(b2)):
            continue
        n1, n2 = int(b1), int(b2)
        nodes.add(n1)
        nodes.add(n2)
        uid = rec.get("uid")
        if skip_uid is not None and uid is not None and int(uid) == int(skip_uid):
            continue
        adj.setdefault(n1, set()).add(n2)
        adj.setdefault(n2, set()).add(n1)
    return adj, nodes


def _components(adj: Dict[int, set], nodes: set) -> List[set]:
    """Helper to components."""
    seen = set()
    comps: List[set] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        comp = set()
        seen.add(n)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _topology_criticality(records: Sequence[Dict[str, float | int | str]], cont_uid: Optional[int], bus1: float, bus2: float) -> Dict[str, float]:
    """Helper to topology criticality."""
    adj_base, nodes_base = _graph(records, skip_uid=None)
    deg_from = int(len(adj_base.get(int(bus1), set()))) if np.isfinite(bus1) else np.nan
    deg_to = int(len(adj_base.get(int(bus2), set()))) if np.isfinite(bus2) else np.nan
    if cont_uid is None:
        return {
            "bus_degree_from": deg_from,
            "bus_degree_to": deg_to,
            "is_bridge": np.nan,
            "n_components_after_trip": np.nan,
            "largest_component_fraction_after_trip": np.nan,
        }
    n_base = len(_components(adj_base, nodes_base))
    adj_after, nodes_after = _graph(records, skip_uid=cont_uid)
    comps_after = _components(adj_after, nodes_after)
    n_after = len(comps_after)
    largest = max((len(c) for c in comps_after), default=0)
    frac = (largest / len(nodes_after)) if nodes_after else np.nan
    return {
        "bus_degree_from": deg_from,
        "bus_degree_to": deg_to,
        "is_bridge": bool(n_after > n_base),
        "n_components_after_trip": int(n_after),
        "largest_component_fraction_after_trip": float(frac) if np.isfinite(frac) else np.nan,
    }


def _system_loading_stats(ss, ratings: Optional[np.ndarray], base_mva: float) -> Tuple[float, float, float]:
    """Helper to system loading stats."""
    n_line = int(getattr(ss.Line, "n", 0))
    records = _line_records(ss)
    rec_by_uid = {int(r["uid"]): r for r in records if r.get("uid") is not None}
    vals = []
    for uid in range(n_line):
        p = _line_flow_component(ss, uid, (("Pij", "v"), ("p1", "v"), ("P1", "v"), ("pf", "v"), ("a1", "e")))
        rec = rec_by_uid.get(uid, {})
        r_raw = _line_rating_from_ss(ss, uid)
        if not np.isfinite(r_raw):
            r_raw = _valid_rating(rec.get("Sn"))
        if not np.isfinite(r_raw) and ratings is not None and uid < len(ratings):
            r_raw = _valid_rating(ratings[uid])
        r_pu = _rating_to_pu(r_raw, base_mva)
        if np.isfinite(p) and np.isfinite(r_pu) and r_pu > 0:
            vals.append(abs(p) / r_pu)
    if not vals:
        return np.nan, np.nan, np.nan
    arr = np.asarray(vals, dtype=float)
    top = np.sort(arr)[::-1][: min(5, arr.size)]
    return float(np.max(arr)), float(np.mean(arr)), float(np.mean(top))


def _total_generation(ss) -> float:
    """Helper to total generation."""
    total = 0.0
    found = False
    for model_name in ("PV", "Slack"):
        mdl = getattr(ss, model_name, None)
        if mdl is None or getattr(mdl, "n", 0) <= 0:
            continue
        for attr in ("Pg", "p"):
            vals = getattr(getattr(mdl, attr, None), "v", None)
            if vals is not None:
                total += float(np.nansum(np.asarray(vals, dtype=float)))
                found = True
                break
    return total if found else np.nan


def _reserve_proxy(ss) -> float:
    """Helper to reserve proxy."""
    total = 0.0
    found = False
    for model_name in ("PV", "Slack"):
        mdl = getattr(ss, model_name, None)
        if mdl is None or getattr(mdl, "n", 0) <= 0:
            continue
        pmax = getattr(getattr(mdl, "pmax", None), "v", None)
        if pmax is None:
            continue
        pgen = None
        for attr in ("Pg", "p"):
            vals = getattr(getattr(mdl, attr, None), "v", None)
            if vals is not None:
                pgen = np.asarray(vals, dtype=float)
                break
        if pgen is None:
            continue
        found = True
        total += float(np.nansum(np.asarray(pmax, dtype=float) - pgen))
    return total if found else np.nan


def _fallback_max_loading(ss, base_mva: float) -> float:
    """Lightweight backup: max |P| / rating using available Line data."""
    vals = []
    n_line = int(getattr(ss.Line, "n", 0))
    records = _line_records(ss)
    for uid in range(n_line):
        p = _line_flow_component(ss, uid, (("Pij", "v"), ("p1", "v"), ("P1", "v"), ("pf", "v"), ("a1", "e")))
        rating_raw = _line_rating_from_ss(ss, uid)
        if not np.isfinite(rating_raw) and uid < len(records):
            rating_raw = _valid_rating(records[uid].get("Sn"))
        rating_pu = _rating_to_pu(rating_raw, base_mva)
        if np.isfinite(p) and np.isfinite(rating_pu) and rating_pu > 0:
            vals.append(abs(p) / rating_pu)
    return float(np.nanmax(vals)) if vals else np.nan


def _global_stress(ss, ratings: Optional[np.ndarray], base_mva: float) -> Dict[str, float]:
    """Helper to global stress."""
    total_load_p = float(np.nansum(np.asarray(ss.PQ.Ppf.v, dtype=float))) if getattr(ss.PQ, "n", 0) > 0 else np.nan
    lmax, lmean, ltop5 = _system_loading_stats(ss, ratings, base_mva)
    return {
        "total_load_p_prefault": _to_float_or_nan(total_load_p),
        "total_gen_p_prefault": _to_float_or_nan(_total_generation(ss)),
        "reserve_proxy_prefault": _to_float_or_nan(_reserve_proxy(ss)),
        "system_max_loading_prefault": _to_float_or_nan(lmax),
        "system_mean_loading_prefault": _to_float_or_nan(lmean),
        "system_top5_loading_mean_prefault": _to_float_or_nan(ltop5),
    }


def _find_outage_branch_row(
    branch: np.ndarray,
    *,
    out_bus1_ppc: float,
    out_bus2_ppc: float,
    cont_uid: Optional[int],
) -> Optional[int]:
    """Helper to find outage branch row."""
    n_branch = int(branch.shape[0])
    if n_branch <= 0:
        return None

    if np.isfinite(out_bus1_ppc) and np.isfinite(out_bus2_ppc):
        b1 = int(out_bus1_ppc)
        b2 = int(out_bus2_ppc)
        fb = branch[:, 0].astype(int)
        tb = branch[:, 1].astype(int)
        cand = np.where(((fb == b1) & (tb == b2)) | ((fb == b2) & (tb == b1)))[0]
        if cand.size == 1:
            return int(cand[0])
        if cand.size > 1:
            # If multiple parallel branches exist, use cont_uid as tie-break when valid.
            if cont_uid is not None and 0 <= int(cont_uid) < n_branch:
                return int(cont_uid)
            return int(cand[0])

    if cont_uid is not None and 0 <= int(cont_uid) < n_branch:
        return int(cont_uid)
    return None


def _map_ss_busnum_to_ppc_pos(ss, pp_net, bus_num: float) -> float:
    """Helper to map ss busnum to ppc pos."""
    if not np.isfinite(bus_num):
        return np.nan
    try:
        bus_ids = pp_net.bus.index.to_numpy(dtype=int)
        bus_df = ss.Bus.as_df()[["idx"]]
        ss_bus_nums = bus_df["idx"].to_numpy(dtype=int)
        ss_bus_uids = bus_df.index.to_numpy(dtype=int)
        bus_pos = {int(bus): i for i, bus in enumerate(bus_ids)}
        bus_num_to_pos = {
            int(num): float(bus_pos[int(uid)])
            for num, uid in zip(ss_bus_nums, ss_bus_uids)
            if int(uid) in bus_pos
        }
        return bus_num_to_pos.get(int(bus_num), np.nan)
    except Exception:
        return np.nan


def _map_ss_busnum_to_ppnet_bus_index(ss, pp_net, bus_num: float) -> Optional[int]:
    """Helper to map ss busnum to ppnet bus index."""
    if not np.isfinite(bus_num):
        return None
    try:
        bus_ids = pp_net.bus.index.to_numpy(dtype=int)
        bus_df = ss.Bus.as_df()[["idx"]]
        ss_bus_nums = bus_df["idx"].to_numpy(dtype=int)
        ss_bus_uids = bus_df.index.to_numpy(dtype=int)
        uid_for_num = {int(num): int(uid) for num, uid in zip(ss_bus_nums, ss_bus_uids)}
        uid = uid_for_num.get(int(bus_num))
        if uid is None:
            return None
        if uid not in set(bus_ids.tolist()):
            return None
        return int(uid)
    except Exception:
        return None


def _dc_sensitivity(ss, cont_uid: Optional[int], out_bus1: float, out_bus2: float, base_mva: float) -> Dict[str, float]:
    """Helper to dc sensitivity."""
    out = {
        "ptdf_l1_norm_outaged_line": 0.0,
        "max_abs_lodf_row": 0.0,
        "predicted_max_post_cont_loading_dc": np.nan,
    }
    if cont_uid is None and not (np.isfinite(out_bus1) and np.isfinite(out_bus2)):
        out["predicted_max_post_cont_loading_dc"] = _finite_or_neg_one(out.get("predicted_max_post_cont_loading_dc"))
        return out
    restored_line_u = None
    line_u_vals = None
    try:
        # PTDF/LODF should be computed on an intact pre-contingency topology.
        # After TDS, the outaged line may already be opened; temporarily force it in service.
        if cont_uid is not None:
            line_u_obj = getattr(ss.Line, "u", None)
            line_u_vals = getattr(line_u_obj, "v", None) if line_u_obj is not None else None
            if line_u_vals is not None and 0 <= int(cont_uid) < len(line_u_vals):
                restored_line_u = float(line_u_vals[int(cont_uid)])
                line_u_vals[int(cont_uid)] = 1.0

        from andes.interop import pandapower as ap
        from pandapower import auxiliary as aux
        from pandapower.pd2ppc import _pd2ppc
        from pandapower.pypower.idx_bus import BUS_I, BUS_TYPE, REF, PQ
        from pandapower.pypower.makeLODF import makeLODF
        from pandapower.pypower.makePTDF import makePTDF
    except Exception as e:
        _warn_once(f"DC proxy unavailable (import failed): {e}")
        out["predicted_max_post_cont_loading_dc"] = _fallback_max_loading(ss, base_mva)
        out["predicted_max_post_cont_loading_dc"] = _finite_or_neg_one(out.get("predicted_max_post_cont_loading_dc"))
        return out
    try:
        with _suppress_pandapower_interop_noise():
            pp_net = ap.to_pandapower(ss, verify=False)
            if not hasattr(pp_net, "_options") or not isinstance(pp_net._options, dict):
                pp_net._options = {}
            if "mode" not in pp_net._options:
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
        # Ensure pp_net has a reference element before ppc conversion.
        def _get_preferred_slack_busnum() -> Optional[int]:
            try:
                slack_bus = getattr(getattr(ss, "Slack", None), "bus", None)
                slack_bus_vals = getattr(slack_bus, "v", None)
                if slack_bus_vals is not None and len(slack_bus_vals) > 0:
                    val = int(slack_bus_vals[0])
                    if np.isfinite(val):
                        return val
            except Exception:
                pass
            return 39

        def _ensure_ext_grid_fallback() -> bool:
            try:
                if not hasattr(pp_net, "ext_grid"):
                    return False
                if len(pp_net.ext_grid) > 0:
                    return True
                slack_busnum = _get_preferred_slack_busnum()
                if slack_busnum is None:
                    return False
                bus_idx = _map_ss_busnum_to_ppnet_bus_index(ss, pp_net, float(slack_busnum))
                if bus_idx is None and len(pp_net.bus.index) > 0:
                    bus_idx = int(pp_net.bus.index[0])
                    _warn_once(
                        "DC proxy ext_grid fallback: slack bus mapping failed; "
                        f"using first pp bus index {bus_idx}"
                    )
                if bus_idx is None:
                    return False
                row = {col: np.nan for col in pp_net.ext_grid.columns}
                row["bus"] = int(bus_idx)
                if "vm_pu" in row:
                    row["vm_pu"] = 1.0
                if "va_degree" in row:
                    row["va_degree"] = 0.0
                if "in_service" in row:
                    row["in_service"] = True
                if "name" in row:
                    row["name"] = "dc_proxy_fallback_slack"
                if "slack_weight" in row:
                    row["slack_weight"] = 1.0
                pp_net.ext_grid.loc[len(pp_net.ext_grid)] = row
                _warn_once(
                    "DC proxy fallback: injected ext_grid at pp bus "
                    f"{bus_idx} (ss bus {slack_busnum})"
                )
                return True
            except Exception as e:
                _warn_once(f"DC proxy ext_grid fallback failed: {e}")
                return False

        def _set_slack_gen_fallback() -> bool:
            if not (hasattr(pp_net, "gen") and len(pp_net.gen) > 0):
                return False
            if "slack" not in pp_net.gen.columns:
                pp_net.gen["slack"] = False
            gen_bus = np.asarray(pp_net.gen["bus"].values, dtype=int)
            match = np.where(gen_bus == 39)[0]
            target = int(match[0]) if match.size > 0 else 0
            pp_net.gen.loc[:, "slack"] = False
            pp_net.gen.iloc[target, pp_net.gen.columns.get_loc("slack")] = True
            _warn_once(
                "DC proxy fallback: forced slack gen at bus "
                f"{int(pp_net.gen.iloc[target]['bus'])} before _pd2ppc"
            )
            return True

        try:
            has_ext_grid = hasattr(pp_net, "ext_grid") and len(pp_net.ext_grid) > 0
            has_slack_gen = (
                hasattr(pp_net, "gen")
                and len(pp_net.gen) > 0
                and "slack" in pp_net.gen.columns
                and bool(np.any(np.asarray(pp_net.gen["slack"].values, dtype=bool)))
            )
            if not has_ext_grid and not has_slack_gen:
                ok = _ensure_ext_grid_fallback()
                if not ok:
                    ok = _set_slack_gen_fallback()
                if not ok:
                    _warn_once("DC proxy fallback: no ext_grid and no generators available for slack")
        except Exception as e:
            _warn_once(f"DC proxy pre-_pd2ppc reference fallback failed: {e}")

        try:
            with _suppress_pandapower_interop_noise():
                ppc, ppci = _pd2ppc(pp_net)
        except Exception as e:
            if "No reference bus is available" not in str(e):
                raise
            # Retry with explicit slack generator fallback.
            try:
                if not _ensure_ext_grid_fallback():
                    if not _set_slack_gen_fallback():
                        _warn_once("DC proxy retry failed: no generators available to set slack")
                with _suppress_pandapower_interop_noise():
                    ppc, ppci = _pd2ppc(pp_net)
            except Exception:
                raise e
        bus = np.asarray(ppci["bus"], dtype=float)
        branch = np.asarray(np.real(ppci["branch"]), dtype=float)

        # Ensure a reference bus exists for DC sensitivity computations.
        # Prefer bus number 39 as requested; otherwise fallback to first bus row.
        ref_rows = np.where(bus[:, BUS_TYPE] == REF)[0]
        if ref_rows.size == 0:
            bus_nums = bus[:, BUS_I].astype(int)
            target_rows = np.where(bus_nums == 39)[0]
            if target_rows.size > 0:
                ref_idx = int(target_rows[0])
            elif bus.shape[0] > 0:
                ref_idx = 0
                _warn_once("DC proxy fallback: bus 39 not found; using first bus as REF")
            else:
                _warn_once("DC proxy failed: empty bus matrix after conversion")
                return out
            bus[:, BUS_TYPE] = np.where(bus[:, BUS_TYPE] == REF, PQ, bus[:, BUS_TYPE])
            bus[ref_idx, BUS_TYPE] = REF
            _warn_once(f"DC proxy fallback: forced REF bus={int(bus[ref_idx, BUS_I])}")

        out_bus1_ppc = _map_ss_busnum_to_ppc_pos(ss, pp_net, out_bus1)
        out_bus2_ppc = _map_ss_busnum_to_ppc_pos(ss, pp_net, out_bus2)

        outage_row = _find_outage_branch_row(
            branch,
            out_bus1_ppc=out_bus1_ppc,
            out_bus2_ppc=out_bus2_ppc,
            cont_uid=cont_uid,
        )
        if outage_row is None:
            _warn_once(
                f"DC proxy skipped: could not map outage to branch row "
                f"(uid={cont_uid}, bus1={out_bus1}, bus2={out_bus2}, "
                f"bus1_ppc={out_bus1_ppc}, bus2_ppc={out_bus2_ppc})"
            )
            out["predicted_max_post_cont_loading_dc"] = _fallback_max_loading(ss, base_mva)
            out["predicted_max_post_cont_loading_dc"] = _finite_or_neg_one(out.get("predicted_max_post_cont_loading_dc"))
            return out

        ptdf = np.asarray(makePTDF(ppc["baseMVA"], bus, branch), dtype=float)
        try:
            lodf = np.asarray(makeLODF(branch, ptdf), dtype=float)
        except TypeError:
            lodf = np.asarray(makeLODF(ptdf, branch), dtype=float)

        f_bus, t_bus = int(branch[outage_row, 0]), int(branch[outage_row, 1])
        bus_numbers = bus[:, 0].astype(int)
        f_idx = int(np.where(bus_numbers == f_bus)[0][0])
        t_idx = int(np.where(bus_numbers == t_bus)[0][0])
        out["ptdf_l1_norm_outaged_line"] = _to_float_or_nan(np.sum(np.abs(ptdf[:, f_idx] - ptdf[:, t_idx])))

        lodf_row = np.asarray(lodf[outage_row, :], dtype=float)
        finite_row = lodf_row[np.isfinite(lodf_row)]
        out["max_abs_lodf_row"] = (
            _to_float_or_nan(np.max(np.abs(finite_row))) if finite_row.size else 0.0
        )

        lodf_col = np.asarray(lodf[:, outage_row], dtype=float)
        if not np.all(np.isfinite(lodf_col)):
            _warn_once(
                f"DC proxy: non-finite LODF outage column for row {outage_row}; "
                "skipping predicted post-contingency loading"
            )
            out["predicted_max_post_cont_loading_dc"] = _finite_or_neg_one(out.get("predicted_max_post_cont_loading_dc"))
            return out

        pf = branch[:, 13]
        pf_pu = pf / base_mva if np.isfinite(base_mva) and base_mva > 0 else pf
        rates = branch[:, 5]
        # Fallback: if rates missing/zero, try SS ratings.
        if not np.any(np.isfinite(rates) & (rates > 0)):
            ratings_ss = np.asarray([_line_rating_from_ss(ss, i) for i in range(branch.shape[0])], dtype=float)
            rates = ratings_ss
        rates_pu = rates / base_mva if np.isfinite(base_mva) and base_mva > 0 else rates
        rates_pu = np.where(np.isfinite(rates_pu) & (rates_pu > 0), rates_pu, np.nan)
        fk = pf_pu[outage_row]
        with np.errstate(invalid="ignore", divide="ignore"):
            post = pf_pu + lodf_col * fk
            loading = np.abs(post) / rates_pu * 100.0
        loading[~np.isfinite(loading)] = np.nan
        if outage_row < loading.size:
            loading[outage_row] = np.nan
        out["predicted_max_post_cont_loading_dc"] = (
            _to_float_or_nan(np.nanmax(loading)) if np.any(np.isfinite(loading)) else _fallback_max_loading(ss, base_mva)
        )
    except Exception as e:
        _warn_once(f"DC proxy failed during computation: {e}")
        out["predicted_max_post_cont_loading_dc"] = _fallback_max_loading(ss, base_mva)
        out["predicted_max_post_cont_loading_dc"] = _finite_or_neg_one(out.get("predicted_max_post_cont_loading_dc"))
        return out
    finally:
        if line_u_vals is not None and restored_line_u is not None and cont_uid is not None:
            if 0 <= int(cont_uid) < len(line_u_vals):
                line_u_vals[int(cont_uid)] = restored_line_u
    out["predicted_max_post_cont_loading_dc"] = _finite_or_neg_one(out.get("predicted_max_post_cont_loading_dc"))
    return out
