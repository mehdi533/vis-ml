from __future__ import annotations

import argparse
import csv
import json
import sys
from glob import glob
from pathlib import Path
from typing import Any

import andes
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_GEN_DIR = ROOT / "data_generation"
if str(DATA_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_DIR))

from extract_metrics import extract_y_metrics  # type: ignore  # noqa: E402


def _resolve(path_like: str, base: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base / p).resolve()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return
    headers: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_summary_paths(run_cfg: dict[str, Any], base: Path) -> list[Path]:
    if run_cfg.get("summary_json"):
        return [_resolve(str(run_cfg["summary_json"]), base)]

    if run_cfg.get("summary_json_glob"):
        pattern = str(_resolve(str(run_cfg["summary_json_glob"]), base))
        return [Path(path).resolve() for path in sorted(glob(pattern))]

    suite_summary_json = run_cfg.get("suite_summary_json")
    if suite_summary_json:
        suite_path = _resolve(str(suite_summary_json), base)
        with suite_path.open("r", encoding="utf-8") as f:
            suite_payload = json.load(f)

        formulation_filter = {str(v) for v in list(run_cfg.get("formulation_ids") or [])}
        scenario_filter = {str(v) for v in list(run_cfg.get("scenario_ids") or [])}
        summary_paths: list[Path] = []
        for row in list(suite_payload.get("rows") or []):
            summary_json = str(row.get("summary_json", "")).strip()
            if not summary_json:
                continue
            formulation_id = str(row.get("formulation_id", row.get("run_id", "")))
            scenario_id = str(row.get("scenario_id", ""))
            if formulation_filter and formulation_id not in formulation_filter:
                continue
            if scenario_filter and scenario_id not in scenario_filter:
                continue
            summary_paths.append(_resolve(summary_json, suite_path.parent))
        return summary_paths

    raise ValueError(
        "Each replay run entry must define one of: summary_json, summary_json_glob, suite_summary_json."
    )


def _configure_tds(ss, tds_cfg: dict[str, Any]) -> None:
    ss.TDS.config.no_tqdm = bool(tds_cfg.get("no_tqdm", True))
    ss.TDS.config.criteria = int(tds_cfg.get("criteria", 0))
    ss.TDS.config.tol = float(tds_cfg.get("tol", 1e-3))
    ss.TDS.config.tf = float(tds_cfg.get("t_end", 20.0))
    ss.TDS.config.tstep = float(tds_cfg.get("t_step", 0.01))
    ss.TDS.config.fixt = int(tds_cfg.get("fixt", 0))
    ss.TDS.config.method = str(tds_cfg.get("method", "backeuler"))
    ss.TDS.config.honest = int(tds_cfg.get("honest", 1))
    ss.TDS.config.max_iter = int(tds_cfg.get("max_iter", 35))
    ss.TDS.config.shrinkt = int(tds_cfg.get("shrinkt", 1))


def _setup_pq_model(ss) -> None:
    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.config.p2i = 0
    ss.PQ.config.q2i = 0
    ss.PQ.config.pq2z = 0


def _apply_base_scale(ss, base_scale: float) -> None:
    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale


def _apply_dispatch(ss, pg_opt: np.ndarray) -> None:
    idx = 0
    for i in range(ss.PV.n):
        if idx < pg_opt.size and np.isfinite(pg_opt[idx]):
            ss.PV.p0.v[i] = float(pg_opt[idx])
        idx += 1
    for i in range(ss.Slack.n):
        if idx < pg_opt.size and np.isfinite(pg_opt[idx]):
            ss.Slack.p0.v[i] = float(pg_opt[idx])
        idx += 1


def _apply_md(ss, m_opt: np.ndarray, d_opt: np.ndarray) -> None:
    for i in range(min(ss.REGCV1.n, m_opt.size)):
        if np.isfinite(m_opt[i]):
            ss.REGCV1.M.v[i] = float(m_opt[i])
    for i in range(min(ss.REGCV1.n, d_opt.size)):
        if np.isfinite(d_opt[i]):
            ss.REGCV1.D.v[i] = float(d_opt[i])


def _add_load_step_contingency(ss, *, time: float, scale: float, pq_targets: list[str] | None) -> None:
    # ANDES `Alter` resolves `dev` by idx, not name. On systems where PQ name !=
    # idx (e.g. IEEE 118: name "PQ 1" vs idx "PQ_1"), passing names silently
    # disables every event -> no disturbance. Resolve names -> idx here.
    name_to_idx = {str(n): i for n, i in zip(ss.PQ.name.v, ss.PQ.idx.v)}
    if pq_targets:
        targets = [name_to_idx.get(str(t), t) for t in pq_targets]
    else:
        targets = list(ss.PQ.idx.v)
    for dev in targets:
        ss.add(model="Alter", param_dict=dict(t=float(time), model="PQ", dev=dev, src="Ppf", attr="v", method="*", amount=float(scale)))
        ss.add(model="Alter", param_dict=dict(t=float(time), model="PQ", dev=dev, src="Qpf", attr="v", method="*", amount=float(scale)))


def _resolve_line_dev(
    ss,
    *,
    line_uid: int | None = None,
    line_dev: Any | None = None,
) -> Any:
    line_idx = list(getattr(getattr(ss, "Line", None), "idx", None).v)
    if line_dev is not None:
        if line_dev in line_idx:
            return line_dev
        line_dev_str = str(line_dev)
        if line_dev_str in line_idx:
            return line_dev_str
    if line_uid is not None:
        uid = int(line_uid)
        if uid < 0 or uid >= len(line_idx):
            raise KeyError(
                f"<Line>: device not exist with uid={uid}. Valid line uid range is [0, {max(len(line_idx) - 1, 0)}]."
            )
        return line_idx[uid]
    raise ValueError("Either line_uid or line_dev must be provided for a line-trip contingency.")


def _add_line_trip_contingency(
    ss,
    *,
    time: float,
    line_uid: int | None = None,
    line_dev: Any | None = None,
) -> None:
    resolved_dev = _resolve_line_dev(ss, line_uid=line_uid, line_dev=line_dev)
    ss.add(
        model="Toggle",
        param_dict={
            "t": float(time),
            "model": "Line",
            "dev": resolved_dev,
        },
    )


def _lookup_metric(y_metrics: dict[str, dict[str, float]], metric_name: str, explicit_map: dict[str, str]) -> float:
    if metric_name in explicit_map:
        path = explicit_map[metric_name]
        if "." in path:
            section, key = path.split(".", 1)
            return float(y_metrics.get(section, {}).get(key, np.nan))

    for section in ("coi", "ibr_response", "bus_frequency", "bus_voltage", "bus_rocof"):
        if metric_name in y_metrics.get(section, {}):
            return float(y_metrics[section][metric_name])
    return float("nan")


def _default_metric_map(output_names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in output_names:
        if name == "rocof_COI":
            out[name] = "coi.rocof_COI"
        elif name == "dev_COI":
            out[name] = "coi.dev_COI"
        elif name.startswith("Delta_P_IBR_"):
            out[name] = f"ibr_response.{name}"
    return out


def _metric_category(metric_name: str) -> str:
    name = str(metric_name)
    if name == "rocof_COI":
        return "rocof"
    if name == "dev_COI":
        return "frequency_deviation"
    if name.startswith("Delta_P_IBR_"):
        return "ibr_power"
    return "other"


def _load_ibr_limit_map(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    artifacts = dict(summary.get("artifacts", {}) or {})
    raw = str(artifacts.get("dispatch_impact_csv", "")).strip()
    if not raw:
        return {}
    path = Path(raw)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    ibr_df = df.loc[df["row_type"].astype(str) == "ibr_summary"].copy()
    if ibr_df.empty:
        return {}

    out: dict[str, dict[str, float]] = {}
    for _, row in ibr_df.iterrows():
        idx = int(float(row.get("index", 0)))
        out[f"Delta_P_IBR_{idx + 1}"] = {
            "scheduled_headroom_up": float(row.get("headroom_up", np.nan)),
            "scheduled_headroom_down": float(row.get("headroom_down", np.nan)),
            "scheduled_reserve_up": float(row.get("reserve_up", np.nan)),
            "physical_min": float(row.get("delta_p_physical_min", np.nan)),
            "physical_max": float(row.get("delta_p_physical_max", np.nan)),
        }
    return out


def _limit_membership(value: float, low: float, high: float) -> int:
    if not (np.isfinite(value) and np.isfinite(low) and np.isfinite(high)):
        return -1
    return int(low - 1e-8 <= value <= high + 1e-8)


def _feasibility_case(predicted_ok: int, replay_ok: int) -> str:
    if predicted_ok < 0 or replay_ok < 0:
        return "not_checked"
    if predicted_ok and replay_ok:
        return "true_safe"
    if predicted_ok and not replay_ok:
        return "false_safe"
    if (not predicted_ok) and replay_ok:
        return "false_unsafe"
    return "true_unsafe"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay optimized schedules in ANDES and compare predicted vs replayed metrics.")
    parser.add_argument("--config", required=True, help="Replay config YAML.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    runs = list(cfg.get("runs") or [])
    if not runs:
        raise ValueError(f"No runs configured in replay config: {cfg_path}")

    tds_cfg = dict(cfg.get("tds", {}) or {})
    contingency_cfg = dict(cfg.get("contingency", {}) or {})
    explicit_metric_map = dict(cfg.get("output_metric_map", {}) or {})
    output_cfg = dict(cfg.get("output", {}) or {})

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for run in runs:
        summary_paths = _load_summary_paths(dict(run or {}), cfg_path.parent)
        if not summary_paths:
            raise FileNotFoundError(
                f"No optimization summary JSONs matched replay entry: {json.dumps(run, indent=2)}"
            )

        for summary_path in summary_paths:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)

            run_id = str(summary.get("run_id", run.get("run_id", summary_path.stem))).strip()
            formulation_id = str(summary.get("formulation_id", run.get("formulation_id", run_id))).strip()
            formulation_name = str(summary.get("formulation_name", formulation_id))
            scenario_id = str(summary.get("scenario_id", run.get("scenario_id", ""))).strip()
            scenario_name = str(summary.get("scenario_name", scenario_id))

            opt_cfg_path_raw = run.get("optimization_config") or summary.get("config_path")
            if not opt_cfg_path_raw:
                raise ValueError(f"Missing optimization_config for run_id={run_id} and no config_path in summary.")
            opt_cfg_path = _resolve(str(opt_cfg_path_raw), cfg_path.parent)
            with opt_cfg_path.open("r", encoding="utf-8") as f:
                opt_cfg = yaml.safe_load(f) or {}

            system_case = str(opt_cfg["system"]["case"])
            base_scale = float(opt_cfg["scenario"]["base_scale"])
            default_step_scale = float(opt_cfg["scenario"]["step_scale"])
            default_step_time = float(opt_cfg["scenario"]["load_step_time"])

            pg_opt = np.asarray(summary.get("dispatch_summary", {}).get("pg_opt", []), dtype=float)
            m_opt = np.asarray(summary.get("dispatch_summary", {}).get("m_opt", []), dtype=float)
            d_opt = np.asarray(summary.get("dispatch_summary", {}).get("d_opt", []), dtype=float)
            pred_names = list(summary.get("predicted_metrics", {}).get("names", []))
            pred_values = np.asarray(summary.get("predicted_metrics", {}).get("values", []), dtype=float)
            pred_lo = np.asarray(summary.get("predicted_metrics", {}).get("limits_low", []), dtype=float)
            pred_hi = np.asarray(summary.get("predicted_metrics", {}).get("limits_high", []), dtype=float)
            ibr_limit_map = _load_ibr_limit_map(summary)

            metric_map = _default_metric_map(pred_names)
            metric_map.update(explicit_metric_map)

            ss = andes.load(system_case, setup=False)
            ss.config.freq = float(opt_cfg.get("system", {}).get("frequency_hz", 50.0))
            system_base_mva = float(getattr(getattr(ss, "config", None), "mva", np.nan))
            _apply_base_scale(ss, base_scale)
            _apply_dispatch(ss, pg_opt)
            _apply_md(ss, m_opt, d_opt)
            _setup_pq_model(ss)

            cont_type = str(contingency_cfg.get("type", "load_step")).lower()
            if cont_type == "load_step":
                step_time = float(contingency_cfg.get("time", default_step_time))
                step_scale = float(contingency_cfg.get("scale", default_step_scale))
                pq_targets = contingency_cfg.get("pq_targets")
                if isinstance(pq_targets, list):
                    targets = [str(v) for v in pq_targets]
                else:
                    targets = None
                _add_load_step_contingency(ss, time=step_time, scale=step_scale, pq_targets=targets)
            elif cont_type == "line_trip":
                trip_time = float(contingency_cfg.get("time", default_step_time))
                raw_line_uid = contingency_cfg.get("line_uid")
                raw_line_dev = contingency_cfg.get("line_dev")
                line_uid = int(raw_line_uid) if raw_line_uid is not None else None
                line_dev = int(raw_line_dev) if raw_line_dev is not None else None
                _add_line_trip_contingency(ss, time=trip_time, line_uid=line_uid, line_dev=line_dev)
            elif cont_type == "none":
                pass
            else:
                raise ValueError(f"Unsupported replay contingency type: {cont_type}")

            ss.setup()
            ss.PFlow.run()
            _configure_tds(ss, tds_cfg)
            ss.TDS.init()
            success = bool(ss.TDS.run())
            ss.TDS.load_plotter()
            y_metrics = extract_y_metrics(ss=ss, plotter=ss.TDS.plotter)

            replay_vals = np.full(len(pred_names), np.nan, dtype=float)
            abs_err = np.full(len(pred_names), np.nan, dtype=float)
            sq_err = np.full(len(pred_names), np.nan, dtype=float)
            rel_err = np.full(len(pred_names), np.nan, dtype=float)
            predicted_ok_flags: list[int] = []
            replay_ok_flags: list[int] = []
            feasibility_cases: list[str] = []
            n_viol = 0
            n_headroom_exceeded = 0
            n_physical_limit_exceeded = 0
            rocof_violation_count = 0
            frequency_violation_count = 0
            ibr_power_violation_count = 0

            for i, name in enumerate(pred_names):
                replay_vals[i] = _lookup_metric(y_metrics, name, metric_map)
                pred_val = float(pred_values[i]) if i < pred_values.size else np.nan
                replay_val = float(replay_vals[i])
                if np.isfinite(pred_val) and np.isfinite(replay_val):
                    diff = replay_val - pred_val
                    abs_err[i] = abs(diff)
                    sq_err[i] = diff * diff
                    rel_err[i] = abs_err[i] / max(1e-12, abs(replay_val))
                low = pred_lo[i] if i < pred_lo.size else np.nan
                high = pred_hi[i] if i < pred_hi.size else np.nan
                predicted_ok = _limit_membership(pred_val, low, high)
                replay_ok = _limit_membership(replay_val, low, high)
                feasibility_case = _feasibility_case(predicted_ok, replay_ok)
                metric_category = _metric_category(name)
                if replay_ok == 0:
                    n_viol += 1
                    if metric_category == "rocof":
                        rocof_violation_count += 1
                    elif metric_category == "frequency_deviation":
                        frequency_violation_count += 1
                    elif metric_category == "ibr_power":
                        ibr_power_violation_count += 1
                predicted_ok_flags.append(predicted_ok)
                replay_ok_flags.append(replay_ok)
                feasibility_cases.append(feasibility_case)

                violation_mag = 0.0
                if np.isfinite(replay_val):
                    if np.isfinite(low) and replay_val < low:
                        violation_mag = float(low - replay_val)
                    elif np.isfinite(high) and replay_val > high:
                        violation_mag = float(replay_val - high)

                ibr_limits = dict(ibr_limit_map.get(name, {}) or {})
                scheduled_headroom_up = float(ibr_limits.get("scheduled_headroom_up", np.nan))
                scheduled_headroom_down = float(ibr_limits.get("scheduled_headroom_down", np.nan))
                scheduled_reserve_up = float(ibr_limits.get("scheduled_reserve_up", np.nan))
                physical_min = float(ibr_limits.get("physical_min", np.nan))
                physical_max = float(ibr_limits.get("physical_max", np.nan))

                headroom_violation_magnitude = 0.0
                headroom_violation_flag = 0
                if np.isfinite(replay_val):
                    if np.isfinite(scheduled_headroom_up) and replay_val > scheduled_headroom_up + 1e-8:
                        headroom_violation_magnitude = float(replay_val - scheduled_headroom_up)
                        headroom_violation_flag = 1
                    elif np.isfinite(scheduled_headroom_down) and replay_val < -scheduled_headroom_down - 1e-8:
                        headroom_violation_magnitude = float((-scheduled_headroom_down) - replay_val)
                        headroom_violation_flag = 1
                if headroom_violation_flag:
                    n_headroom_exceeded += 1

                physical_limit_violation_magnitude = 0.0
                physical_limit_violation_flag = 0
                if np.isfinite(replay_val):
                    if np.isfinite(physical_min) and replay_val < physical_min - 1e-8:
                        physical_limit_violation_magnitude = float(physical_min - replay_val)
                        physical_limit_violation_flag = 1
                    elif np.isfinite(physical_max) and replay_val > physical_max + 1e-8:
                        physical_limit_violation_magnitude = float(replay_val - physical_max)
                        physical_limit_violation_flag = 1
                if physical_limit_violation_flag:
                    n_physical_limit_exceeded += 1

                relevant_limit = np.nan
                if metric_category == "ibr_power" and np.isfinite(replay_val):
                    if replay_val >= 0:
                        relevant_limit = scheduled_headroom_up if np.isfinite(scheduled_headroom_up) else physical_max
                    else:
                        relevant_limit = -scheduled_headroom_down if np.isfinite(scheduled_headroom_down) else physical_min
                elif np.isfinite(replay_val):
                    relevant_limit = high if replay_val >= 0 else low

                detail_rows.append(
                    {
                        "run_id": run_id,
                        "formulation_id": formulation_id,
                        "formulation_name": formulation_name,
                        "scenario_id": scenario_id,
                        "scenario_name": scenario_name,
                        "summary_json": str(summary_path),
                        "optimization_config": str(opt_cfg_path),
                        "metric_name": name,
                        "predicted_value": pred_val,
                        "replayed_value": replay_val,
                        "signed_error": (replay_val - pred_val) if np.isfinite(pred_val) and np.isfinite(replay_val) else np.nan,
                        "abs_error": float(abs_err[i]) if np.isfinite(abs_err[i]) else np.nan,
                        "squared_error": float(sq_err[i]) if np.isfinite(sq_err[i]) else np.nan,
                        "rel_error": float(rel_err[i]) if np.isfinite(rel_err[i]) else np.nan,
                        "limit_low": float(low) if np.isfinite(low) else np.nan,
                        "limit_high": float(high) if np.isfinite(high) else np.nan,
                        "predicted_within_limits": int(predicted_ok),
                        "replayed_within_limits": int(replay_ok),
                        "is_replay_within_limits": int(replay_ok),
                        "feasibility_case": feasibility_case,
                        "replay_violation_magnitude": violation_mag,
                        "metric_category": metric_category,
                        "system_base_mva": system_base_mva,
                        "scheduled_headroom_up": scheduled_headroom_up,
                        "scheduled_headroom_down": scheduled_headroom_down,
                        "scheduled_reserve_up": scheduled_reserve_up,
                        "physical_min": physical_min,
                        "physical_max": physical_max,
                        "scheduled_headroom_violation_flag": int(headroom_violation_flag),
                        "scheduled_headroom_violation_magnitude": float(headroom_violation_magnitude),
                        "physical_limit_violation_flag": int(physical_limit_violation_flag),
                        "physical_limit_violation_magnitude": float(physical_limit_violation_magnitude),
                        "relevant_limit": float(relevant_limit) if np.isfinite(relevant_limit) else np.nan,
                    }
                )

            valid_sq = sq_err[np.isfinite(sq_err)]
            summary_rows.append(
                {
                    "run_id": run_id,
                    "formulation_id": formulation_id,
                    "formulation_name": formulation_name,
                    "scenario_id": scenario_id,
                    "scenario_name": scenario_name,
                    "status_from_optimization": str(summary.get("status", "")),
                    "tds_success": int(success),
                    "contingency_type": cont_type,
                    "n_metrics_compared": int(len(pred_names)),
                    "n_metrics_with_limits": int(sum(v >= 0 for v in predicted_ok_flags)),
                    "n_predicted_within_limits": int(sum(v == 1 for v in predicted_ok_flags)),
                    "n_replayed_within_limits": int(sum(v == 1 for v in replay_ok_flags)),
                    "n_false_safe": int(sum(case == "false_safe" for case in feasibility_cases)),
                    "n_false_unsafe": int(sum(case == "false_unsafe" for case in feasibility_cases)),
                    "mean_abs_error": float(np.nanmean(abs_err)) if abs_err.size else np.nan,
                    "rmse": float(np.sqrt(np.nanmean(valid_sq))) if valid_sq.size else np.nan,
                    "max_abs_error": float(np.nanmax(abs_err)) if abs_err.size else np.nan,
                    "mean_rel_error": float(np.nanmean(rel_err)) if rel_err.size else np.nan,
                    "max_rel_error": float(np.nanmax(rel_err)) if rel_err.size else np.nan,
                    "replay_violation_count": int(n_viol),
                    "rocof_violation_count": int(rocof_violation_count),
                    "frequency_violation_count": int(frequency_violation_count),
                    "ibr_power_violation_count": int(ibr_power_violation_count),
                    "scheduled_headroom_exceedance_count": int(n_headroom_exceeded),
                    "physical_limit_exceedance_count": int(n_physical_limit_exceeded),
                    "system_base_mva": system_base_mva,
                    "summary_json": str(summary_path),
                    "optimization_config": str(opt_cfg_path),
                }
            )

    out_summary_csv = _resolve(str(output_cfg.get("summary_csv", "results/replay_summary.csv")), cfg_path.parent)
    out_detail_csv = _resolve(str(output_cfg.get("detail_csv", "results/replay_detail.csv")), cfg_path.parent)
    out_summary_json = _resolve(str(output_cfg.get("summary_json", "results/replay_summary.json")), cfg_path.parent)

    _write_csv(out_summary_csv, summary_rows)
    _write_csv(out_detail_csv, detail_rows)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    with out_summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_path": str(cfg_path),
                "n_runs": len(summary_rows),
                "summary_rows": summary_rows,
                "detail_rows_count": len(detail_rows),
                "summary_csv": str(out_summary_csv),
                "detail_csv": str(out_detail_csv),
            },
            f,
            indent=2,
        )

    print(f"[replay_validation] Wrote: {out_summary_csv}")
    print(f"[replay_validation] Wrote: {out_detail_csv}")
    print(f"[replay_validation] Wrote: {out_summary_json}")


if __name__ == "__main__":
    main()
