from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_bounds(summary_json: Path) -> tuple[float, float]:
    payload = _load_json(summary_json)
    predicted = dict(payload.get("predicted_metrics", {}) or {})
    lo = list(predicted.get("limits_low") or [])
    hi = list(predicted.get("limits_high") or [])
    if len(lo) < 2 or len(hi) < 2:
        return (0.8, 1.0)
    delta_f = max(abs(float(lo[1])), abs(float(hi[1])))
    rocof = max(abs(float(lo[0])), abs(float(hi[0])))
    return (delta_f, rocof)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build replay-panel config from a tight-bounds sweep summary.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--out-config", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--strictest-limit-lines", action="store_true")
    args = parser.parse_args()

    summary_path = Path(args.summary_json).resolve()
    suite = _load_json(summary_path)
    rows = list(suite.get("rows") or [])
    feasible = [row for row in rows if str(row.get("status", "")).startswith("optimal")]
    if not feasible:
        raise RuntimeError(f"No optimal runs found in suite summary: {summary_path}")

    runs = []
    bound_rows: list[tuple[float, float]] = []
    for row in feasible:
        summary_json = Path(str(row["summary_json"])).resolve()
        delta_f, rocof = _extract_bounds(summary_json)
        label = f"{row.get('formulation_name', row.get('formulation_id', 'run'))} (|df|<={delta_f:.2f}, |RoCoF|<={rocof:.2f})"
        runs.append({"label": label, "summary_json": str(summary_json)})
        bound_rows.append((delta_f, rocof))

    if args.strictest_limit_lines:
        delta_f_limit = min(delta_f for delta_f, _ in bound_rows)
        rocof_limit = min(rocof for _, rocof in bound_rows)
    else:
        delta_f_limit = 0.8
        rocof_limit = 1.0

    first_summary = _load_json(Path(str(runs[0]["summary_json"])))
    scenario = dict(first_summary.get("scenario", {}) or {})
    cfg = {
        "tds": {
            "t_end": 20.0,
            "t_step": 0.01,
            "method": "backeuler",
            "no_tqdm": True,
            "criteria": 0,
        },
        "contingency": {
            "type": "load_step",
            "time": float(scenario.get("load_step_time", 1.0)),
            "scale": float(scenario.get("step_scale", 1.0)),
        },
        "limits": {
            "frequency_hz": 50.0,
            "delta_f_hz": float(delta_f_limit),
            "rocof_hz_per_s": float(rocof_limit),
        },
        "output": {
            "directory": str(Path(args.out_directory).resolve()),
            "title": "Replay traces under progressively tightened frequency-security boxes",
        },
        "runs": runs,
    }
    out_path = Path(args.out_config).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[build_tight_frequency_replay_config] Wrote: {out_path}")


if __name__ == "__main__":
    main()
