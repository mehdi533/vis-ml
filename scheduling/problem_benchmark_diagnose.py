from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'PyYAML'. Install requirements first:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_joblib(path: str | None, *, name: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] {name} not found at {p}", file=sys.stderr)
        return None
    try:
        import joblib
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'joblib'. Install requirements first:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
        ) from exc
    return joblib.load(p)


def _max_abs_diff(a, b) -> float:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'numpy'. Install requirements first:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
        ) from exc
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.abs(a - b)))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quick diagnosis for infeasibility in scheduling/problem_benchmark.py."
    )
    parser.add_argument(
        "--cost-config",
        default="scheduling/mtlsh_convex.yaml",
        help="Path to cost YAML (same as problem_benchmark.py).",
    )
    parser.add_argument(
        "--benchmark-path",
        default=str(Path(__file__).with_name("problem_benchmark.py")),
        help="Path to the benchmark script to inspect.",
    )
    args = parser.parse_args()

    cost_cfg = _load_yaml(Path(args.cost_config))
    models_cfg = cost_cfg.get("models", {}) or {}
    scalers_cfg = cost_cfg.get("scalers", {}) or {}
    feats_cfg = cost_cfg.get("features", {}) or {}

    y_features = list(feats_cfg.get("y_features", []) or [])
    state1 = models_cfg.get("state_dict1")
    state2 = models_cfg.get("state_dict2")

    print("== Config summary ==")
    print(f"- y_features[0:4]: {y_features[:4]}")
    print(f"- y_features[4:8]: {y_features[4:8]}")
    print(f"- models.state_dict1: {state1}")
    print(f"- models.state_dict2: {state2}")
    print(f"- scalers.y_scaler_path: {scalers_cfg.get('y_scaler_path')}")
    print(f"- scalers.y_scaler_freq_path: {scalers_cfg.get('y_scaler_freq_path')}")
    print(f"- scalers.y_scaler_dp_path: {scalers_cfg.get('y_scaler_dp_path')}")

    # --- Model-to-output slice mapping sanity check ---
    def _tag(path_str: str | None) -> str:
        if not path_str:
            return "unknown"
        s = path_str.lower()
        if "deltap" in s or "delta_p" in s or "delta" in s:
            return "dp"
        if "freq" in s:
            return "freq"
        return "unknown"

    tag1 = _tag(state1)
    tag2 = _tag(state2)

    pb_text = _read_text(Path(args.benchmark_path))
    uses_old_zip = "zip([model1, model2], [y[:4], y[4:]])" in pb_text
    uses_full_y_scaler_only = (
        "y_scaler_freq_path" not in pb_text and "y_scaler_dp_path" not in pb_text
    )

    print("\n== Likely issues ==")
    if uses_old_zip and tag1 == "dp" and tag2 == "freq":
        print(
            "- High confidence: model/output order is reversed. "
            "Your config suggests state_dict1 is deltaP and state_dict2 is freq, "
            "but the benchmark ties the first model to y[:4] (freq outputs) and the second to y[4:] (deltaP)."
        )
    elif uses_old_zip:
        print(
            "- The benchmark uses `zip([model1, model2], [y[:4], y[4:]])`. "
            "Double-check model1/model2 correspond to (freq, deltaP) in that order."
        )
    else:
        print("- Model/output slice mapping: could not detect the old zip pattern (maybe already fixed).")

    # --- Scaler mismatch check (common cause of combined infeasibility) ---
    y_scaler_full = _load_joblib(scalers_cfg.get("y_scaler_path"), name="y_scaler_path")
    y_scaler_freq = _load_joblib(
        scalers_cfg.get("y_scaler_freq_path"), name="y_scaler_freq_path"
    )
    y_scaler_dp = _load_joblib(scalers_cfg.get("y_scaler_dp_path"), name="y_scaler_dp_path")

    if y_scaler_full is not None and y_scaler_freq is not None:
        d_scale = _max_abs_diff(y_scaler_full.scale_[:4], y_scaler_freq.scale_)
        d_min = _max_abs_diff(y_scaler_full.min_[:4], y_scaler_freq.min_)
        print(f"- y_scaler(freq) mismatch vs full[:4]: max|scale diff|={d_scale:.3g}, max|min diff|={d_min:.3g}")
    else:
        print("- Skipped freq-scaler diff (missing y_scaler_path or y_scaler_freq_path).")

    if y_scaler_full is not None and y_scaler_dp is not None:
        d_scale = _max_abs_diff(y_scaler_full.scale_[4:8], y_scaler_dp.scale_)
        d_min = _max_abs_diff(y_scaler_full.min_[4:8], y_scaler_dp.min_)
        print(f"- y_scaler(dp) mismatch vs full[4:8]: max|scale diff|={d_scale:.3g}, max|min diff|={d_min:.3g}")
    else:
        print("- Skipped dp-scaler diff (missing y_scaler_path or y_scaler_dp_path).")

    if uses_full_y_scaler_only and (y_scaler_freq is not None or y_scaler_dp is not None):
        print(
            "- High confidence: benchmark ignores per-model y scalers. "
            "If the two 4-output MLPs were trained with different y scaling, "
            "combining them with a single 8-output y_scaler can make `combined` infeasible."
        )

    print("\n== What to do ==")
    print(
        "- Ensure the freq model constrains y[:4] and the deltaP model constrains y[4:].\n"
        "- Use `y_scaler_freq_path` for y[:4] and `y_scaler_dp_path` for y[4:] (and for dp_ibr scaling), "
        "or rewrite the formulation to use raw (unscaled) y and add affine scale/unscale constraints."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

