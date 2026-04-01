import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import pandas as pd


CORE_ARTIFACTS = [
    "run_config.yaml",
    "model.txt",
    "model_stats.json",
    "artifact_manifest.json",
    "x_scaler.pkl",
    "y_scaler.pkl",
    "metrics_by_target.csv",
    "metrics_summary.json",
    "rmse_results.txt",
    "training_summary.txt",
]


def _parse_filters(values):
    filters = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid filter '{value}'. Expected field=value.")
        key, raw = value.split("=", 1)
        filters.append((key.strip(), raw.strip()))
    return filters


def _coerce_filter_value(series: pd.Series, raw: str):
    if pd.api.types.is_bool_dtype(series):
        return raw.lower() in {"1", "true", "yes", "y"}
    if pd.api.types.is_integer_dtype(series):
        return int(raw)
    if pd.api.types.is_float_dtype(series):
        return float(raw)
    return raw


def _apply_filters(df: pd.DataFrame, filters):
    out = df
    for key, raw in filters:
        if key not in out.columns:
            raise KeyError(f"Unknown filter column '{key}'.")
        out = out[out[key] == _coerce_filter_value(out[key], raw)]
    return out


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _checkpoint_artifacts(source_run_dir: Path) -> list[str]:
    manifest_path = source_run_dir / "artifact_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        checkpoint_files = manifest.get("checkpoint_files", {})
        names = [checkpoint_files.get("best_state_dict"), checkpoint_files.get("final_state_dict")]
        return [name for name in names if name]
    return ["vis_mlp_state_dict_best.pt", "vis_mlp_state_dict.pt"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the retained model artifacts for thesis handoff.")
    parser.add_argument("--dest-dir", required=True, help="Destination folder for the retained model bundle.")
    parser.add_argument("--source-run-dir", default=None, help="Explicit source run directory.")
    parser.add_argument("--summary-csv", default=None, help="Optional sweep_run_summary.csv used for selection.")
    parser.add_argument("--filters", nargs="*", default=None, help="Optional field=value filters for selection.")
    parser.add_argument(
        "--metric",
        default="agg_rmse_mean",
        help="Metric used to select the retained run when --summary-csv is given.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Select the largest value instead of the smallest one.",
    )
    parser.add_argument("--note", default="", help="Optional note saved with the export metadata.")
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    selection_payload = {
        "metric": args.metric,
        "descending": bool(args.descending),
        "filters": list(args.filters or []),
        "note": str(args.note),
    }

    if args.source_run_dir:
        source_run_dir = Path(args.source_run_dir)
        selected_row = None
    else:
        if args.summary_csv is None:
            raise ValueError("Provide either --source-run-dir or --summary-csv.")
        df = pd.read_csv(args.summary_csv)
        df = _apply_filters(df, _parse_filters(args.filters))
        if df.empty:
            raise ValueError("No runs matched the requested filters.")
        if args.metric not in df.columns:
            raise KeyError(f"Selection metric '{args.metric}' not found in {args.summary_csv}.")
        df = df.sort_values(args.metric, ascending=not args.descending)
        selected_row = df.iloc[0].to_dict()
        source_run_dir = Path(str(selected_row["run_dir"]))
        selection_payload["selected_row"] = selected_row

    if not source_run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {source_run_dir}")

    artifacts_dir = dest_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in CORE_ARTIFACTS + _checkpoint_artifacts(source_run_dir):
        _copy_if_exists(source_run_dir / name, artifacts_dir / name)

    selection_payload["source_run_dir"] = str(source_run_dir)
    with (dest_dir / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(selection_payload, f, indent=2)

    print(f"Exported retained model bundle to {dest_dir}")


if __name__ == "__main__":
    main()
