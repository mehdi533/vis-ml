import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import pandas as pd


def _default_source_path(input_dir: Path, source: str) -> Path:
    if source == "run":
        return input_dir / "sweep_run_summary.csv"
    return input_dir / "sweep_results.csv"


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
            raise KeyError(f"Unknown filter column '{key}'. Available columns: {list(out.columns)}")
        value = _coerce_filter_value(out[key], raw)
        out = out[out[key] == value]
    return out


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join(str(part) for part in col if part))
        else:
            flat_cols.append(str(col))
    df.columns = flat_cols
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize model sweep outputs for thesis tables.")
    parser.add_argument("--input-dir", required=True, help="Sweep output directory.")
    parser.add_argument(
        "--source",
        choices=["run", "label"],
        default="run",
        help="Use per-run or per-label summary data.",
    )
    parser.add_argument("--input-csv", default=None, help="Optional explicit CSV instead of the default file.")
    parser.add_argument("--group-by", nargs="*", default=None, help="Columns used to group rows.")
    parser.add_argument("--value-cols", nargs="*", default=None, help="Metric columns to aggregate.")
    parser.add_argument(
        "--agg-fns",
        nargs="*",
        default=["mean", "std", "min", "max"],
        help="Aggregation functions used when grouping.",
    )
    parser.add_argument("--filters", nargs="*", default=None, help="Optional field=value filters.")
    parser.add_argument("--sort-by", default=None, help="Column used to sort the final table.")
    parser.add_argument("--descending", action="store_true", help="Sort descending instead of ascending.")
    parser.add_argument("--top-k", type=int, default=None, help="Keep only the first K rows after sorting.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV file.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    input_csv = Path(args.input_csv) if args.input_csv else _default_source_path(input_dir, args.source)
    df = pd.read_csv(input_csv)
    df = _apply_filters(df, _parse_filters(args.filters))

    if args.group_by:
        if args.value_cols is None:
            args.value_cols = (
                ["agg_rmse_mean", "agg_mae_mean", "best_val_loss"]
                if args.source == "run"
                else ["rmse", "mae", "rmse_norm", "mae_norm"]
            )
        grouped = df.groupby(args.group_by, dropna=False)[args.value_cols].agg(args.agg_fns).reset_index()
        grouped = _flatten_columns(grouped)

        if args.source == "run" and args.sort_by and "run_dir" in df.columns:
            best_idx = (
                df.sort_values(args.sort_by, ascending=not args.descending)
                .groupby(args.group_by, dropna=False)
                .head(1)
            )
            best_cols = args.group_by + ["run_dir"]
            best_idx = best_idx[best_cols].rename(columns={"run_dir": "best_run_dir"})
            grouped = grouped.merge(best_idx, on=args.group_by, how="left")
        out_df = grouped
    else:
        out_df = df

    if args.sort_by:
        out_df = out_df.sort_values(args.sort_by, ascending=not args.descending)
    if args.top_k is not None:
        out_df = out_df.head(int(args.top_k))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved summary to {output_csv}")


if __name__ == "__main__":
    main()
