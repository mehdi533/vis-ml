import argparse
from pathlib import Path

import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join(str(part) for part in col if part))
        else:
            flat_cols.append(str(col))
    df.columns = flat_cols
    return df


def _read_run_summary(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "sweep_run_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing run summary: {path}")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and summarize multiple sweep_run_summary.csv files.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Sweep output directories to merge.")
    parser.add_argument("--group-by", nargs="+", required=True, help="Columns used to group rows.")
    parser.add_argument("--value-cols", nargs="+", required=True, help="Metric columns to aggregate.")
    parser.add_argument(
        "--agg-fns",
        nargs="*",
        default=["mean", "std", "min", "max"],
        help="Aggregation functions applied to value columns.",
    )
    parser.add_argument("--output-csv", required=True, help="Destination CSV path.")
    args = parser.parse_args()

    frames = [_read_run_summary(Path(input_dir)) for input_dir in args.input_dirs]
    merged = pd.concat(frames, ignore_index=True)
    if "run_dir" in merged.columns:
        merged = merged.drop_duplicates(subset=["run_dir"], keep="last")

    grouped = (
        merged.groupby(args.group_by, dropna=False)[args.value_cols].agg(args.agg_fns).reset_index()
    )
    grouped = _flatten_columns(grouped)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_csv, index=False)
    print(f"Saved merged summary to {output_csv}")


if __name__ == "__main__":
    main()
