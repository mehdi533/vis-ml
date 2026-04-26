import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rank_by_seed(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, seed_df in run_df.groupby("seed", dropna=False):
        ranked = seed_df.sort_values(["agg_rmse_mean", "agg_mae_mean", "model"], ascending=[True, True, True])
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append({"seed": seed, "model": row["model"], "rank": rank})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate shortlist seed-robustness results.")
    parser.add_argument("--input-dir", required=True, help="Seed robustness sweep output directory.")
    parser.add_argument("--output-csv", required=True, help="Aggregated run-level summary CSV.")
    parser.add_argument("--by-label-csv", required=True, help="Aggregated label-level summary CSV.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    run_df = pd.read_csv(input_dir / "sweep_run_summary.csv")
    label_df = pd.read_csv(input_dir / "sweep_results.csv")

    rank_df = _rank_by_seed(run_df)
    rank_summary = (
        rank_df.groupby("model", dropna=False)["rank"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "rank_mean",
                "std": "rank_std",
                "min": "rank_best",
                "max": "rank_worst",
            }
        )
    )

    run_summary = (
        run_df.groupby("model", dropna=False)[
            [
                "agg_rmse_mean",
                "agg_mae_mean",
                "max_rmse",
                "best_val_loss",
                "n_parameters_trainable",
                "relu_units_estimate",
                "train_wall_time_sec",
            ]
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    run_summary.columns = [
        "model" if col == ("model", "") else "_".join(str(part) for part in col if part)
        for col in run_summary.columns
    ]
    run_summary["n_seeds"] = run_df.groupby("model", dropna=False)["seed"].nunique().values
    run_summary = run_summary.merge(rank_summary, on="model", how="left")
    run_summary.to_csv(args.output_csv, index=False)

    label_summary = (
        label_df.groupby(["model", "label"], dropna=False)[["rmse", "mae", "rmse_norm", "mae_norm"]]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    label_summary.columns = [
        col if isinstance(col, str) else "_".join(str(part) for part in col if part)
        for col in label_summary.columns
    ]
    label_summary.to_csv(args.by_label_csv, index=False)
    print(f"Saved seed robustness summaries to {args.output_csv} and {args.by_label_csv}")


if __name__ == "__main__":
    main()
