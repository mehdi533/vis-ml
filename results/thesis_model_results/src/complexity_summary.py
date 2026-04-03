import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.workflow_utils import load_yaml


EMBEDDABILITY_MAP = {
    "MLP": ("yes", "ReLU network; standard mixed-integer embedding."),
    "MTLSH": ("yes", "Shared-trunk ReLU network; standard mixed-integer embedding."),
    "MTLGSH": ("yes", "Grouped shared-head ReLU network; standard mixed-integer embedding."),
    "FICNN": ("yes", "Convex architecture under nonnegative-weight ICNN assumptions."),
    "PICNN": ("yes", "PICNN embedding requires the designated control/input split."),
    "PICNN_MTLSH": ("yes", "PICNN-style embedding with shared-head structure and designated control split."),
    "MTLGSH_ATT": ("limited", "Attention uses softmax gating and is not used as the main embeddable surrogate."),
    "MTLGSH_KAN_SHARED": ("limited", "KAN spline layers are exploratory; no standard optimization embedding used here."),
    "MTLGSH_KAN": ("limited", "KAN spline layers are exploratory; no standard optimization embedding used here."),
}


def _architecture_spec(run_dir: Path) -> str:
    cfg = load_yaml(run_dir / "run_config.yaml")
    model_cfg = cfg.get("model", {})
    model_type = cfg.get("resolved", {}).get("model_type", run_dir.name)
    parts = [str(model_type)]
    for key in ("hidden_sizes", "shared_sizes", "group_shared_sizes", "head_sizes", "group_head_indices"):
        if key in model_cfg and model_cfg[key]:
            parts.append(f"{key}={model_cfg[key]}")
    if "attention_hidden_dim" in model_cfg:
        parts.append(f"attention_hidden_dim={model_cfg['attention_hidden_dim']}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a thesis-ready architecture complexity summary.")
    parser.add_argument("--input-dir", required=True, help="Architecture sweep output directory.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV file.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    run_df = pd.read_csv(input_dir / "sweep_run_summary.csv")
    rows = []
    for _, row in run_df.sort_values("agg_rmse_mean", ascending=True).iterrows():
        model = str(row["model"])
        embeddable, note = EMBEDDABILITY_MAP.get(model, ("unknown", "No embeddability note available."))
        run_dir = Path(str(row["run_dir"]))
        rows.append(
            {
                "model": model,
                "run_dir": str(run_dir),
                "agg_rmse_mean": row.get("agg_rmse_mean", ""),
                "agg_mae_mean": row.get("agg_mae_mean", ""),
                "best_val_loss": row.get("best_val_loss", ""),
                "n_parameters_trainable": row.get("n_parameters_trainable", ""),
                "n_parameters_total": row.get("n_parameters_total", ""),
                "relu_units_estimate": row.get("relu_units_estimate", ""),
                "train_wall_time_sec": row.get("train_wall_time_sec", ""),
                "eval_wall_time_sec": row.get("eval_wall_time_sec", ""),
                "architecture_spec": _architecture_spec(run_dir),
                "optimization_embeddable": embeddable,
                "embedding_note": note,
            }
        )

    out_df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved complexity summary to {output_csv}")


if __name__ == "__main__":
    main()
