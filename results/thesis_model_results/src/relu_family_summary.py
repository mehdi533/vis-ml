import argparse
import ast
from pathlib import Path

import pandas as pd
import yaml


SMALL_MTLSH_LABELS = {"16|8", "32|16", "64|32"}


def _parse_size(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            parsed = ast.literal_eval(text)
            return [int(v) for v in parsed]
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return [int(value)]


def _format_size(value) -> str:
    sizes = _parse_size(value)
    return ",".join(str(v) for v in sizes)


def _load_csv(path: Path, rerun_hint: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {rerun_hint}")
    return pd.read_csv(path)


def _hidden_label_from_run_dir(run_dir: str) -> str:
    cfg_path = Path(run_dir) / "run_config.yaml"
    if not cfg_path.exists():
        return ""
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _format_size((cfg.get("model") or {}).get("hidden_sizes"))


def _rows_from_mlp_picnn(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        family = str(row["model"])
        size_label = _format_size(row.get("model_hidden_sizes"))
        rows.append(
            {
                "family": family,
                "label": size_label,
                "agg_rmse_mean": float(row["agg_rmse_mean"]),
                "agg_mae_mean": float(row["agg_mae_mean"]),
                "n_parameters_trainable": float(row["n_parameters_trainable"]),
                "relu_units_estimate": float(row["relu_units_estimate"]),
            }
        )
    return rows


def _rows_from_mtlgsh(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        label = "|".join(
            [
                _format_size(row.get("model_shared_sizes")),
                _format_size(row.get("model_group_shared_sizes")),
                _format_size(row.get("model_head_sizes")),
            ]
        )
        rows.append(
            {
                "family": "MTLGSH",
                "label": label,
                "agg_rmse_mean": float(row["agg_rmse_mean"]),
                "agg_mae_mean": float(row["agg_mae_mean"]),
                "n_parameters_trainable": float(row["n_parameters_trainable"]),
                "relu_units_estimate": float(row["relu_units_estimate"]),
            }
        )
    return rows


def _rows_from_mtlsh_tradeoff(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in df.iterrows():
        label = f"{_format_size(row.get('model_shared_sizes'))}|{_format_size(row.get('model_head_sizes'))}"
        if label not in SMALL_MTLSH_LABELS:
            continue
        rows.append(
            {
                "family": "MTLSH",
                "label": label,
                "agg_rmse_mean": float(row["agg_rmse_mean_mean"]),
                "agg_mae_mean": float(row["agg_mae_mean_mean"]),
                "n_parameters_trainable": float(row["n_parameters_trainable_mean"]),
                "relu_units_estimate": float(row["relu_units_estimate_mean"]),
            }
        )
    found = {row["label"] for row in rows}
    missing = SMALL_MTLSH_LABELS - found
    if missing:
        raise ValueError(
            "The merged MTLSH tradeoff table is missing the expected small variants "
            f"{sorted(missing)}. Rerun results/thesis_model_results/commands/07_mtlsh_embeddability_tradeoff.sh."
        )
    return rows


def _rows_from_she(df: pd.DataFrame) -> list[dict]:
    target_lists = df["data_target_cols"].apply(ast.literal_eval)
    freq_mask = target_lists.apply(lambda items: set(items) == {"rocof_COI", "dev_COI"})
    power_mask = target_lists.apply(lambda items: all(str(item).startswith("Delta_P_IBR_") for item in items))

    if freq_mask.sum() != 1 or power_mask.sum() != 1:
        raise ValueError(
            "Expected exactly one frequency-only and one Delta_P_IBR-only run in the She-style sweep output."
        )

    freq_row = df.loc[freq_mask].iloc[0]
    power_row = df.loc[power_mask].iloc[0]
    freq_targets = len(ast.literal_eval(freq_row["data_target_cols"]))
    power_targets = len(ast.literal_eval(power_row["data_target_cols"]))
    total_targets = freq_targets + power_targets

    agg_rmse = (
        freq_targets * float(freq_row["agg_rmse_mean"]) + power_targets * float(power_row["agg_rmse_mean"])
    ) / total_targets
    agg_mae = (
        freq_targets * float(freq_row["agg_mae_mean"]) + power_targets * float(power_row["agg_mae_mean"])
    ) / total_targets

    she_hidden = _hidden_label_from_run_dir(str(freq_row["run_dir"])) or _hidden_label_from_run_dir(
        str(power_row["run_dir"])
    )
    return [
        {
            "family": "She-2xMLP",
            "label": f"2x{she_hidden}",
            "agg_rmse_mean": agg_rmse,
            "agg_mae_mean": agg_mae,
            "n_parameters_trainable": float(freq_row["n_parameters_trainable"])
            + float(power_row["n_parameters_trainable"]),
            "relu_units_estimate": float(freq_row["relu_units_estimate"])
            + float(power_row["relu_units_estimate"]),
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the small-family ReLU-size comparison table.")
    parser.add_argument("--mtlsh-tradeoff-csv", required=True, help="Merged MTLSH tradeoff CSV from command 07.")
    parser.add_argument("--mlp-picnn-dir", required=True, help="Sweep output directory for small MLP/PICNN runs.")
    parser.add_argument("--mtlgsh-dir", required=True, help="Sweep output directory for small MTLGSH runs.")
    parser.add_argument("--she-dir", required=True, help="Sweep output directory for the She-style two-MLP runs.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV path.")
    args = parser.parse_args()

    mtlsh_df = _load_csv(
        Path(args.mtlsh_tradeoff_csv),
        "Rerun results/thesis_model_results/commands/07_mtlsh_embeddability_tradeoff.sh first.",
    )
    mlp_picnn_df = _load_csv(
        Path(args.mlp_picnn_dir) / "sweep_run_summary.csv",
        "Rerun the new small-family sweep command to regenerate MLP/PICNN rows.",
    )
    mtlgsh_df = _load_csv(
        Path(args.mtlgsh_dir) / "sweep_run_summary.csv",
        "Rerun the new small-family sweep command to regenerate MTLGSH rows.",
    )
    she_df = _load_csv(
        Path(args.she_dir) / "sweep_run_summary.csv",
        "Rerun results/thesis_model_results/commands/08_mlp_she_style_comparison.sh first.",
    )

    rows = []
    rows.extend(_rows_from_mtlsh_tradeoff(mtlsh_df))
    rows.extend(_rows_from_mlp_picnn(mlp_picnn_df))
    rows.extend(_rows_from_mtlgsh(mtlgsh_df))
    rows.extend(_rows_from_she(she_df))

    out_df = pd.DataFrame(rows)
    family_order = {"She-2xMLP": 0, "MLP": 1, "PICNN": 2, "MTLSH": 3, "MTLGSH": 4}
    out_df["family_order"] = out_df["family"].map(family_order).fillna(99).astype(int)
    out_df = out_df.sort_values(["family_order", "relu_units_estimate", "label"]).reset_index(drop=True)
    out_df = out_df.drop(columns=["family_order"])

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved ReLU-family comparison to {output_csv}")


if __name__ == "__main__":
    main()
