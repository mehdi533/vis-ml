#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "results/thesis_optimization_results/outputs/plot_data/cost_vs_replay_safety_scatter.csv"
DEFAULT_OUT_PDF = ROOT / "results/thesis_optimization_results/outputs/figures/global_load_mismatch_cost_sweep_by_formulation.pdf"


FORMULATION_ORDER = [
    "ed",
    "ed_line",
    "ed_line_n1",
    "ed_surrogate",
    "ed_line_n1_surrogate",
]

FORMULATION_LABEL = {
    "ed": "A",
    "ed_line": "B",
    "ed_line_n1": "C",
    "ed_surrogate": "D",
    "ed_line_n1_surrogate": "E",
}

FORMULATION_COLOR = {
    "ed": "#7f8c8d",
    "ed_line": "#2c7fb8",
    "ed_line_n1": "#41ab5d",
    "ed_surrogate": "#f28e2b",
    "ed_line_n1_surrogate": "#d95f02",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot global load-mismatch objective cost as scatter points by base scale with A-E formulation colors."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-pdf", default=str(DEFAULT_OUT_PDF))
    args = parser.parse_args()

    source = Path(args.source)
    out_pdf = Path(args.out_pdf)

    df = pd.read_csv(source)
    df = df.loc[df["scenario_id"].astype(str).str.startswith("global_")].copy()
    df = df.loc[df["objective_total"].notna()].copy()
    if df.empty:
        raise SystemExit(f"No global_* rows with objective_total found in {source}")

    # Keep only formulations present in the canonical A..E order.
    present = [fid for fid in FORMULATION_ORDER if fid in set(df["formulation_id"].unique())]
    if not present:
        raise SystemExit("None of the expected formulations A..E were found in the source data.")

    df["base_scale"] = df["base_scale"].astype(float)
    df["objective_total"] = df["objective_total"].astype(float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)

    offsets = {
        "ed": -0.016,
        "ed_line": -0.008,
        "ed_line_n1": 0.0,
        "ed_surrogate": 0.008,
        "ed_line_n1_surrogate": 0.016,
    }

    legend_handles = []
    for fid in present:
        color = FORMULATION_COLOR.get(fid, "#333333")
        sub = df.loc[df["formulation_id"] == fid].copy()
        if sub.empty:
            continue
        x = sub["base_scale"] + offsets.get(fid, 0.0)
        y = sub["objective_total"]
        ax.scatter(
            x,
            y,
            s=54,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=FORMULATION_LABEL.get(fid, fid),
            )
        )

    base_scales = sorted(df["base_scale"].unique())
    ax.set_xlabel("Base scale")
    ax.set_ylabel("Objective cost [$]")
    ax.set_title("Global load-mismatch sweep: objective cost by base scale")
    ax.set_xticks(base_scales)
    ax.set_xticklabels([f"{x:.2f}" for x in base_scales])
    ax.legend(handles=legend_handles, title="Formulation", ncol=len(legend_handles), fontsize=9, title_fontsize=9, loc="best")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
