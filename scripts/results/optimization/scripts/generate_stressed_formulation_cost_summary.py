#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "results/thesis_optimization_results/outputs/plot_data/cost_vs_replay_safety_scatter.csv"
DEFAULT_TABLE_CSV = ROOT / "results/thesis_optimization_results/outputs/tables/stressed_formulation_cost_summary.csv"
DEFAULT_TABLE_TEX = ROOT / "results/thesis_optimization_results/outputs/tables/stressed_formulation_cost_summary.tex"
DEFAULT_FIGURE = ROOT / "results/thesis_optimization_results/outputs/figures/stressed_formulation_cost_summary.pdf"


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _fmt_optional(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "--"
    try:
        if pd.isna(x):
            return "--"
    except TypeError:
        pass
    return _fmt(float(x), digits)


def _k(x: float) -> float:
    return float(x) / 1000.0


def _fmt_scaled_cost(x: float, digits: int = 3) -> str:
    x = _k(x)
    if abs(x) < 0.5 * 10 ** (-digits):
        x = 0.0
    return _fmt(x, digits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the stressed formulation cost table and PDF figure.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-csv", default=str(DEFAULT_TABLE_CSV))
    parser.add_argument("--out-tex", default=str(DEFAULT_TABLE_TEX))
    parser.add_argument("--out-pdf", default=str(DEFAULT_FIGURE))
    parser.add_argument("--scenario-id", default="b0p600_s1p200_t1p000")
    args = parser.parse_args()

    source = Path(args.source)
    out_csv = Path(args.out_csv)
    out_tex = Path(args.out_tex)
    out_pdf = Path(args.out_pdf)

    df = pd.read_csv(source)
    df = df.loc[df["scenario_id"] == args.scenario_id].copy()
    if df.empty:
        raise SystemExit(f"No rows found for scenario_id={args.scenario_id!r} in {source}")

    order = [
        ("ed", "A"),
        ("ed_line", "B"),
        ("ed_line_n1", "C"),
        ("ed_surrogate", "D"),
        ("ed_line_n1_surrogate", "E"),
    ]
    labels = {
        "ed": "A",
        "ed_line": "B",
        "ed_line_n1": "C",
        "ed_surrogate": "D",
        "ed_line_n1_surrogate": "E",
    }
    display_names = {
        "ed": "ED",
        "ed_line": "ED + Line",
        "ed_line_n1": "ED + Line + N-1",
        "ed_surrogate": "ED + Surrogate",
        "ed_line_n1_surrogate": "ED + Line + N-1 + Surrogate",
    }

    rows = []
    ed_row = df.loc[df["formulation_id"] == "ed"].iloc[0]
    ed_dispatch_only = float(ed_row["objective_total"]) - float(ed_row["objective_reserve_only"])
    ed_total = ed_dispatch_only + float(ed_row["objective_reserve_up_only"])
    for fid, letter in order:
        row = df.loc[df["formulation_id"] == fid].iloc[0]
        total_raw = float(row["objective_total"])
        reserve_postcont = row.get("objective_reserve_postcont_only")
        dispatch_only = float(row["objective_total"]) - float(row["objective_reserve_only"])
        total = dispatch_only + float(row["objective_reserve_up_only"])
        dispatch_impact_csv = Path(str(row["dispatch_impact_csv"]))
        dispatch_df = pd.read_csv(dispatch_impact_csv)
        gen_df = dispatch_df.loc[dispatch_df["row_type"] == "generator_dispatch"].copy()
        ibr_df = dispatch_df.loc[dispatch_df["row_type"] == "ibr_summary"].copy()
        ibr_gen_indices = set(int(x) for x in ibr_df["gen_index"].dropna().tolist())
        reserve_sg_cost = float(
            gen_df.loc[~gen_df["index"].isin(ibr_gen_indices), "reserve_up_cost_component"].fillna(0.0).sum()
        )
        reserve_ibr_cost_net = float(row["objective_reserve_only"]) - reserve_sg_cost
        rows.append(
            {
                "formulation": f"{letter}) {display_names[fid]}",
                "objective_cost_raw": total_raw,
                "objective_cost_total": total,
                "dispatch_only_cost": dispatch_only,
                "reserve_sg_cost": reserve_sg_cost,
                "reserve_ibr_cost_net": reserve_ibr_cost_net,
                "objective_reserve_up_only": float(row["objective_reserve_up_only"]),
                "objective_reserve_postcont_only": 0.0 if pd.isna(reserve_postcont) else float(reserve_postcont),
                "solve_time_sec": float(row["solve_time_sec"]),
                "delta_vs_ed_pct": 100.0 * (total - ed_total) / ed_total,
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    latex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{A} & \textbf{B} & \textbf{C} & \textbf{D} & \textbf{E} \\",
        r"\midrule",
        f"Objective cost [$10^3$ obj. units] & {_fmt_scaled_cost(out_df.loc[0, 'objective_cost_total'])} & {_fmt_scaled_cost(out_df.loc[1, 'objective_cost_total'])} & {_fmt_scaled_cost(out_df.loc[2, 'objective_cost_total'])} & {_fmt_scaled_cost(out_df.loc[3, 'objective_cost_total'])} & {_fmt_scaled_cost(out_df.loc[4, 'objective_cost_total'])} \\\\",
        f"Dispatch-only cost [$10^3$ dispatch units] & {_fmt_scaled_cost(out_df.loc[0, 'dispatch_only_cost'])} & {_fmt_scaled_cost(out_df.loc[1, 'dispatch_only_cost'])} & {_fmt_scaled_cost(out_df.loc[2, 'dispatch_only_cost'])} & {_fmt_scaled_cost(out_df.loc[3, 'dispatch_only_cost'])} & {_fmt_scaled_cost(out_df.loc[4, 'dispatch_only_cost'])} \\\\",
        f"Reserve SG cost [$10^3$ obj. units] & {_fmt_scaled_cost(out_df.loc[0, 'reserve_sg_cost'])} & {_fmt_scaled_cost(out_df.loc[1, 'reserve_sg_cost'])} & {_fmt_scaled_cost(out_df.loc[2, 'reserve_sg_cost'])} & {_fmt_scaled_cost(out_df.loc[3, 'reserve_sg_cost'])} & {_fmt_scaled_cost(out_df.loc[4, 'reserve_sg_cost'])} \\\\",
        f"Reserve IBR cost [$10^3$ obj. units] & {_fmt_scaled_cost(out_df.loc[0, 'reserve_ibr_cost_net'])} & {_fmt_scaled_cost(out_df.loc[1, 'reserve_ibr_cost_net'])} & {_fmt_scaled_cost(out_df.loc[2, 'reserve_ibr_cost_net'])} & {_fmt_scaled_cost(out_df.loc[3, 'reserve_ibr_cost_net'])} & {_fmt_scaled_cost(out_df.loc[4, 'reserve_ibr_cost_net'])} \\\\",
        f"Solve time [s] & {_fmt(out_df.loc[0, 'solve_time_sec'], 6)} & {_fmt(out_df.loc[1, 'solve_time_sec'], 6)} & {_fmt(out_df.loc[2, 'solve_time_sec'], 6)} & {_fmt(out_df.loc[3, 'solve_time_sec'], 6)} & {_fmt(out_df.loc[4, 'solve_time_sec'], 6)} \\\\",
        f"$\\Delta$ vs ED [\\%] (on objective cost) & {_fmt(out_df.loc[0, 'delta_vs_ed_pct'], 2)} & {_fmt(out_df.loc[1, 'delta_vs_ed_pct'], 2)} & {_fmt(out_df.loc[2, 'delta_vs_ed_pct'], 2)} & {_fmt(out_df.loc[3, 'delta_vs_ed_pct'], 2)} & {_fmt(out_df.loc[4, 'delta_vs_ed_pct'], 2)} \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Cost decomposition for the stressed comparison case \texttt{b0p600\_s1p200\_t1p000}. Costs are scaled to $10^3$ units for readability. The reported objective keeps dispatch plus pre-contingency reserve cost and excludes the post-contingency credit term. The reserve split is reported as net objective contributions from synchronous-generator reserve and IBR reserve. Columns A--E match the formulation ladder used in the chapter.}",
        r"\label{tab:stressed_formulation_cost_summary}",
        r"\end{table}",
        "",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(latex), encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    palette = ["#7f8c8d", "#2c7fb8", "#41ab5d", "#f28e2b", "#d95f02"]
    x = range(len(out_df))
    letters = ["A", "B", "C", "D", "E"]
    total_vals = out_df["objective_cost_total"].tolist()
    solve_vals = out_df["solve_time_sec"].tolist()

    ax = axes[0]
    bars = ax.bar(x, total_vals, color=palette, width=0.72)
    ax.set_xticks(list(x), letters)
    ax.set_ylabel("Solved objective")
    ax.set_title("Objective by formulation")
    for bar, delta in zip(bars, out_df["delta_vs_ed_pct"].tolist()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{delta:+.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    ax = axes[1]
    bars = ax.bar(x, solve_vals, color=palette, width=0.72)
    ax.set_xticks(list(x), letters)
    ax.set_ylabel("Solve time [s]")
    ax.set_title("Solve time by formulation")
    ax.set_yscale("log")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            _fmt(bar.get_height(), 3),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    fig.suptitle(r"Stressed case \texttt{b0p600\_s1p200\_t1p000}: formulations A--E")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
