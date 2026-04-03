from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def classify_limit_case(predicted_ok: float, replay_ok: float) -> str:
    if predicted_ok < 0 or replay_ok < 0:
        return "not_checked"
    if predicted_ok >= 0.5 and replay_ok >= 0.5:
        return "true_safe"
    if predicted_ok >= 0.5 and replay_ok < 0.5:
        return "false_safe"
    if predicted_ok < 0.5 and replay_ok >= 0.5:
        return "false_unsafe"
    return "true_unsafe"


def add_replay_feasibility_flags(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df.copy()

    df = detail_df.copy()
    pred = pd.to_numeric(df.get("predicted_within_limits"), errors="coerce").fillna(-1)
    replay = pd.to_numeric(df.get("replayed_within_limits"), errors="coerce").fillna(-1)
    df["predicted_within_limits"] = pred.astype(int)
    df["replayed_within_limits"] = replay.astype(int)
    df["feasibility_case"] = [
        classify_limit_case(float(p), float(r))
        for p, r in zip(df["predicted_within_limits"], df["replayed_within_limits"])
    ]

    replayed = pd.to_numeric(df.get("replayed_value"), errors="coerce")
    low = pd.to_numeric(df.get("limit_low"), errors="coerce")
    high = pd.to_numeric(df.get("limit_high"), errors="coerce")
    below = (low - replayed).clip(lower=0.0)
    above = (replayed - high).clip(lower=0.0)
    df["replay_violation_magnitude"] = np.nanmax(np.vstack([below.to_numpy(), above.to_numpy()]), axis=0)
    return df


def summarize_replay_feasibility(detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = add_replay_feasibility_flags(detail_df)
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty

    by_run = (
        df.groupby(
            ["run_id", "formulation_id", "formulation_name", "scenario_id", "scenario_name"],
            dropna=False,
        )
        .agg(
            metrics_checked=("metric_name", "count"),
            predicted_dynamic_safe_rate=("predicted_within_limits", lambda s: float(np.nanmean(np.asarray(s, dtype=float) >= 0.5))),
            replay_dynamic_safe_rate=("replayed_within_limits", lambda s: float(np.nanmean(np.asarray(s, dtype=float) >= 0.5))),
            predicted_dynamic_safe_all=("predicted_within_limits", lambda s: int(np.all(np.asarray(s, dtype=float) >= 0.5))),
            replay_dynamic_safe_all=("replayed_within_limits", lambda s: int(np.all(np.asarray(s, dtype=float) >= 0.5))),
            false_safe_count=("feasibility_case", lambda s: int(np.sum(pd.Series(s) == "false_safe"))),
            false_unsafe_count=("feasibility_case", lambda s: int(np.sum(pd.Series(s) == "false_unsafe"))),
            true_unsafe_count=("feasibility_case", lambda s: int(np.sum(pd.Series(s) == "true_unsafe"))),
            max_replay_violation_magnitude=("replay_violation_magnitude", "max"),
        )
        .reset_index()
    )

    by_formulation = (
        df.groupby(["formulation_id", "formulation_name"], dropna=False)
        .agg(
            metrics_checked=("metric_name", "count"),
            replay_dynamic_safe_rate=("replayed_within_limits", lambda s: float(np.nanmean(np.asarray(s, dtype=float) >= 0.5))),
            false_safe_count=("feasibility_case", lambda s: int(np.sum(pd.Series(s) == "false_safe"))),
            false_unsafe_count=("feasibility_case", lambda s: int(np.sum(pd.Series(s) == "false_unsafe"))),
            true_unsafe_count=("feasibility_case", lambda s: int(np.sum(pd.Series(s) == "true_unsafe"))),
            max_replay_violation_magnitude=("replay_violation_magnitude", "max"),
        )
        .reset_index()
    )

    return by_run, by_formulation


def add_line_security_flags(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df.copy()

    df = run_df.copy()
    base_viol = pd.to_numeric(df.get("base_n_violations"), errors="coerce").fillna(0)
    n1_viol = pd.to_numeric(df.get("n1_total_line_violations"), errors="coerce").fillna(0)
    use_n1 = pd.to_numeric(df.get("use_n1"), errors="coerce").fillna(0)
    df["line_security_safe_base"] = (base_viol <= 0).astype(int)
    df["line_security_safe_n1"] = np.where(use_n1 >= 0.5, (n1_viol <= 0).astype(int), 1)
    df["line_security_safe_all"] = ((df["line_security_safe_base"] >= 0.5) & (df["line_security_safe_n1"] >= 0.5)).astype(int)
    df["line_security_max_violation_pct"] = np.nanmax(
        np.vstack(
            [
                pd.to_numeric(df.get("base_max_violation_pct"), errors="coerce").fillna(0.0).to_numpy(),
                pd.to_numeric(df.get("n1_max_violation_pct"), errors="coerce").fillna(0.0).to_numpy(),
            ]
        ),
        axis=0,
    )
    return df
