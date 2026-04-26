import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _univariate_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, _r2(y, X @ beta)


def _bivariate_fit(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    X = np.column_stack([np.ones_like(x1), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, _r2(y, X @ beta)


def _standardized_coefficients(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x1s = (x1 - np.mean(x1)) / np.std(x1)
    x2s = (x2 - np.mean(x2)) / np.std(x2)
    ys = (y - np.mean(y)) / np.std(y)
    beta, _ = _bivariate_fit(x1s, x2s, ys)
    return float(beta[1]), float(beta[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize simple physical proxies for REGCV1 power response.")
    parser.add_argument("--csv", required=True, help="Dataset CSV used for the analysis.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV for per-IBR rows.")
    parser.add_argument("--summary-csv", required=True, help="Destination CSV for mean/median summary.")
    args = parser.parse_args()

    usecols = ["rocof_COI", "dev_COI"]
    for idx in range(1, 5):
        usecols.extend([f"M_{idx}", f"D_{idx}", f"Delta_P_IBR_{idx}"])

    df = pd.read_csv(args.csv, usecols=usecols)
    rows = []
    for idx in range(1, 5):
        y = df[f"Delta_P_IBR_{idx}"].to_numpy(dtype=float)
        inertia_proxy = -(df[f"M_{idx}"] * df["rocof_COI"]).to_numpy(dtype=float)
        damping_proxy = -(df[f"D_{idx}"] * df["dev_COI"]).to_numpy(dtype=float)
        combined_proxy = inertia_proxy + damping_proxy

        beta_inertia, r2_inertia = _univariate_fit(inertia_proxy, y)
        beta_damping, r2_damping = _univariate_fit(damping_proxy, y)
        beta_two_term, r2_two_term = _bivariate_fit(inertia_proxy, damping_proxy, y)
        std_beta_inertia, std_beta_damping = _standardized_coefficients(inertia_proxy, damping_proxy, y)

        rows.append(
            {
                "ibr": f"IBR_{idx}",
                "corr_inertia_proxy": float(np.corrcoef(y, inertia_proxy)[0, 1]),
                "corr_damping_proxy": float(np.corrcoef(y, damping_proxy)[0, 1]),
                "corr_combined_proxy": float(np.corrcoef(y, combined_proxy)[0, 1]),
                "r2_inertia_proxy": float(r2_inertia),
                "r2_damping_proxy": float(r2_damping),
                "r2_two_term_proxy": float(r2_two_term),
                "beta_inertia_intercept": float(beta_inertia[0]),
                "beta_inertia_slope": float(beta_inertia[1]),
                "beta_damping_intercept": float(beta_damping[0]),
                "beta_damping_slope": float(beta_damping[1]),
                "beta_two_term_intercept": float(beta_two_term[0]),
                "beta_two_term_inertia": float(beta_two_term[1]),
                "beta_two_term_damping": float(beta_two_term[2]),
                "beta_two_term_inertia_std": float(std_beta_inertia),
                "beta_two_term_damping_std": float(std_beta_damping),
            }
        )

    out_df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    summary_rows = []
    for label, reducer in (("mean", pd.Series.mean), ("median", pd.Series.median)):
        summary = {"statistic": label}
        numeric = out_df.drop(columns=["ibr"])
        reduced = numeric.apply(reducer, axis=0)
        summary.update({col: float(reduced[col]) for col in numeric.columns})
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)

    print(f"Saved REGCV1 proxy analysis to {output_csv} and {args.summary_csv}")


if __name__ == "__main__":
    main()
