from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

try:
    from .config_analysis import (
        CASE_WORKBOOK_PATH,
        DEFAULT_SCENARIO_CSV_NAME,
        FEATURE_NAME_CONFIG,
        FIGURES_DIR,
        LOCAL_COMBINED_RESULTS_CSV,
        LOCAL_SCENARIO_RESULTS_DIR,
        PRIMARY_COMBINED_RESULTS_CSV,
        PRIMARY_SCENARIO_RESULTS_DIR,
        REPO_ROOT,
        SCENARIO_LABELS,
        SCENARIO_ORDER,
        SYSTEM_BASE_MVA,
        TABLES_DIR,
        TRAIN_SWEEP_CONFIG,
    )
except ImportError:
    from config_analysis import (  # type: ignore
        CASE_WORKBOOK_PATH,
        DEFAULT_SCENARIO_CSV_NAME,
        FEATURE_NAME_CONFIG,
        FIGURES_DIR,
        LOCAL_COMBINED_RESULTS_CSV,
        LOCAL_SCENARIO_RESULTS_DIR,
        PRIMARY_COMBINED_RESULTS_CSV,
        PRIMARY_SCENARIO_RESULTS_DIR,
        REPO_ROOT,
        SCENARIO_LABELS,
        SCENARIO_ORDER,
        SYSTEM_BASE_MVA,
        TABLES_DIR,
        TRAIN_SWEEP_CONFIG,
    )


class DatasetNotReadyError(RuntimeError):
    """Raised when the expected generated dataset is not available yet."""


@dataclass
class DatasetBundle:
    retained: pd.DataFrame
    attempted: Optional[pd.DataFrame]
    source_label: str
    source_paths: List[Path]
    train_cfg_path: Optional[Path]
    train_cfg: Dict
    scenario_family_col: str = "scenario_family"


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_feature_name_schema(path: Optional[Path] = None) -> Dict:
    schema_path = Path(path) if path is not None else FEATURE_NAME_CONFIG
    if not schema_path.exists():
        raise DatasetNotReadyError(
            "Feature-name config is missing.\n"
            f"Expected: {schema_path}\n"
            "Generate or restore the repository config before running the analysis notebooks."
        )
    return load_yaml(schema_path)


def load_regcv1_device_bases(case_workbook_path: Optional[Path] = None) -> Dict[int, float]:
    workbook_path = Path(case_workbook_path) if case_workbook_path is not None else CASE_WORKBOOK_PATH
    if not workbook_path.exists():
        raise DatasetNotReadyError(
            "REGCV1 workbook is missing.\n"
            f"Expected: {workbook_path}\n"
            "Restore the ANDES case workbook before running the analysis notebooks."
        )
    regcv1 = pd.read_excel(workbook_path, sheet_name="REGCV1")
    if "Sn" not in regcv1.columns or "idx" not in regcv1.columns:
        raise DatasetNotReadyError(
            "The REGCV1 sheet does not contain the expected `idx` and `Sn` columns.\n"
            f"Workbook: {workbook_path}"
        )

    device_bases: Dict[int, float] = {}
    for _, row in regcv1.iterrows():
        idx = str(row["idx"])
        if not idx.startswith("REGCV1_"):
            continue
        try:
            unit_id = int(idx.split("_")[-1])
        except ValueError:
            continue
        device_bases[unit_id] = float(row["Sn"])
    return device_bases


def add_ibr_device_base_columns(
    df: pd.DataFrame,
    *,
    system_base_mva: float = SYSTEM_BASE_MVA,
    case_workbook_path: Optional[Path] = None,
) -> pd.DataFrame:
    out = df.copy()
    device_bases = load_regcv1_device_bases(case_workbook_path)
    for unit_id, sn_mva in device_bases.items():
        if not np.isfinite(sn_mva) or sn_mva <= 0.0:
            continue
        factor = float(system_base_mva) / float(sn_mva)
        column_map = {
            f"P_REGCV1_{unit_id}": f"P_REGCV1_IBRBASE_{unit_id}",
            f"P_REGCV1_RESERVE_{unit_id}": f"P_REGCV1_RESERVE_IBRBASE_{unit_id}",
            f"Delta_P_IBR_{unit_id}": f"Delta_P_IBR_IBRBASE_{unit_id}",
            f"Delta_P_IBR_abs_{unit_id}": f"Delta_P_IBR_abs_IBRBASE_{unit_id}",
        }
        for old_col, new_col in column_map.items():
            if old_col not in out.columns:
                continue
            values = pd.to_numeric(out[old_col], errors="coerce")
            out[new_col] = values * factor
    return out


def discover_training_config_candidates() -> List[Path]:
    candidates = [TRAIN_SWEEP_CONFIG]
    for path in sorted((REPO_ROOT / "__results").glob("**/train_sweep.yaml")):
        candidates.append(path)
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        unique.append(path)
    return unique


def discover_training_csv_candidates() -> List[Tuple[Path, Path]]:
    out: List[Tuple[Path, Path]] = []
    for cfg_path in discover_training_config_candidates():
        cfg = load_yaml(cfg_path)
        csv_path = cfg.get("data", {}).get("csv_path")
        if not csv_path:
            continue
        resolved = (cfg_path.parent / csv_path).resolve() if not Path(csv_path).is_absolute() else Path(csv_path)
        if not resolved.exists():
            resolved = (REPO_ROOT / csv_path).resolve()
        out.append((cfg_path, resolved))
    return out


def discover_combined_dataset_csv_candidates() -> List[Path]:
    candidates = [PRIMARY_COMBINED_RESULTS_CSV, LOCAL_COMBINED_RESULTS_CSV]
    unique: List[Path] = []
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        unique.append(path.resolve())
    return unique


def discover_scenario_csvs(
    roots: Optional[Sequence[Path]] = None,
    csv_name: str = DEFAULT_SCENARIO_CSV_NAME,
) -> List[Tuple[str, Path]]:
    search_roots = list(roots or [PRIMARY_SCENARIO_RESULTS_DIR, LOCAL_SCENARIO_RESULTS_DIR])
    found: List[Tuple[str, Path]] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob(f"*/{csv_name}")):
            family = path.parent.name
            key = (family, path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append((family, path.resolve()))
    return found


def _load_scenario_dataframe(path: Path, scenario_family: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["scenario_family"] = scenario_family
    df["source_csv"] = str(path)
    return df


def load_scenario_dataset(
    scenario_csvs: Optional[Sequence[Tuple[str, Path]]] = None,
) -> pd.DataFrame:
    csvs = list(scenario_csvs or discover_scenario_csvs())
    if not csvs:
        raise DatasetNotReadyError(
            "No generated scenario CSVs were found.\n"
            f"Expected files like: {PRIMARY_SCENARIO_RESULTS_DIR / '*' / DEFAULT_SCENARIO_CSV_NAME}\n"
            "Generate the thesis data results first, then rerun the notebook."
        )
    frames = [_load_scenario_dataframe(path, family) for family, path in csvs]
    return pd.concat(frames, axis=0, ignore_index=True, sort=False)


def load_dataset_bundle(
    *,
    training_csv_override: Optional[Path] = None,
    scenario_csvs: Optional[Sequence[Tuple[str, Path]]] = None,
    retain_success_only: bool = True,
    prefer_training_csv: bool = False,
    prefer_scenario_csvs: bool = False,
) -> DatasetBundle:
    training_candidates = discover_training_csv_candidates()
    default_cfg_path: Optional[Path]
    if TRAIN_SWEEP_CONFIG.exists():
        default_cfg_path = TRAIN_SWEEP_CONFIG
    else:
        default_cfg_path = next(iter(discover_training_config_candidates()), None)
    default_train_cfg = load_yaml(default_cfg_path) if default_cfg_path is not None else {}

    selected_training_csv: Optional[Path] = None
    selected_training_cfg_path: Optional[Path] = None
    selected_training_cfg: Dict = {}
    if training_csv_override is not None:
        selected_training_csv = Path(training_csv_override)
        selected_training_cfg_path = None
        selected_training_cfg = {}
    else:
        combined_candidates = discover_combined_dataset_csv_candidates()
        if combined_candidates:
            selected_training_csv = combined_candidates[0]
            selected_training_cfg_path = default_cfg_path
            selected_training_cfg = default_train_cfg
        else:
            for cfg_path, csv_path in training_candidates:
                if csv_path.exists():
                    selected_training_csv = csv_path
                    selected_training_cfg_path = cfg_path
                    selected_training_cfg = load_yaml(cfg_path) if cfg_path is not None else {}
                    break

    if selected_training_csv is not None and (training_csv_override is not None or prefer_training_csv):
        attempted = pd.read_csv(selected_training_csv, low_memory=False)
        retained = filter_retained_dataset(attempted) if retain_success_only else attempted.copy()
        return DatasetBundle(
            retained=retained,
            attempted=attempted,
            source_label="training_csv",
            source_paths=[selected_training_csv],
            train_cfg_path=selected_training_cfg_path,
            train_cfg=selected_training_cfg,
        )

    if prefer_scenario_csvs:
        attempted = load_scenario_dataset(scenario_csvs=scenario_csvs)
        retained = filter_retained_dataset(attempted) if retain_success_only else attempted.copy()
        if retained.empty:
            raise DatasetNotReadyError(
                "Scenario CSVs were found, but no retained rows are available after filtering.\n"
                "Check the `success` column in the generated outputs and rerun the data-generation jobs if needed."
            )
        return DatasetBundle(
            retained=retained,
            attempted=attempted,
            source_label="combined_scenarios",
            source_paths=[Path(value) for value in retained["source_csv"].dropna().unique()],
            train_cfg_path=default_cfg_path,
            train_cfg=default_train_cfg,
        )

    if training_csv_override is None:
        combined_candidates = discover_combined_dataset_csv_candidates()
        if combined_candidates:
            combined_csv = combined_candidates[0]
            attempted = pd.read_csv(combined_csv, low_memory=False)
            retained = filter_retained_dataset(attempted) if retain_success_only else attempted.copy()
            return DatasetBundle(
                retained=retained,
                attempted=attempted,
                source_label="combined_results_csv",
                source_paths=[combined_csv],
                train_cfg_path=default_cfg_path,
                train_cfg=default_train_cfg,
            )

    scenario_error: Optional[DatasetNotReadyError] = None
    try:
        attempted = load_scenario_dataset(scenario_csvs=scenario_csvs)
    except DatasetNotReadyError as exc:
        scenario_error = exc
        attempted = None

    if attempted is not None:
        retained = filter_retained_dataset(attempted) if retain_success_only else attempted.copy()
        if retained.empty:
            raise DatasetNotReadyError(
                "Scenario CSVs were found, but no retained rows are available after filtering.\n"
                "Check the `success` column in the generated outputs and rerun the data-generation jobs if needed."
            )
        return DatasetBundle(
            retained=retained,
            attempted=attempted,
            source_label="combined_scenarios",
            source_paths=[Path(value) for value in retained["source_csv"].dropna().unique()],
            train_cfg_path=default_cfg_path,
            train_cfg=default_train_cfg,
        )

    if selected_training_csv is not None:
        attempted = pd.read_csv(selected_training_csv, low_memory=False)
        retained = filter_retained_dataset(attempted) if retain_success_only else attempted.copy()
        return DatasetBundle(
            retained=retained,
            attempted=attempted,
            source_label="training_csv",
            source_paths=[selected_training_csv],
            train_cfg_path=selected_training_cfg_path,
            train_cfg=selected_training_cfg,
        )

    if scenario_error is not None:
        raise scenario_error

    raise DatasetNotReadyError(
        "No generated dataset source could be located.\n"
        f"Optional retained training CSV configured from: {TRAIN_SWEEP_CONFIG}"
    )


def filter_retained_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if "success" not in df.columns:
        return df.copy()
    success = df["success"]
    if success.dtype == bool:
        mask = success
    else:
        as_text = success.astype(str).str.lower()
        mask = success.eq(1) | success.eq(1.0) | as_text.isin({"true", "1", "yes"})
    return df.loc[mask].copy()


def infer_target_columns(df: pd.DataFrame, train_cfg: Dict) -> List[str]:
    configured = list(train_cfg.get("data", {}).get("target_cols") or [])
    if configured:
        return [col for col in configured if col in df.columns]

    schema = load_feature_name_schema()
    target_cols: List[str] = []
    target_cols.extend(schema.get("y", {}).get("coi_fields", []) or [])
    for prefix_key in ("delta_p_ibr", "delta_p_ibr_abs", "bus_freq_max_abs_dev", "bus_v_max_abs_dev", "bus_rocof_max_abs"):
        prefix = schema.get("y", {}).get("prefixes", {}).get(prefix_key)
        if prefix:
            target_cols.extend([col for col in df.columns if str(col).startswith(prefix)])
    return [col for col in target_cols if col in df.columns]


def infer_feature_columns(df: pd.DataFrame, train_cfg: Dict, target_cols: Sequence[str]) -> List[str]:
    data_cfg = train_cfg.get("data", {}) or {}
    drop_cols = [str(value) for value in data_cfg.get("drop_cols", []) or []]
    drop_prefixes = [str(value) for value in data_cfg.get("drop_prefixes", []) or []]
    blocked = set(target_cols) | set(drop_cols)
    return [
        col
        for col in df.columns
        if col not in blocked and not any(str(col).startswith(prefix) for prefix in drop_prefixes)
    ]


def summarize_dataset_counts(bundle: DatasetBundle) -> pd.DataFrame:
    attempted = bundle.attempted if bundle.attempted is not None else bundle.retained
    retained = bundle.retained
    attempted_count = int(len(attempted))
    retained_count = int(len(retained))
    retention_rate = retained_count / attempted_count if attempted_count else np.nan
    target_cols = infer_target_columns(retained, bundle.train_cfg)
    feature_cols = infer_feature_columns(retained, bundle.train_cfg, target_cols)

    rows = [
        {"metric": "attempted_rows", "value": attempted_count},
        {"metric": "retained_rows", "value": retained_count},
        {"metric": "retention_rate", "value": retention_rate},
        {"metric": "n_feature_columns", "value": len(feature_cols)},
        {"metric": "n_target_columns", "value": len(target_cols)},
        {"metric": "n_total_columns", "value": retained.shape[1]},
        {"metric": "data_source", "value": bundle.source_label},
    ]
    return pd.DataFrame(rows)


def scenario_counts(df: pd.DataFrame, scenario_family_col: str = "scenario_family") -> pd.DataFrame:
    if scenario_family_col in df.columns:
        scenario_series = df[scenario_family_col].fillna("unknown").astype(str)
    else:
        scenario_series = infer_scenario_family(df)

    counts = (
        scenario_series
        .value_counts(dropna=False)
        .rename_axis("scenario_family")
        .reset_index(name="count")
        .reset_index(drop=True)
    )
    order_lookup = {name: idx for idx, name in enumerate(SCENARIO_ORDER)}
    counts["scenario_label"] = counts["scenario_family"].map(SCENARIO_LABELS).fillna(counts["scenario_family"])
    counts["_scenario_order"] = counts["scenario_family"].map(order_lookup).fillna(len(order_lookup))
    counts = counts.sort_values(["_scenario_order", "scenario_label"]).reset_index(drop=True)
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts.drop(columns="_scenario_order")


def split_by_scenario(df: pd.DataFrame, scenario_family_col: str = "scenario_family") -> Dict[str, pd.DataFrame]:
    scenario_series = infer_scenario_family(df) if scenario_family_col not in df.columns else df[scenario_family_col]
    scenario_series = scenario_series.fillna("unknown").astype(str)
    frames: Dict[str, pd.DataFrame] = {}
    for scenario_family in SCENARIO_ORDER:
        mask = scenario_series == scenario_family
        if mask.any():
            frames[scenario_family] = df.loc[mask].copy()
    for scenario_family in sorted(set(scenario_series) - set(frames.keys())):
        mask = scenario_series == scenario_family
        if mask.any():
            frames[scenario_family] = df.loc[mask].copy()
    return frames


def split_dataframe(
    df: pd.DataFrame,
    split_cfg: Optional[Dict] = None,
) -> Dict[str, pd.DataFrame]:
    cfg = split_cfg or {}
    test_size = float(cfg.get("test_size", 0.3))
    val_fraction = float(cfg.get("val_fraction", 0.5))
    random_state = int(cfg.get("random_state", 42))

    idx = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx, test_size=test_size, random_state=random_state)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=val_fraction, random_state=random_state)
    return {
        "train": df.iloc[idx_train].copy(),
        "val": df.iloc[idx_val].copy(),
        "test": df.iloc[idx_test].copy(),
    }


def split_summary_table(
    split_frames: Dict[str, pd.DataFrame],
    columns: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    quantiles = [0.05, 0.5, 0.95]
    for split_name, frame in split_frames.items():
        for col in columns:
            if col not in frame.columns:
                continue
            series = pd.to_numeric(frame[col], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "split": split_name,
                    "column": col,
                    "count": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "q05": float(series.quantile(quantiles[0])),
                    "median": float(series.quantile(quantiles[1])),
                    "q95": float(series.quantile(quantiles[2])),
                }
            )
    return pd.DataFrame(rows)


def numeric_summary_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        rows.append(
            {
                "column": col,
                "count": int(series.shape[0]),
                "min": float(series.min()),
                "q05": float(series.quantile(0.05)),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "median": float(series.median()),
                "q95": float(series.quantile(0.95)),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def pick_existing_columns(df: pd.DataFrame, preferred: Sequence[str], *, allow_prefix: bool = False) -> List[str]:
    out: List[str] = []
    for name in preferred:
        if name in df.columns:
            out.append(name)
        elif allow_prefix:
            out.extend([col for col in df.columns if str(col).startswith(name)])
    # preserve order and uniqueness
    seen = set()
    deduped: List[str] = []
    for col in out:
        if col in seen:
            continue
        seen.add(col)
        deduped.append(col)
    return deduped


def infer_scenario_family(df: pd.DataFrame) -> pd.Series:
    if "scenario_family" in df.columns:
        return df["scenario_family"].fillna("unknown").astype(str)
    if "cont_type" in df.columns:
        return df["cont_type"].fillna("unknown").astype(str)
    return pd.Series(["unknown"] * len(df), index=df.index, name="scenario_family")


def rank_spearman_associations(
    df: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        subset = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if subset.shape[0] < 3:
            continue
        rho = subset[x_col].corr(subset[y_col], method="spearman")
        rows.append(
            {
                "x": x_col,
                "y": y_col,
                "n": int(subset.shape[0]),
                "spearman_rho": float(rho),
                "abs_spearman_rho": abs(float(rho)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_spearman_rho", "x", "y"], ascending=[False, True, True]).reset_index(drop=True)
    return out


def write_table(df: pd.DataFrame, filename: str) -> Path:
    ensure_output_dirs()
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    return path


def latex_rows(df: pd.DataFrame, *, columns: Sequence[str], float_fmt: str = ".3f") -> str:
    lines: List[str] = []
    for _, row in df.iterrows():
        parts: List[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                parts.append(format(float(value), float_fmt))
            else:
                parts.append(str(value))
        lines.append(" & ".join(parts) + r" \\")
    return "\n".join(lines)
