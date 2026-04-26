from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = REPO_ROOT / "results" / "thesis_data_generation_results"
NOTEBOOKS_DIR = ANALYSIS_ROOT / "notebooks"
SRC_DIR = ANALYSIS_ROOT / "src"
OUTPUTS_DIR = ANALYSIS_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

PRIMARY_SCENARIO_RESULTS_DIR = ANALYSIS_ROOT / "results"
LOCAL_SCENARIO_RESULTS_DIR = REPO_ROOT / "__results" / "thesis_data_generation_results" / "results"
PRIMARY_COMBINED_RESULTS_CSV = PRIMARY_SCENARIO_RESULTS_DIR / "simulation_results.csv"
LOCAL_COMBINED_RESULTS_CSV = LOCAL_SCENARIO_RESULTS_DIR / "simulation_results.csv"

FEATURE_NAME_CONFIG = REPO_ROOT / "configs" / "data_generation_feature_names.yaml"
TRAIN_SWEEP_CONFIG = REPO_ROOT / "models" / "train_sweep.yaml"
CASE_WORKBOOK_PATH = REPO_ROOT / "data_generation" / "andes_cases" / "ieee39_full_ibrs.xlsx"
SYSTEM_BASE_MVA = 100.0

SCENARIO_CONFIG_DIR = ANALYSIS_ROOT / "configs"
LOCAL_SCENARIO_CONFIG_DIR = REPO_ROOT / "__results" / "configs"

DEFAULT_SCENARIO_CSV_NAME = "simulation_results.csv"

SCENARIO_ORDER = [
    "no_mismatch",
    "load_mismatch_only",
    "zone_based_load_mismatch",
    "line_outages_only",
    "line_outages_plus_global_load_mismatch",
    "line_outages_plus_zone_based_load_mismatch",
    "load",
    "line",
    "line_plus_load",
]

SCENARIO_LABELS = {
    "no_mismatch": "No mismatch reference",
    "load_mismatch_only": "Global load mismatch",
    "zone_based_load_mismatch": "Zone-based mismatch",
    "line_outages_only": "Line outages",
    "line_outages_plus_global_load_mismatch": "Line outages + global mismatch",
    "line_outages_plus_zone_based_load_mismatch": "Line outages + zone mismatch",
    "load": "Load mismatch",
    "line": "Line outages",
    "line_plus_load": "Line outages + load mismatch",
}

SCENARIO_COLORS = {
    "no_mismatch": "#9a8c7c",
    "load_mismatch_only": "#c65d3b",
    "zone_based_load_mismatch": "#4f7c5d",
    "line_outages_only": "#2f6690",
    "line_outages_plus_global_load_mismatch": "#805e73",
    "line_outages_plus_zone_based_load_mismatch": "#d1a054",
    "load": "#c65d3b",
    "line": "#2f6690",
    "line_plus_load": "#805e73",
}

COLUMN_LABELS = {
    "base_load_scale": "Base-load scale [-]",
    "load_step_scale": "Load-step scale [-]",
    "M_agg": "Aggregated virtual inertia M [-]",
    "D_agg": "Aggregated damping D [-]",
    "reserve_p_total_prefault": "Prefault active reserve [p.u.]",
    "reserve_p_ibr": "IBR active reserve [p.u.]",
    "reserve_p_genrou": "Synchronous active reserve [p.u.]",
    "P_REGCV1_SHARE": "IBR share of dispatch [-]",
    "rocof_COI": "COI RoCoF [Hz/s]",
    "rocof_abs_COI": "Absolute COI RoCoF [Hz/s]",
    "dev_COI": "Signed COI frequency deviation [Hz]",
    "dev_abs_COI": "Absolute COI frequency deviation [Hz]",
    "f_min_COI": "COI frequency nadir [Hz]",
    "f_max_COI": "COI frequency zenith [Hz]",
    "Delta_P_IBR_abs_1": "IBR 1 max |ΔP| [p.u. on 100 MVA base]",
    "Delta_P_IBR_abs_2": "IBR 2 max |ΔP| [p.u. on 100 MVA base]",
    "Delta_P_IBR_abs_3": "IBR 3 max |ΔP| [p.u. on 100 MVA base]",
    "Delta_P_IBR_abs_4": "IBR 4 max |ΔP| [p.u. on 100 MVA base]",
    "P_REGCV1_IBRBASE_1": "IBR 1 prefault active power [p.u. on IBR base]",
    "P_REGCV1_IBRBASE_2": "IBR 2 prefault active power [p.u. on IBR base]",
    "P_REGCV1_IBRBASE_3": "IBR 3 prefault active power [p.u. on IBR base]",
    "P_REGCV1_IBRBASE_4": "IBR 4 prefault active power [p.u. on IBR base]",
    "P_REGCV1_RESERVE_IBRBASE_1": "IBR 1 reserve [p.u. on IBR base]",
    "P_REGCV1_RESERVE_IBRBASE_2": "IBR 2 reserve [p.u. on IBR base]",
    "P_REGCV1_RESERVE_IBRBASE_3": "IBR 3 reserve [p.u. on IBR base]",
    "P_REGCV1_RESERVE_IBRBASE_4": "IBR 4 reserve [p.u. on IBR base]",
    "Delta_P_IBR_IBRBASE_1": "IBR 1 signed ΔP [p.u. on IBR base]",
    "Delta_P_IBR_IBRBASE_2": "IBR 2 signed ΔP [p.u. on IBR base]",
    "Delta_P_IBR_IBRBASE_3": "IBR 3 signed ΔP [p.u. on IBR base]",
    "Delta_P_IBR_IBRBASE_4": "IBR 4 signed ΔP [p.u. on IBR base]",
    "Delta_P_IBR_abs_IBRBASE_1": "IBR 1 max |ΔP| [p.u. on IBR base]",
    "Delta_P_IBR_abs_IBRBASE_2": "IBR 2 max |ΔP| [p.u. on IBR base]",
    "Delta_P_IBR_abs_IBRBASE_3": "IBR 3 max |ΔP| [p.u. on IBR base]",
    "Delta_P_IBR_abs_IBRBASE_4": "IBR 4 max |ΔP| [p.u. on IBR base]",
    "bus_rocof_max_abs_any": "Max bus RoCoF [Hz/s]",
    "bus_freq_max_abs_dev_any": "Max bus |Δf| [Hz]",
    "pre_fault_loading": "Pre-fault loading of outaged line [-]",
    "predicted_max_post_cont_loading_dc": "Predicted max post-contingency loading [-]",
}

SCENARIO_CONTROL_COLUMNS = {
    "load_mismatch_only": ["base_load_scale", "load_step_scale", "M_agg", "D_agg"],
    "line_outages_only": ["base_load_scale", "M_agg", "D_agg", "pre_fault_loading"],
    "line_outages_plus_global_load_mismatch": ["base_load_scale", "load_step_scale", "M_agg", "pre_fault_loading"],
    "zone_based_load_mismatch": ["base_load_scale", "load_step_scale", "M_agg", "D_agg"],
    "line_outages_plus_zone_based_load_mismatch": ["base_load_scale", "load_step_scale", "M_agg", "pre_fault_loading"],
    "load": ["base_load_scale", "load_step_scale", "M_agg", "D_agg"],
    "line": ["base_load_scale", "M_agg", "D_agg", "pre_fault_loading"],
    "line_plus_load": ["base_load_scale", "load_step_scale", "M_agg", "pre_fault_loading"],
}

SCENARIO_OUTPUT_COLUMNS = {
    "load_mismatch_only": ["rocof_abs_COI", "dev_abs_COI", "Delta_P_IBR_abs_IBRBASE_1", "bus_rocof_max_abs_any"],
    "line_outages_only": ["rocof_abs_COI", "dev_abs_COI", "predicted_max_post_cont_loading_dc", "bus_freq_max_abs_dev_any"],
    "line_outages_plus_global_load_mismatch": ["rocof_abs_COI", "dev_abs_COI", "Delta_P_IBR_abs_IBRBASE_1", "predicted_max_post_cont_loading_dc"],
    "zone_based_load_mismatch": ["rocof_abs_COI", "dev_abs_COI", "Delta_P_IBR_abs_IBRBASE_1", "bus_rocof_max_abs_any"],
    "line_outages_plus_zone_based_load_mismatch": ["rocof_abs_COI", "dev_abs_COI", "Delta_P_IBR_abs_IBRBASE_1", "predicted_max_post_cont_loading_dc"],
    "load": ["rocof_abs_COI", "dev_abs_COI", "Delta_P_IBR_abs_IBRBASE_1", "bus_rocof_max_abs_any"],
    "line": ["rocof_abs_COI", "dev_abs_COI", "predicted_max_post_cont_loading_dc", "bus_freq_max_abs_dev_any"],
    "line_plus_load": ["rocof_abs_COI", "dev_abs_COI", "Delta_P_IBR_abs_IBRBASE_1", "predicted_max_post_cont_loading_dc"],
}

PREFERRED_CONTROL_COLUMNS = [
    "base_load_scale",
    "load_step_scale",
    "M_agg",
    "D_agg",
    "reserve_p_total_prefault",
    "reserve_p_ibr",
    "reserve_p_genrou",
    "P_REGCV1_SHARE",
]

PREFERRED_OUTPUT_COLUMNS = [
    "rocof_COI",
    "rocof_abs_COI",
    "dev_COI",
    "dev_abs_COI",
    "f_min_COI",
    "f_max_COI",
    "Delta_P_IBR_abs_IBRBASE_1",
    "Delta_P_IBR_abs_IBRBASE_2",
    "Delta_P_IBR_abs_IBRBASE_3",
    "Delta_P_IBR_abs_IBRBASE_4",
    "bus_rocof_max_abs_any",
    "bus_freq_max_abs_dev_any",
]

PREFERRED_PHYSICAL_PAIRS = [
    ("M_agg", "rocof_abs_COI"),
    ("M_agg", "rocof_COI"),
    ("D_agg", "dev_abs_COI"),
    ("D_agg", "dev_COI"),
    ("load_step_scale", "rocof_abs_COI"),
    ("load_step_scale", "rocof_COI"),
    ("load_step_scale", "dev_abs_COI"),
    ("load_step_scale", "dev_COI"),
    ("base_load_scale", "rocof_abs_COI"),
    ("base_load_scale", "rocof_COI"),
    ("base_load_scale", "dev_abs_COI"),
    ("base_load_scale", "dev_COI"),
    ("pre_fault_loading", "rocof_abs_COI"),
    ("pre_fault_loading", "rocof_COI"),
    ("predicted_max_post_cont_loading_dc", "dev_abs_COI"),
    ("predicted_max_post_cont_loading_dc", "dev_COI"),
]

PREFERRED_SPLIT_COLUMNS = [
    "base_load_scale",
    "load_step_scale",
    "M_agg",
    "D_agg",
    "rocof_abs_COI",
    "rocof_COI",
    "dev_abs_COI",
    "dev_COI",
    "Delta_P_IBR_abs_IBRBASE_1",
    "Delta_P_IBR_IBRBASE_1",
    "Delta_P_IBR_abs_1",
    "Delta_P_IBR_1",
]
