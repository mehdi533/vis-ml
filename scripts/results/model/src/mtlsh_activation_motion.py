from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle, Wedge


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.data_utils import split_data
from models.models import create_model
from models.utils import build_model_kwargs, load_feature_name_registry, load_yaml, resolve_data_config


STYLE_PATH = REPO_ROOT / "configs/figure/thesis_plot_style.yaml"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs/model/mtlsh_activation_motion.yaml"


@dataclass
class PairSelection:
    sample_a: int
    sample_b: int
    score: float
    shared_flips: int
    focus_head_flips: int
    total_head_flips: int
    non_sched_diff: float
    sched_diff: float


@dataclass
class RunBundle:
    run_dir: Path
    model: torch.nn.Module
    feature_cols: list[str]
    target_cols: list[str]
    x_raw: np.ndarray
    x_norm: np.ndarray
    x_scaler: Any
    shared_weight: np.ndarray
    shared_bias: np.ndarray
    head_weights: list[np.ndarray]
    head_biases: list[np.ndarray]
    output_weights: list[np.ndarray]
    output_biases: list[np.ndarray]


def _load_style_config() -> dict[str, Any]:
    if not STYLE_PATH.exists():
        return {}
    return load_yaml(STYLE_PATH)


def _apply_thesis_style() -> dict[str, Any]:
    style_cfg = _load_style_config()
    rc_defaults = {
        "figure.figsize": (7.0, 4.4),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Serif",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.color": "#8a8179",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#4b433d",
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "regular",
        "axes.labelsize": 10.5,
        "axes.labelcolor": "#2b2826",
        "axes.labelpad": 4.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.borderaxespad": 0.6,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.8,
        "patch.edgecolor": "#2b2826",
        "patch.linewidth": 0.6,
    }
    rc_from_cfg = dict((style_cfg.get("style", {}) or {}).get("matplotlib", {}).get("rc_params", {}) or {})
    rc_defaults.update(rc_from_cfg)
    plt.rcParams.update(rc_defaults)
    return style_cfg


def _save_figure(fig: plt.Figure, stem: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _sched_feature_mask(feature_cols: list[str], mode: str = "all_sched") -> np.ndarray:
    mode_key = str(mode).strip().lower()
    if mode_key == "md_only":
        return np.array(
            [name in {"M_agg", "D_agg"} or name.startswith("M_") or name.startswith("D_") for name in feature_cols],
            dtype=bool,
        )
    if mode_key != "all_sched":
        raise ValueError(f"Unsupported sched_feature_mode '{mode}'.")
    return np.array(
        [
            name in {"M_agg", "D_agg"}
            or name.startswith("M_")
            or name.startswith("D_")
            or name.startswith("P_GENROU_RESERVE_")
            or name.startswith("P_REGCV1_RESERVE_")
            for name in feature_cols
        ],
        dtype=bool,
    )


def _collapsed_group_counts(feature_cols: list[str], sched_mask: np.ndarray) -> dict[str, int]:
    counts = {"x_op": 0, "x_cont": 0, "other fixed context": 0}
    for is_sched, name in zip(sched_mask, feature_cols):
        if is_sched:
            continue
        if name.startswith("P_BASE_") or name.startswith("Q_BASE_"):
            counts["x_op"] += 1
        elif name.startswith("DELTA_P_") or name.startswith("line_oh_uid_"):
            counts["x_cont"] += 1
        elif name in {
            "base_load_scale",
            "base_load_p_total",
            "base_load_q_total",
            "total_load_p_prefault",
            "total_load_q_prefault",
            "total_gen_p_prefault",
            "total_gen_q_prefault",
            "reserve_p_total_prefault",
            "reserve_q_total_prefault",
            "bus_v_min_prefault",
            "bus_v_max_prefault",
            "bus_v_mean_prefault",
            "bus_v_std_prefault",
            "bus_angle_min_prefault",
            "bus_angle_max_prefault",
            "bus_angle_spread_prefault",
            "n_buses",
            "n_lines",
            "n_pq_loads",
            "n_genrou",
            "n_regcv1",
        }:
            counts["x_op"] += 1
        elif name in {
            "load_step_scale",
            "load_step_time",
            "DELTA_PQ_tot",
            "line_rating",
            "line_fn",
            "line_Vn1",
            "line_Vn2",
            "line_r",
            "line_x",
            "line_b",
            "line_g",
            "line_b1",
            "line_g1",
            "line_b2",
            "line_g2",
            "line_trans",
            "line_tap",
            "line_phi",
            "line_x_over_r",
            "pre_fault_flow",
            "pre_fault_loading",
            "pre_p_from",
            "pre_p_to",
            "pre_loading_from",
            "pre_loading_to",
            "pre_flow_direction_p",
            "pre_v_from",
            "pre_v_to",
            "pre_theta_from",
            "pre_theta_to",
            "pre_delta_theta",
            "bus_degree_from",
            "bus_degree_to",
            "largest_component_fraction_after_trip",
            "system_max_loading_prefault",
            "system_mean_loading_prefault",
            "system_top5_loading_mean_prefault",
            "ptdf_l1_norm_outaged_line",
            "max_abs_lodf_row",
            "predicted_max_post_cont_loading_dc",
        }:
            counts["x_cont"] += 1
        else:
            counts["other fixed context"] += 1
    return counts


def _load_run_bundle(run_dir: Path) -> RunBundle:
    cfg = load_yaml(run_dir / "run_config.yaml")
    feature_cols = list(cfg.get("resolved", {}).get("feature_cols") or [])
    target_cols = list(cfg.get("resolved", {}).get("target_cols") or cfg.get("data", {}).get("target_cols") or [])
    if not feature_cols or not target_cols:
        raise ValueError("Run config is missing resolved feature/target columns.")

    feature_name_registry = load_feature_name_registry(cfg["data"].get("feature_names_path"))
    model, device = create_model(
        cfg["resolved"]["model_type"],
        in_dim=len(feature_cols),
        out_dim=len(target_cols),
        **build_model_kwargs(
            cfg.get("model", {}),
            feature_cols,
            train_cfg=cfg.get("training", {}),
            feature_name_registry=feature_name_registry,
        ),
    )
    state_path = run_dir / "mtlsh_state_dict_best.pt"
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    data_cfg = resolve_data_config(cfg["data"])
    usecols = list(dict.fromkeys(feature_cols + target_cols))
    df = pd.read_csv(REPO_ROOT / str(data_cfg["csv_path"]), usecols=usecols, low_memory=False)
    missing_features = [col for col in feature_cols if col not in df.columns]
    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_features or missing_targets:
        raise ValueError(
            f"Missing required columns in retained-run dataset. features={missing_features} targets={missing_targets}"
        )
    missing_fill_value = data_cfg.get("missing_fill_value")
    if missing_fill_value is not None:
        df[usecols] = df[usecols].fillna(float(missing_fill_value))
    x_raw = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    _, _, x_test, _, _, _ = split_data(
        x_raw,
        np.zeros((x_raw.shape[0], len(target_cols)), dtype=np.float32),
        test_size=float(cfg.get("split", {}).get("test_size", 0.3)),
        val_fraction=float(cfg.get("split", {}).get("val_fraction", 0.5)),
        random_state=int(cfg.get("split", {}).get("random_state", 42)),
    )
    x_scaler = joblib.load(run_dir / "x_scaler.pkl")
    x_norm = np.asarray(x_scaler.transform(x_test), dtype=np.float32)

    shared_layer = model.shared[0]
    shared_weight = shared_layer.weight.detach().cpu().numpy()
    shared_bias = shared_layer.bias.detach().cpu().numpy()
    head_weights: list[np.ndarray] = []
    head_biases: list[np.ndarray] = []
    output_weights: list[np.ndarray] = []
    output_biases: list[np.ndarray] = []
    for head in model.heads:
        head_weights.append(head[0].weight.detach().cpu().numpy())
        head_biases.append(head[0].bias.detach().cpu().numpy())
        output_weights.append(head[3].weight.detach().cpu().numpy())
        output_biases.append(head[3].bias.detach().cpu().numpy())

    return RunBundle(
        run_dir=run_dir,
        model=model,
        feature_cols=feature_cols,
        target_cols=target_cols,
        x_raw=np.asarray(x_test, dtype=np.float32),
        x_norm=x_norm,
        x_scaler=x_scaler,
        shared_weight=shared_weight,
        shared_bias=shared_bias,
        head_weights=head_weights,
        head_biases=head_biases,
        output_weights=output_weights,
        output_biases=output_biases,
    )


def _evaluate_batch(bundle: RunBundle, x_norm: np.ndarray) -> dict[str, np.ndarray]:
    shared_pre = np.asarray(x_norm @ bundle.shared_weight.T + bundle.shared_bias, dtype=np.float32)
    shared_act = np.maximum(shared_pre, 0.0)
    shared_mask = shared_pre > 0.0

    head_pre = []
    head_mask = []
    preds = []
    for head_weight, head_bias, out_weight, out_bias in zip(
        bundle.head_weights,
        bundle.head_biases,
        bundle.output_weights,
        bundle.output_biases,
    ):
        z = np.asarray(shared_act @ head_weight.T + head_bias, dtype=np.float32)
        h = np.maximum(z, 0.0)
        pred = np.asarray(h @ out_weight.T + out_bias, dtype=np.float32)
        head_pre.append(z)
        head_mask.append(z > 0.0)
        preds.append(pred[:, 0])

    return {
        "shared_pre": shared_pre,
        "shared_act": shared_act,
        "shared_mask": shared_mask,
        "head_pre": np.stack(head_pre, axis=1),
        "head_mask": np.stack(head_mask, axis=1),
        "pred": np.stack(preds, axis=1),
    }


def _select_pair(
    x_norm: np.ndarray,
    shared_mask: np.ndarray,
    head_mask: np.ndarray,
    sched_mask: np.ndarray,
    focus_head_idx: int,
    candidate_search_size: int,
    weights: dict[str, float],
    min_shared_flip_target: int,
    min_focus_head_flip_target: int,
) -> PairSelection:
    n = min(int(candidate_search_size), int(x_norm.shape[0]))
    if n < 2:
        raise ValueError("Need at least two test samples for pair selection.")
    search_sizes = [n]
    if n < x_norm.shape[0]:
        search_sizes.append(int(x_norm.shape[0]))

    sched_idx = np.flatnonzero(sched_mask)
    non_sched_idx = np.flatnonzero(~sched_mask)
    best_any: PairSelection | None = None

    for search_round, size in enumerate(search_sizes):
        subset = x_norm[:size]
        subset_mask = shared_mask[:size]
        subset_head_mask = head_mask[:size]
        best_target: PairSelection | None = None
        best_score = -np.inf
        best_fallback: PairSelection | None = None
        best_fallback_score = -np.inf
        if search_round == 0:
            anchor_indices = np.arange(size, dtype=int)
            print(f"Pair search: exact scan over first {size} test samples.", flush=True)
        else:
            anchor_count = min(256, size)
            anchor_indices = np.linspace(0, size - 1, num=anchor_count, dtype=int)
            anchor_indices = np.unique(anchor_indices)
            print(
                f"Pair search: widened fallback over {size} test samples using {anchor_indices.size} anchor rows.",
                flush=True,
            )

        for i in anchor_indices:
            non_sched_diff = np.abs(subset[:, non_sched_idx] - subset[i, non_sched_idx]).mean(axis=1)
            sched_diff = np.abs(subset[:, sched_idx] - subset[i, sched_idx]).mean(axis=1)
            shared_flips = np.not_equal(subset_mask, subset_mask[i]).sum(axis=1)
            focus_head_flips = np.not_equal(subset_head_mask[:, focus_head_idx], subset_head_mask[i, focus_head_idx]).sum(axis=1)
            total_head_flips = np.not_equal(subset_head_mask, subset_head_mask[i]).sum(axis=(1, 2))
            score = (
                float(weights.get("shared_flips", 1.0)) * shared_flips
                + float(weights.get("focus_head_flips", 0.0)) * focus_head_flips
                + float(weights.get("total_head_flips", 0.0)) * total_head_flips
                + float(weights.get("non_sched_diff", -8.0)) * non_sched_diff
                + float(weights.get("sched_diff", 2.5)) * sched_diff
            )
            score[i] = -np.inf
            j = int(np.argmax(score))
            candidate = PairSelection(
                sample_a=i,
                sample_b=j,
                score=float(score[j]),
                shared_flips=int(shared_flips[j]),
                focus_head_flips=int(focus_head_flips[j]),
                total_head_flips=int(total_head_flips[j]),
                non_sched_diff=float(non_sched_diff[j]),
                sched_diff=float(sched_diff[j]),
            )
            if candidate.score > best_fallback_score:
                best_fallback = candidate
                best_fallback_score = candidate.score
            if (
                candidate.shared_flips >= int(min_shared_flip_target)
                and candidate.focus_head_flips >= int(min_focus_head_flip_target)
                and candidate.score > best_score
            ):
                best_target = candidate
                best_score = candidate.score

        if best_target is not None:
            return best_target
        if best_any is None and best_fallback is not None:
            best_any = best_fallback

    if best_any is None:
        raise RuntimeError("Failed to select a matched sample pair.")
    if (
        best_any.shared_flips <= 0
        or best_any.focus_head_flips <= 0
        or best_any.sched_diff <= 0.0
        or best_any.sample_a == best_any.sample_b
    ):
        raise RuntimeError("Pair-selection fallback did not produce a valid schedule-varying activation change.")
    return best_any


def _top_changed_sched_rows(
    feature_cols: list[str],
    x_raw: np.ndarray,
    x_norm: np.ndarray,
    sched_mask: np.ndarray,
    pair: PairSelection,
) -> pd.DataFrame:
    sched_idx = np.flatnonzero(sched_mask)
    rows = []
    for idx in sched_idx:
        rows.append(
            {
                "feature": feature_cols[int(idx)],
                "sample_a_raw": float(x_raw[pair.sample_a, idx]),
                "sample_b_raw": float(x_raw[pair.sample_b, idx]),
                "sample_a_norm": float(x_norm[pair.sample_a, idx]),
                "sample_b_norm": float(x_norm[pair.sample_b, idx]),
                "abs_norm_delta": float(abs(x_norm[pair.sample_b, idx] - x_norm[pair.sample_a, idx])),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_norm_delta", ascending=False).reset_index(drop=True)


def _shared_activation_table(eval_data: dict[str, np.ndarray], pair: PairSelection) -> pd.DataFrame:
    rows = []
    for tag, sample_idx in [("sample_a", pair.sample_a), ("sample_b", pair.sample_b)]:
        for neuron_idx in range(eval_data["shared_pre"].shape[1]):
            rows.append(
                {
                    "sample": tag,
                    "sample_idx": int(sample_idx),
                    "shared_neuron": int(neuron_idx),
                    "pre_activation": float(eval_data["shared_pre"][sample_idx, neuron_idx]),
                    "active": bool(eval_data["shared_mask"][sample_idx, neuron_idx]),
                }
            )
    return pd.DataFrame(rows)


def _head_activation_table(eval_data: dict[str, np.ndarray], pair: PairSelection, target_cols: list[str]) -> pd.DataFrame:
    rows = []
    for tag, sample_idx in [("sample_a", pair.sample_a), ("sample_b", pair.sample_b)]:
        for head_idx, target in enumerate(target_cols):
            for neuron_idx in range(eval_data["head_pre"].shape[2]):
                rows.append(
                    {
                        "sample": tag,
                        "sample_idx": int(sample_idx),
                        "head_idx": int(head_idx),
                        "target": target,
                        "head_neuron": int(neuron_idx),
                        "pre_activation": float(eval_data["head_pre"][sample_idx, head_idx, neuron_idx]),
                        "active": bool(eval_data["head_mask"][sample_idx, head_idx, neuron_idx]),
                    }
                )
    return pd.DataFrame(rows)


def _pair_metadata(
    bundle: RunBundle,
    pair: PairSelection,
    eval_data: dict[str, np.ndarray],
    top_changed_sched: pd.DataFrame,
    focus_target: str,
    focus_head_idx: int,
) -> dict[str, Any]:
    pred_delta = eval_data["pred"][pair.sample_b] - eval_data["pred"][pair.sample_a]
    head_flip_counts = np.not_equal(
        eval_data["head_mask"][pair.sample_a],
        eval_data["head_mask"][pair.sample_b],
    ).sum(axis=1)
    return {
        "run_dir": str(bundle.run_dir),
        "sample_a": int(pair.sample_a),
        "sample_b": int(pair.sample_b),
        "score": float(pair.score),
        "shared_flips": int(pair.shared_flips),
        "focus_target": focus_target,
        "focus_head_flips": int(pair.focus_head_flips),
        "total_head_flips": int(pair.total_head_flips),
        "non_sched_diff": float(pair.non_sched_diff),
        "sched_diff": float(pair.sched_diff),
        "shared_active_count_a": int(eval_data["shared_mask"][pair.sample_a].sum()),
        "shared_active_count_b": int(eval_data["shared_mask"][pair.sample_b].sum()),
        "focus_output_delta": float(pred_delta[focus_head_idx]),
        "head_flip_counts": {target: int(head_flip_counts[idx]) for idx, target in enumerate(bundle.target_cols)},
        "pred_delta": {target: float(pred_delta[idx]) for idx, target in enumerate(bundle.target_cols)},
        "top_changed_sched_features": top_changed_sched.head(8).to_dict(orient="records"),
    }


def _rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    r, g, b, _ = mcolors.to_rgba(hex_color)
    return (r, g, b, alpha)


def _draw_circle(ax: plt.Axes, x: float, y: float, r: float, facecolor: Any, edgecolor: str, lw: float, z: int = 3) -> None:
    ax.add_patch(Circle((x, y), r, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=z))


def _draw_split_circle(
    ax: plt.Axes,
    x: float,
    y: float,
    r: float,
    left_face: Any,
    right_face: Any,
    edgecolor: str,
    lw: float,
    outline_color: str | None = None,
    outline_lw: float = 0.0,
) -> None:
    ax.add_patch(Wedge((x, y), r, 90, 270, facecolor=left_face, edgecolor="none", zorder=4))
    ax.add_patch(Wedge((x, y), r, -90, 90, facecolor=right_face, edgecolor="none", zorder=4))
    ax.add_patch(Circle((x, y), r, facecolor="none", edgecolor=edgecolor, linewidth=lw, zorder=5))
    if outline_color is not None and outline_lw > 0.0:
        ax.add_patch(Circle((x, y), r * 1.18, facecolor="none", edgecolor=outline_color, linewidth=outline_lw, zorder=6))


def _draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, facecolor: Any, edgecolor: str, label: str, fontsize: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2.0, y - h / 2.0),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.1,
            zorder=2,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="#2b2826", zorder=3)


def _draw_edge(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: Any, lw: float, alpha: float, z: int = 1) -> None:
    ax.add_line(Line2D([x0, x1], [y0, y1], color=color, linewidth=lw, alpha=alpha, zorder=z))


def _network_layout(head_specs: list[tuple[int, str]], sched_labels: list[str]) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "sched_inputs": {},
        "collapsed_inputs": {},
        "shared": {},
        "heads": {},
        "outputs": {},
        "columns": {"inputs": 0.08, "shared": 0.38},
    }

    sched_y = np.linspace(0.93, 0.07, len(sched_labels))
    for idx, name in enumerate(sched_labels):
        layout["sched_inputs"][name] = (0.08, float(sched_y[idx]))

    collapsed = [
        ("x_op", 0.84),
        ("x_cont", 0.50),
        ("other fixed context", 0.16),
    ]
    for name, y in collapsed:
        layout["collapsed_inputs"][name] = (0.20, y)

    shared_y = np.linspace(0.95, 0.05, 32)
    for idx, y in enumerate(shared_y):
        layout["shared"][idx] = (0.43, float(y))

    if len(head_specs) == 1:
        head_idx, target = head_specs[0]
        x = 0.80
        y_top = 0.50
        ys = np.linspace(y_top + 0.22, y_top - 0.22, 16)
        layout["heads"][head_idx] = {
            "target": target,
            "x": x,
            "ys": ys,
            "label_pos": (x, y_top + 0.29),
        }
        layout["outputs"][head_idx] = (x + 0.08, y_top)
        return layout

    x_positions = [0.64, 0.79, 0.94]
    y_bases = [0.77, 0.29]
    for display_idx, (head_idx, target) in enumerate(head_specs):
        row = display_idx // 3
        col = display_idx % 3
        x = x_positions[col]
        y_top = y_bases[row]
        ys = np.linspace(y_top + 0.13, y_top - 0.13, 16)
        layout["heads"][head_idx] = {
            "target": target,
            "x": x,
            "ys": ys,
            "label_pos": (x, y_top + 0.19),
        }
        layout["outputs"][head_idx] = (x + 0.05, y_top)

    return layout


def _edge_scales(values: np.ndarray, floor: float, ceiling: float) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if math.isclose(vmin, vmax):
        return np.full_like(values, (floor + ceiling) / 2.0, dtype=float)
    return floor + (values - vmin) / (vmax - vmin) * (ceiling - floor)


def _draw_architecture_edges(
    ax: plt.Axes,
    bundle: RunBundle,
    layout: dict[str, Any],
    sched_features: list[str],
    collapsed_counts: dict[str, int],
    display_head_indices: list[int],
    top_sched_per_neuron: int,
    top_shared_to_head_per_neuron: int,
    palette: dict[str, str],
) -> None:
    sched_indices = {name: idx for idx, name in enumerate(bundle.feature_cols) if name in set(sched_features)}
    collapsed_anchor = {
        "x_op": "#c9d7e6",
        "x_cont": "#e3c9b7",
        "other fixed context": "#d7d4cf",
    }
    for shared_idx in range(bundle.shared_weight.shape[0]):
        weight_row = bundle.shared_weight[shared_idx]
        sched_abs = np.array([abs(weight_row[sched_indices[name]]) for name in sched_features], dtype=float)
        if sched_abs.size:
            top_sched_idx = np.argsort(-sched_abs)[: int(top_sched_per_neuron)]
            alphas = _edge_scales(sched_abs[top_sched_idx], 0.18, 0.90)
            for alpha, local_idx in zip(alphas, top_sched_idx):
                name = sched_features[int(local_idx)]
                x0, y0 = layout["sched_inputs"][name]
                x1, y1 = layout["shared"][shared_idx]
                _draw_edge(ax, x0 + 0.016, y0, x1 - 0.012, y1, palette["sched_edge"], 1.1, float(alpha))

        if collapsed_counts["x_op"] > 0:
            x0, y0 = layout["collapsed_inputs"]["x_op"]
            x1, y1 = layout["shared"][shared_idx]
            mean_abs = float(np.mean(np.abs(weight_row[:80])))
            _draw_edge(ax, x0 + 0.052, y0, x1 - 0.012, y1, _rgba(collapsed_anchor["x_op"], 0.22 + 0.35 * min(1.0, mean_abs)), 0.8, 0.7)
        if collapsed_counts["x_cont"] > 0:
            x0, y0 = layout["collapsed_inputs"]["x_cont"]
            x1, y1 = layout["shared"][shared_idx]
            mean_abs = float(np.mean(np.abs(weight_row[80:168])))
            _draw_edge(ax, x0 + 0.052, y0, x1 - 0.012, y1, _rgba(collapsed_anchor["x_cont"], 0.22 + 0.35 * min(1.0, mean_abs)), 0.8, 0.7)
        if collapsed_counts["other fixed context"] > 0:
            x0, y0 = layout["collapsed_inputs"]["other fixed context"]
            x1, y1 = layout["shared"][shared_idx]
            mean_abs = float(np.mean(np.abs(weight_row)))
            _draw_edge(ax, x0 + 0.062, y0, x1 - 0.012, y1, _rgba(collapsed_anchor["other fixed context"], 0.15 + 0.25 * min(1.0, mean_abs)), 0.7, 0.55)

    for head_idx in display_head_indices:
        weight = bundle.head_weights[head_idx]
        for neuron_idx in range(weight.shape[0]):
            abs_vals = np.abs(weight[neuron_idx])
            top_idx = np.argsort(-abs_vals)[: int(top_shared_to_head_per_neuron)]
            alphas = _edge_scales(abs_vals[top_idx], 0.15, 0.85)
            x1 = layout["heads"][head_idx]["x"]
            y1 = float(layout["heads"][head_idx]["ys"][neuron_idx])
            for alpha, shared_idx in zip(alphas, top_idx):
                x0, y0 = layout["shared"][int(shared_idx)]
                _draw_edge(ax, x0 + 0.012, y0, x1 - 0.012, y1, palette["shared_edge"], 0.9, float(alpha))


def _draw_network_nodes(
    ax: plt.Axes,
    bundle: RunBundle,
    layout: dict[str, Any],
    sched_features: list[str],
    collapsed_counts: dict[str, int],
    sample_a_shared: np.ndarray | None,
    sample_b_shared: np.ndarray | None,
    sample_a_heads: np.ndarray | None,
    sample_b_heads: np.ndarray | None,
    palette: dict[str, str],
    title: str,
    target_header: str,
) -> None:
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, 1.02)
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=8.0, fontsize=12.5, fontweight="semibold")

    ax.text(0.08, 1.0, "Schedulable inputs", fontsize=11.2, fontweight="semibold", ha="left", va="bottom")
    ax.text(0.18, 1.0, "Collapsed fixed context", fontsize=11.2, fontweight="semibold", ha="center", va="bottom")
    ax.text(0.43, 1.0, "Shared trunk (32 ReLU)", fontsize=11.2, fontweight="semibold", ha="center", va="bottom")
    ax.text(0.80, 1.0, target_header, fontsize=11.2, fontweight="semibold", ha="center", va="bottom")

    for name in sched_features:
        x, y = layout["sched_inputs"][name]
        _draw_circle(ax, x, y, 0.010, palette["sched_fill"], palette["sched_edge"], 0.8)
        ax.text(x - 0.018, y, name, ha="right", va="center", fontsize=7.7)

    for label, (x, y) in layout["collapsed_inputs"].items():
        count = collapsed_counts[label]
        _draw_box(ax, x, y, 0.10, 0.07, palette["collapsed_fill"], palette["neutral_edge"], f"{label}\n{count} feats", 8.2)

    shared_radius = 0.0076
    for shared_idx, (x, y) in layout["shared"].items():
        if sample_a_shared is None or sample_b_shared is None:
            _draw_circle(ax, x, y, shared_radius, palette["neutral_fill"], palette["neutral_edge"], 0.6)
        else:
            changed = bool(sample_a_shared[shared_idx] != sample_b_shared[shared_idx])
            _draw_split_circle(
                ax,
                x,
                y,
                shared_radius,
                palette["active_fill"] if sample_a_shared[shared_idx] else palette["inactive_fill"],
                palette["active_fill_b"] if sample_b_shared[shared_idx] else palette["inactive_fill"],
                palette["neutral_edge"],
                0.55,
                outline_color=palette["changed_edge"] if changed else None,
                outline_lw=1.0 if changed else 0.0,
            )

    head_radius = 0.0055
    for head_idx, head_layout in layout["heads"].items():
        label_x, label_y = head_layout["label_pos"]
        ax.text(label_x, label_y, head_layout["target"], ha="center", va="bottom", fontsize=8.3, fontweight="semibold")
        x = head_layout["x"]
        ys = head_layout["ys"]
        if sample_a_heads is None or sample_b_heads is None:
            for y in ys:
                _draw_circle(ax, x, float(y), head_radius, palette["neutral_fill"], palette["neutral_edge"], 0.45)
        else:
            for neuron_idx, y in enumerate(ys):
                changed = bool(sample_a_heads[head_idx, neuron_idx] != sample_b_heads[head_idx, neuron_idx])
                _draw_split_circle(
                    ax,
                    x,
                    float(y),
                    head_radius,
                    palette["active_fill"] if sample_a_heads[head_idx, neuron_idx] else palette["inactive_fill"],
                    palette["active_fill_b"] if sample_b_heads[head_idx, neuron_idx] else palette["inactive_fill"],
                    palette["neutral_edge"],
                    0.4,
                    outline_color=palette["changed_edge"] if changed else None,
                    outline_lw=0.8 if changed else 0.0,
                )

        out_x, out_y = layout["outputs"][head_idx]
        _draw_box(ax, out_x, out_y, 0.060, 0.038, palette["output_fill"], palette["output_edge"], "y", 8.0)
        _draw_edge(ax, x + 0.010, out_y, out_x - 0.026, out_y, palette["shared_edge"], 0.9, 0.7)


def _build_palette(style_cfg: dict[str, Any]) -> dict[str, str]:
    palettes = (style_cfg.get("style", {}) or {}).get("palettes", {}) or {}
    schedule = dict(palettes.get("schedule", {}) or {})
    replay = dict(palettes.get("replay", {}) or {})
    accents = dict((style_cfg.get("style", {}) or {}).get("accents", {}) or {})
    return {
        "sched_fill": schedule.get("high_vis", "#b56576"),
        "sched_edge": schedule.get("low_vis", "#355070"),
        "collapsed_fill": accents.get("neutral_node_fill", "#d7d4cf"),
        "neutral_fill": "#f4f2ef",
        "neutral_edge": accents.get("neutral_node_edge", "#5d544e"),
        "shared_edge": replay.get("default", "#1f77b4"),
        "active_fill": schedule.get("low_vis", "#355070"),
        "active_fill_b": schedule.get("medium_vis", "#6d597a"),
        "inactive_fill": "#ece8e1",
        "changed_edge": accents.get("constraint_limit", "#c46646"),
        "output_fill": accents.get("highlighted_ibr_fill", "#2a9d8f"),
        "output_edge": accents.get("highlighted_ibr_edge", "#264653"),
    }


def _render_static_figure(
    bundle: RunBundle,
    layout: dict[str, Any],
    pair: PairSelection,
    eval_data: dict[str, np.ndarray],
    top_changed_sched: pd.DataFrame,
    output_dir: Path,
    cfg: dict[str, Any],
    style_cfg: dict[str, Any],
    focus_target: str,
    focus_head_idx: int,
    sched_mask: np.ndarray,
) -> None:
    palette = _build_palette(style_cfg)
    sched_features = [name for name, flag in zip(bundle.feature_cols, sched_mask) if flag]
    collapsed_counts = _collapsed_group_counts(bundle.feature_cols, sched_mask)
    head_specs = [(focus_head_idx, focus_target)]

    fig = plt.figure(figsize=(float(cfg["style"]["figure_width"]), float(cfg["style"]["figure_height"])))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.10, hspace=0.18)
    ax_arch = fig.add_subplot(gs[:, 0])
    ax_compare = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[1, 1])

    _draw_architecture_edges(
        ax_arch,
        bundle,
        layout,
        sched_features,
        collapsed_counts,
        [focus_head_idx],
        int(cfg["edge_display"]["top_sched_to_shared_per_neuron"]),
        int(cfg["edge_display"]["top_shared_to_head_per_neuron"]),
        palette,
    )
    _draw_network_nodes(
        ax_arch,
        bundle,
        layout,
        sched_features,
        collapsed_counts,
        None,
        None,
        None,
        None,
        palette,
        f"A. {focus_target} head with M/D schedulable inputs",
        f"Selected head: {focus_target} (16 ReLU)",
    )

    _draw_architecture_edges(
        ax_compare,
        bundle,
        layout,
        sched_features,
        collapsed_counts,
        [focus_head_idx],
        int(cfg["edge_display"]["top_sched_to_shared_per_neuron"]),
        int(cfg["edge_display"]["top_shared_to_head_per_neuron"]),
        palette,
    )
    _draw_network_nodes(
        ax_compare,
        bundle,
        layout,
        sched_features,
        collapsed_counts,
        eval_data["shared_mask"][pair.sample_a],
        eval_data["shared_mask"][pair.sample_b],
        eval_data["head_mask"][pair.sample_a],
        eval_data["head_mask"][pair.sample_b],
        palette,
        f"B. ReLU pattern change for {focus_target}",
        f"Selected head: {focus_target} (16 ReLU)",
    )
    head_flip_counts = np.not_equal(
        eval_data["head_mask"][pair.sample_a],
        eval_data["head_mask"][pair.sample_b],
    ).sum(axis=1)
    ax_compare.text(
        0.01,
        -0.03,
        "Left half = sample A, right half = sample B. Highlighted outlines mark neurons whose ReLU state flips.\n"
        f"Shared flips: {pair.shared_flips}. {focus_target} head flips: {pair.focus_head_flips}. "
        f"Total head flips across all outputs: {pair.total_head_flips}.",
        transform=ax_compare.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        color="#3f3935",
    )

    top_bars = top_changed_sched.iloc[::-1]
    ax_summary.barh(top_bars["feature"], top_bars["abs_norm_delta"], color=palette["sched_fill"], alpha=0.85)
    ax_summary.set_title("C. M/D input changes driving the switch", loc="left", fontsize=12.0, fontweight="semibold")
    ax_summary.set_xlabel("Absolute normalized change")
    ax_summary.grid(axis="x", alpha=0.18)
    ax_summary.grid(axis="y", alpha=0.0)
    for spine in ("top", "right"):
        ax_summary.spines[spine].set_visible(False)

    pred_delta = eval_data["pred"][pair.sample_b] - eval_data["pred"][pair.sample_a]
    summary_text = f"{focus_target}: {float(pred_delta[focus_head_idx]):+.3f}"
    ax_summary.text(
        1.02,
        0.98,
        "Predicted output delta\n" + summary_text,
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#faf7f2", edgecolor="#c3b8ac", linewidth=0.8),
    )
    ax_summary.text(
        0.0,
        -0.26,
        f"Changing only the M/D scheduling channels moves the active ReLU region. "
        f"For {focus_target}, the shared mask and selected head binaries switch on/off enough that a fixed activation pattern is not credible.",
        transform=ax_summary.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#3f3935",
    )

    legend_handles = [
        Patch(facecolor=palette["active_fill"], edgecolor=palette["neutral_edge"], label="Active in sample A"),
        Patch(facecolor=palette["active_fill_b"], edgecolor=palette["neutral_edge"], label="Active in sample B"),
        Patch(facecolor=palette["inactive_fill"], edgecolor=palette["neutral_edge"], label="Inactive"),
        Line2D([0], [0], color=palette["changed_edge"], linewidth=2.0, label="State flip"),
    ]
    ax_compare.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, -0.24), ncol=2)

    fig.suptitle(
        f"MTLSH ReLU-Region Motion For {focus_target} Under M/D Scheduling Changes",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=14.0,
        fontweight="semibold",
    )
    _save_figure(fig, "mtlsh_activation_motion_static", output_dir)


def _frame_inputs(bundle: RunBundle, pair: PairSelection, frames: int, sched_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lambdas = np.linspace(0.0, 1.0, int(frames), dtype=np.float32)
    x_frames = np.repeat(bundle.x_norm[pair.sample_a][None, :], int(frames), axis=0)
    start_sched = bundle.x_norm[pair.sample_a, sched_mask]
    end_sched = bundle.x_norm[pair.sample_b, sched_mask]
    x_frames[:, sched_mask] = (1.0 - lambdas[:, None]) * start_sched[None, :] + lambdas[:, None] * end_sched[None, :]
    return lambdas, x_frames


def _render_animation(
    bundle: RunBundle,
    layout: dict[str, Any],
    pair: PairSelection,
    output_dir: Path,
    cfg: dict[str, Any],
    style_cfg: dict[str, Any],
    sched_mask: np.ndarray,
    focus_target: str,
    focus_head_idx: int,
) -> pd.DataFrame:
    palette = _build_palette(style_cfg)
    sched_features = [name for name, flag in zip(bundle.feature_cols, sched_mask) if flag]
    collapsed_counts = _collapsed_group_counts(bundle.feature_cols, sched_mask)
    lambdas, x_frames = _frame_inputs(bundle, pair, int(cfg["animation_frames"]), sched_mask)
    eval_frames = _evaluate_batch(bundle, x_frames)
    base_shared = eval_frames["shared_mask"][0]
    base_heads = eval_frames["head_mask"][0]

    summary_rows = []
    for frame_idx, lam in enumerate(lambdas):
        head_flips = np.not_equal(eval_frames["head_mask"][frame_idx], base_heads).sum(axis=1)
        summary_rows.append(
            {
                "frame": int(frame_idx),
                "lambda": float(lam),
                "shared_active_count": int(eval_frames["shared_mask"][frame_idx].sum()),
                "shared_flip_vs_frame0": int(np.not_equal(eval_frames["shared_mask"][frame_idx], base_shared).sum()),
                "focus_head_flip_vs_frame0": int(head_flips[focus_head_idx]),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    fig = plt.figure(figsize=(14.5, 9.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[12.0, 1.7], hspace=0.05)
    ax_net = fig.add_subplot(gs[0, 0])
    ax_footer = fig.add_subplot(gs[1, 0])

    def draw_frame(frame_idx: int) -> None:
        ax_net.clear()
        ax_footer.clear()
        _draw_architecture_edges(
            ax_net,
            bundle,
            layout,
            sched_features,
            collapsed_counts,
            [focus_head_idx],
            int(cfg["edge_display"]["top_sched_to_shared_per_neuron"]),
            int(cfg["edge_display"]["top_shared_to_head_per_neuron"]),
            palette,
        )
        _draw_network_nodes(
            ax_net,
            bundle,
            layout,
            sched_features,
            collapsed_counts,
            base_shared,
            eval_frames["shared_mask"][frame_idx],
            base_heads,
            eval_frames["head_mask"][frame_idx],
            palette,
            f"{focus_target} activation motion while only M/D schedulable inputs interpolate",
            f"Selected head: {focus_target} (16 ReLU)",
        )
        ax_net.text(
            0.01,
            -0.03,
            "Left half = frame 0 baseline, right half = current frame. "
            "Changed outlines mark schedule-driven ReLU binary switches.",
            transform=ax_net.transAxes,
            ha="left",
            va="top",
            fontsize=8.4,
            color="#3f3935",
        )

        row = summary_df.iloc[int(frame_idx)]
        ax_footer.set_axis_off()
        ax_footer.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax_footer.transAxes, facecolor="#faf7f2", edgecolor="#d7cfc4"))
        footer_text = (
            f"lambda = {row['lambda']:.2f}    "
            f"shared flips vs frame 0 = {int(row['shared_flip_vs_frame0'])}    "
            f"{focus_target} head flips vs frame 0 = {int(row['focus_head_flip_vs_frame0'])}"
        )
        ax_footer.text(0.02, 0.50, footer_text, transform=ax_footer.transAxes, ha="left", va="center", fontsize=10.2, color="#2b2826")

    draw_frame(0)
    anim = animation.FuncAnimation(fig, draw_frame, frames=int(cfg["animation_frames"]), interval=300, repeat=True)

    gif_path = output_dir / "mtlsh_activation_motion.gif"
    anim.save(gif_path, writer=animation.PillowWriter(fps=4))

    mp4_path = output_dir / "mtlsh_activation_motion.mp4"
    mp4_status = {"saved": False, "path": str(mp4_path)}
    try:
        writer = animation.FFMpegWriter(fps=4, bitrate=1600)
        anim.save(mp4_path, writer=writer)
        mp4_status["saved"] = True
    except Exception as exc:
        mp4_status["error"] = str(exc)
        print(f"MP4 export skipped: {exc}")

    plt.close(fig)
    with (output_dir / "animation_export_status.json").open("w", encoding="utf-8") as handle:
        json.dump(mp4_status, handle, indent=2)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize MTLSH ReLU-region motion under schedulable-input changes.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the activation-motion YAML config.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = REPO_ROOT / str(cfg["run_dir"])
    output_dir = REPO_ROOT / str(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    style_cfg = _apply_thesis_style()
    print(f"Loading retained run from {run_dir}", flush=True)
    bundle = _load_run_bundle(run_dir)
    if len(bundle.target_cols) != 6:
        raise ValueError(f"Expected 6 heads/targets, found {len(bundle.target_cols)}.")

    focus_target = str(cfg.get("focus_target", "dev_COI"))
    if focus_target not in bundle.target_cols:
        raise ValueError(f"focus_target '{focus_target}' not found in run targets {bundle.target_cols}.")
    focus_head_idx = int(bundle.target_cols.index(focus_target))

    sched_feature_mode = str(cfg.get("sched_feature_mode", "all_sched"))
    sched_mask = _sched_feature_mask(bundle.feature_cols, sched_feature_mode)
    expected_sched = 10 if sched_feature_mode == "md_only" else 20
    if int(sched_mask.sum()) != expected_sched:
        raise ValueError(f"Expected {expected_sched} schedulable inputs for mode '{sched_feature_mode}', found {int(sched_mask.sum())}.")

    print(
        f"Evaluating activation masks for target '{focus_target}' with {int(sched_mask.sum())} schedulable inputs.",
        flush=True,
    )
    eval_data = _evaluate_batch(bundle, bundle.x_norm)
    if eval_data["shared_mask"].shape[1] != 32:
        raise ValueError(f"Expected shared mask width 32, found {eval_data['shared_mask'].shape[1]}.")
    if eval_data["head_mask"].shape[1:] != (6, 16):
        raise ValueError(f"Expected head mask shape (n_samples, 6, 16), found {eval_data['head_mask'].shape}.")

    pair = _select_pair(
        bundle.x_norm,
        eval_data["shared_mask"],
        eval_data["head_mask"],
        sched_mask,
        focus_head_idx,
        int(cfg["candidate_search_size"]),
        dict(cfg.get("pair_selection", {}).get("score_weights", {}) or {}),
        int(cfg.get("pair_selection", {}).get("min_shared_flip_target", 4)),
        int(cfg.get("pair_selection", {}).get("min_focus_head_flip_target", 1)),
    )
    if pair.sample_a == pair.sample_b:
        raise RuntimeError("Selected identical sample indices.")
    if pair.shared_flips <= 0:
        raise RuntimeError("Selected pair has no shared-layer activation flips.")

    top_changed_sched = _top_changed_sched_rows(bundle.feature_cols, bundle.x_raw, bundle.x_norm, sched_mask, pair)
    if top_changed_sched["abs_norm_delta"].max() <= 0.0:
        raise RuntimeError("Selected pair does not change schedulable inputs.")

    shared_table = _shared_activation_table(eval_data, pair)
    head_table = _head_activation_table(eval_data, pair, bundle.target_cols)
    pair_meta = _pair_metadata(bundle, pair, eval_data, top_changed_sched, focus_target, focus_head_idx)
    pair_meta_df = pd.DataFrame(
        [
            {
                "sample_a": pair_meta["sample_a"],
                "sample_b": pair_meta["sample_b"],
                "score": pair_meta["score"],
                "shared_flips": pair_meta["shared_flips"],
                "focus_target": pair_meta["focus_target"],
                "focus_head_flips": pair_meta["focus_head_flips"],
                "total_head_flips": pair_meta["total_head_flips"],
                "non_sched_diff": pair_meta["non_sched_diff"],
                "sched_diff": pair_meta["sched_diff"],
                "shared_active_count_a": pair_meta["shared_active_count_a"],
                "shared_active_count_b": pair_meta["shared_active_count_b"],
                "focus_output_delta": pair_meta["focus_output_delta"],
            }
        ]
    )

    pair_meta_df.to_csv(output_dir / "selected_pair_metadata.csv", index=False)
    top_changed_sched.to_csv(output_dir / "selected_pair_changed_sched_features.csv", index=False)
    shared_table.to_csv(output_dir / "shared_activation_table.csv", index=False)
    head_table.to_csv(output_dir / "head_activation_table.csv", index=False)
    with (output_dir / "selected_pair_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(pair_meta, handle, indent=2)

    sched_features = [name for name, flag in zip(bundle.feature_cols, sched_mask) if flag]
    layout = _network_layout([(focus_head_idx, focus_target)], sched_features)
    print("Rendering static figure.", flush=True)
    _render_static_figure(bundle, layout, pair, eval_data, top_changed_sched, output_dir, cfg, style_cfg, focus_target, focus_head_idx, sched_mask)
    print("Rendering animation.", flush=True)
    frame_summary = _render_animation(bundle, layout, pair, output_dir, cfg, style_cfg, sched_mask, focus_target, focus_head_idx)
    frame_summary.to_csv(output_dir / "animation_frame_summary.csv", index=False)

    print(f"Saved MTLSH activation-motion figure and animation to {output_dir}")


if __name__ == "__main__":
    main()
