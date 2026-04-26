# plotting.py
# Training/evaluation plotting utilities for model experiments.

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Domain scaling metadata
# -----------------------------

@dataclass(frozen=True)
class PlotContext:
    """Optional unit-conversion context used for target plotting."""

    nominal_frequency_hz: Optional[float] = None
    system_base_mva: Optional[float] = None
    ibr_device_base_mva: Mapping[int, float] = field(default_factory=dict)
    case_path: Optional[str] = None


# -----------------------------
# Label / axis helpers
# -----------------------------

def _maybe_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(out) and out > 0.0:
        return out
    return None


def _default_case_path() -> Path:
    env_case = os.environ.get("VISML_ANDES_CASE_XLSX", "").strip()
    if env_case:
        return Path(env_case)
    return Path("data_generation/andes_cases/ieee39_full_ibrs.xlsx")


@lru_cache(maxsize=8)
def _load_regcv1_sheet(case_path_str: str):
    workbook_path = Path(case_path_str)
    if not workbook_path.exists():
        return None

    try:
        import pandas as pd

        regcv1 = pd.read_excel(workbook_path, sheet_name="REGCV1")
    except Exception:
        return None

    if "idx" not in regcv1.columns:
        return None
    return regcv1


def _resolve_plot_context(plot_context: Optional[PlotContext | Mapping]) -> PlotContext:
    if isinstance(plot_context, PlotContext):
        base = plot_context
    else:
        ctx = dict(plot_context or {})
        base = PlotContext(
            nominal_frequency_hz=_maybe_float(ctx.get("nominal_frequency_hz")),
            system_base_mva=_maybe_float(ctx.get("system_base_mva")),
            ibr_device_base_mva={
                int(k): float(v)
                for k, v in dict(ctx.get("ibr_device_base_mva", {}) or {}).items()
                if _maybe_float(v) is not None
            },
            case_path=ctx.get("case_path"),
        )

    case_path = str(base.case_path or _default_case_path())
    regcv1 = _load_regcv1_sheet(case_path)

    freq_hz = base.nominal_frequency_hz or _maybe_float(os.environ.get("VISML_SYSTEM_BASE_HZ"))
    system_mva = base.system_base_mva or _maybe_float(os.environ.get("VISML_SYSTEM_BASE_MVA"))
    ibr_base_map = {int(k): float(v) for k, v in base.ibr_device_base_mva.items()}

    if regcv1 is not None:
        if freq_hz is None and "fn" in regcv1.columns:
            values = [_maybe_float(v) for v in regcv1["fn"].tolist()]
            values = [v for v in values if v is not None]
            if values:
                freq_hz = float(np.median(values))

        if "Sn" in regcv1.columns:
            for _, row in regcv1.iterrows():
                idx = str(row.get("idx", ""))
                match = re.fullmatch(r"REGCV1_(\d+)", idx)
                if match is None:
                    continue
                sn_mva = _maybe_float(row.get("Sn"))
                if sn_mva is None:
                    continue
                unit_id = int(match.group(1))
                ibr_base_map.setdefault(unit_id, sn_mva)

    return PlotContext(
        nominal_frequency_hz=freq_hz,
        system_base_mva=system_mva,
        ibr_device_base_mva=ibr_base_map,
        case_path=case_path,
    )


def _target_plot_metadata(name: str, context: PlotContext):
    label = str(name)
    freq_hz = _maybe_float(context.nominal_frequency_hz)
    system_mva = _maybe_float(context.system_base_mva)

    if label == "rocof_COI":
        if freq_hz is not None:
            return {
                "display_name": "COI RoCoF",
                "axis_unit": f"p.u./s on {freq_hz:g} Hz base",
                "scale": 1.0 / freq_hz,
            }
        return {
            "display_name": "COI RoCoF",
            "axis_unit": "Hz/s",
            "scale": 1.0,
        }

    if label == "dev_COI":
        if freq_hz is not None:
            return {
                "display_name": "COI frequency deviation",
                "axis_unit": f"p.u. on {freq_hz:g} Hz base",
                "scale": 1.0 / freq_hz,
            }
        return {
            "display_name": "COI frequency deviation",
            "axis_unit": "Hz",
            "scale": 1.0,
        }

    match = re.fullmatch(r"Delta_P_IBR_(\d+)", label)
    if match is not None:
        unit_id = int(match.group(1))
        ibr_base = _maybe_float(context.ibr_device_base_mva.get(unit_id))
        if system_mva is not None and ibr_base is not None:
            return {
                "display_name": f"IBR {unit_id} power excursion",
                "axis_unit": "p.u. on IBR base",
                "scale": system_mva / ibr_base,
            }
        return {
            "display_name": f"IBR {unit_id} power excursion",
            "axis_unit": "p.u. (system base)",
            "scale": 1.0,
        }

    return {
        "display_name": label,
        "axis_unit": "raw",
        "scale": 1.0,
    }


def _apply_scientific_format(ax, axis="both"):
    try:
        ax.ticklabel_format(axis=axis, style="sci", scilimits=(0, 0), useMathText=True)
    except Exception:
        return


# -----------------------------
# Plot exports
# -----------------------------

def _save_figure(fig, out_path, dpi=150):
    fig.savefig(out_path, dpi=dpi)
    base_path = Path(out_path)
    if base_path.suffix.lower() != ".pdf":
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path)
        return [str(base_path), str(pdf_path)]
    return [str(base_path)]


def plot_losses(train_losses, val_losses, test_mse, train_eval_losses=None, out_path="loss_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss")
    if train_eval_losses is not None:
        ax.plot(epochs, train_eval_losses, label="Train eval loss")
    ax.plot(epochs, val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True)
    _apply_scientific_format(ax, axis="y")
    fig.tight_layout()
    saved_paths = _save_figure(fig, out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss curves to {', '.join(saved_paths)}")


def plot_scatter_per_target(
    y_true,
    y_pred,
    target_cols,
    out_dir="output/runx/",
    *,
    plot_context: Optional[PlotContext | Mapping] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    context = _resolve_plot_context(plot_context)

    for i, name in enumerate(target_cols):
        meta = _target_plot_metadata(str(name), context)
        scale = float(meta["scale"])
        y_true_plot = np.asarray(y_true[:, i], dtype=float) * scale
        y_pred_plot = np.asarray(y_pred[:, i], dtype=float) * scale
        min_val = min(float(np.min(y_true_plot)), float(np.min(y_pred_plot)))
        max_val = max(float(np.max(y_true_plot)), float(np.max(y_pred_plot)))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_true_plot, y_pred_plot, alpha=0.4, s=10)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        ax.set_xlabel(f"True {meta['display_name']} [{meta['axis_unit']}]")
        ax.set_ylabel(f"Predicted {meta['display_name']} [{meta['axis_unit']}]")
        ax.set_title(f"Test Set: {meta['display_name']}")
        ax.legend()
        ax.grid(True)
        _apply_scientific_format(ax, axis="both")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"scatter_{name}.png")
        saved_paths = _save_figure(fig, out_path, dpi=150)
        plt.close(fig)
        print(f"Saved scatter plot {', '.join(saved_paths)}")
