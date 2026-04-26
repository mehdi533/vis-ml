from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from matplotlib import pyplot as plt
import yaml

try:
    from .config_analysis import FIGURES_DIR
except ImportError:
    from config_analysis import FIGURES_DIR  # type: ignore


FORMULATION_COLORS = {
    "ed": "#4c6a92",
    "ed_line": "#7a8da6",
    "ed_line_n1": "#8f6f4b",
    "ed_surrogate": "#c46646",
    "ed_line_n1_surrogate": "#2f7f6d",
    "ed_line_n1_surrogate_redispatch": "#5b8c45",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STYLE_CONFIG = REPO_ROOT / "configs/figure/thesis_plot_style.yaml"


def _load_style_config(style_config_path: Optional[Path] = None) -> dict[str, Any]:
    path = Path(style_config_path) if style_config_path is not None else DEFAULT_STYLE_CONFIG
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_thesis_style(style_config_path: Optional[Path] = None) -> None:
    style_cfg = _load_style_config(style_config_path)
    rc_from_cfg = dict(style_cfg.get("style", {}).get("matplotlib", {}).get("rc_params", {}) or {})
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
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#4b433d",
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "regular",
        "axes.labelsize": 10.5,
        "axes.labelcolor": "#2b2826",
        "axes.labelpad": 4.0,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.borderaxespad": 0.6,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#2b2826",
        "ytick.color": "#2b2826",
        "lines.linewidth": 1.8,
        "patch.edgecolor": "#2b2826",
        "patch.linewidth": 0.6,
    }
    rc_defaults.update(rc_from_cfg)
    plt.rcParams.update(rc_defaults)


def formulation_color(formulation_id: str) -> str:
    style_cfg = _load_style_config()
    formulation_palette = dict(style_cfg.get("style", {}).get("palettes", {}).get("formulation", {}) or {})
    merged = dict(FORMULATION_COLORS)
    merged.update({str(k): str(v) for k, v in formulation_palette.items()})
    return merged.get(str(formulation_id), "#4c6a92")


def save_figure(fig: plt.Figure, stem: str, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    target_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    png_path = target_dir / f"{stem}.png"
    pdf_path = target_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path
