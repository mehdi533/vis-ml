from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from matplotlib import pyplot as plt

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


def set_thesis_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.4),
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "grid.color": "#6e6259",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#4b433d",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#2b2826",
            "legend.fontsize": 9,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#2b2826",
            "ytick.color": "#2b2826",
        }
    )


def formulation_color(formulation_id: str) -> str:
    return FORMULATION_COLORS.get(str(formulation_id), "#4c6a92")


def save_figure(fig: plt.Figure, stem: str, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    target_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    png_path = target_dir / f"{stem}.png"
    pdf_path = target_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path
