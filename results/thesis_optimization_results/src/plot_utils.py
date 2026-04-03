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
            "figure.figsize": (6.4, 4.2),
            "figure.dpi": 140,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )


def formulation_color(formulation_id: str) -> str:
    return FORMULATION_COLORS.get(str(formulation_id), "#4c6a92")


def save_figure(fig: plt.Figure, stem: str, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    target_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    png_path = target_dir / f"{stem}.png"
    pdf_path = target_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path
