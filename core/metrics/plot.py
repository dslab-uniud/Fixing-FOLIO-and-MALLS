"""
Plot Safety–Usefulness curves for E1, E2, and E3.

Each public function accepts a dict mapping method-label → list of (x, y)
points (as returned by the curve_e* helpers in compute.py) and saves
the figure to *output_path*.

Design conventions (matching metrics.md):
  - A curve with ≥2 points is drawn as a connected line + small markers.
  - A single point is drawn as a large isolated marker (no connecting line).
  - Axes are fixed to [0, 1] × [0, 1].
  - A dashed diagonal reference line (random classifier) is drawn at y=x.
  - Methods are assigned distinct colors automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Color palette — enough for 12+ methods
# ---------------------------------------------------------------------------
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
]


def _color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------

def _draw_curve(
    ax,
    points: List[Tuple[Optional[float], Optional[float]]],
    color: str,
    label: str,
    marker: str = "o",
) -> None:
    """Draw one method curve onto *ax*."""
    # Filter out completely None points
    valid = [(x, y) for x, y in points if x is not None]
    if not valid:
        return

    xs = [p[0] for p in valid]
    ys = [p[1] if p[1] is not None else np.nan for p in valid]

    is_single = len(valid) == 1

    if is_single:
        ax.scatter(xs, ys, color=color, s=120, zorder=5, label=label, marker=marker)
    else:
        # Sort by usefulness for a clean left-to-right curve
        pairs = sorted(zip(xs, ys), key=lambda p: p[0])
        xs_s = [p[0] for p in pairs]
        ys_s = [p[1] for p in pairs]
        ax.plot(
            xs_s, ys_s,
            color=color, linewidth=1.6, label=label,
            marker=marker, markersize=5, zorder=4,
        )


def _finish_ax(
    ax,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1.0, zorder=1)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.8)


# ---------------------------------------------------------------------------
# E1 — Verdict Quality  (3 plots in one figure)
# ---------------------------------------------------------------------------

def plot_e1(
    method_curves_agg: Dict[str, List[Tuple]],
    method_curves_pos: Dict[str, List[Tuple]],
    method_curves_neg: Dict[str, List[Tuple]],
    output_path: str | Path,
    title_prefix: str = "",
) -> Path:
    """
    Generate a single figure with three side-by-side Safety–Usefulness plots
    for E1 (aggregate, positive, negative safety).

    Parameters
    ----------
    method_curves_agg / _pos / _neg :
        Dict mapping method label → list of (usefulness1, safety_variant) points.
    output_path :
        File path to save the figure (PNG/PDF/SVG inferred from extension).
    title_prefix :
        Optional string prepended to each sub-plot title (e.g. dataset name).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{title_prefix}E1 — Verdict Quality".strip(),
        fontsize=13, y=1.01,
    )

    all_methods = sorted(
        set(method_curves_agg) | set(method_curves_pos) | set(method_curves_neg)
    )
    color_map = {m: _color(i) for i, m in enumerate(all_methods)}

    subplots = [
        (axes[0], method_curves_agg, "Safety₁ (aggregate)", "E1-a: Safety₁ vs Usefulness₁"),
        (axes[1], method_curves_pos, "Safety₁_yes (positive)", "E1-b: Safety₁_yes vs Usefulness₁"),
        (axes[2], method_curves_neg, "Safety₁_no (negative)", "E1-c: Safety₁_no vs Usefulness₁"),
    ]

    for ax, curves, ylabel, title in subplots:
        for method, pts in curves.items():
            _draw_curve(ax, pts, color=color_map[method], label=method)
        _finish_ax(ax, xlabel="Usefulness₁", ylabel=ylabel, title=title)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# E2 — Refinement Quality  (1 plot)
# ---------------------------------------------------------------------------

def plot_e2(
    method_curves: Dict[str, List[Tuple]],
    output_path: str | Path,
    title_prefix: str = "",
) -> Path:
    """
    Generate a Safety₂ vs Usefulness₂ plot for methods that produce ψ.

    Parameters
    ----------
    method_curves :
        Dict mapping method label → list of (usefulness2, safety2) points.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(
        f"{title_prefix}E2 — Refinement Quality".strip(),
        fontsize=13,
    )

    all_methods = sorted(method_curves)
    color_map = {m: _color(i) for i, m in enumerate(all_methods)}

    for method, pts in method_curves.items():
        _draw_curve(ax, pts, color=color_map[method], label=method)

    _finish_ax(ax, xlabel="Usefulness₂", ylabel="Safety₂", title="Safety₂ vs Usefulness₂")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# E3 — Combined Quality  (E3 overlaid with E1 aggregate)
# ---------------------------------------------------------------------------

def plot_e3(
    method_curves_e3: Dict[str, List[Tuple]],
    method_curves_e1: Dict[str, List[Tuple]],
    output_path: str | Path,
    title_prefix: str = "",
) -> Path:
    """
    Generate the E3 plot (Safety₃ vs Usefulness₃) with E1 aggregate curves
    overlaid on the same axes.

    E1 curves are drawn with dashed lines; E3 curves with solid lines.
    The vertical gap Safety₁ − Safety₃ makes the refinement cost visible.

    Parameters
    ----------
    method_curves_e3 :
        Dict mapping method label → list of (usefulness3, safety3) points.
    method_curves_e1 :
        Dict mapping method label → list of (usefulness1, safety1_agg) points.
        Use the same method labels as method_curves_e3.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        f"{title_prefix}E3 — Combined Quality (with E1 overlay)".strip(),
        fontsize=13,
    )

    all_methods = sorted(set(method_curves_e3) | set(method_curves_e1))
    color_map = {m: _color(i) for i, m in enumerate(all_methods)}

    for method in all_methods:
        color = color_map[method]
        # E3 — solid
        if method in method_curves_e3:
            _draw_curve(
                ax, method_curves_e3[method],
                color=color, label=f"{method} (E3)",
                marker="o",
            )
        # E1 aggregate — dashed (same color, different linestyle)
        if method in method_curves_e1:
            pts = method_curves_e1[method]
            valid = [(x, y) for x, y in pts if x is not None]
            if valid:
                xs = [p[0] for p in sorted(valid)]
                ys = [p[1] if p[1] is not None else np.nan
                      for p in sorted(valid)]
                if len(valid) == 1:
                    ax.scatter(xs, ys, color=color, s=80, zorder=5,
                               marker="^", label=f"{method} (E1)",
                               facecolors="none", edgecolors=color, linewidths=1.5)
                else:
                    pairs = sorted(zip(xs, ys), key=lambda p: p[0])
                    ax.plot(
                        [p[0] for p in pairs],
                        [p[1] for p in pairs],
                        color=color, linestyle="--", linewidth=1.4,
                        marker="^", markersize=4, zorder=3,
                        label=f"{method} (E1)",
                    )

    # Custom legend: solid = E3, dashed = E1
    legend_extras = [
        mlines.Line2D([], [], color="black", linestyle="-",  label="── E3 (combined)"),
        mlines.Line2D([], [], color="black", linestyle="--", label="╌╌ E1 (verdict only)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + legend_extras,
        labels + [h.get_label() for h in legend_extras],
        fontsize=7, loc="lower right", framealpha=0.85,
    )

    _finish_ax(
        ax,
        xlabel="Usefulness₃",
        ylabel="Safety₃ / Safety₁",
        title="Safety₃ vs Usefulness₃  (dashed = E1 aggregate)",
    )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
