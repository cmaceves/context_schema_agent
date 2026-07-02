"""Standard figure aesthetics for de_ppi plots: seaborn whitegrid, no top/right spines, Paul Tol colors.

Usage:
    from plot_style import apply_style, TOL, ARM_COLOR, despine
    apply_style()
    ax.hist(..., color=ARM_COLOR["crohn"])
    despine(ax)                     # (redundant if apply_style ran, but safe for per-axes control)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import tol_colors as tc

_B = tc.bright                                        # Paul Tol 'bright' qualitative (colorblind-safe)
TOL = {"blue": _B.blue, "red": _B.red, "green": _B.green, "cyan": _B.cyan,
       "purple": _B.purple, "yellow": _B.yellow, "grey": _B.grey}

# canonical per-disease-arm colors (+ a neutral grey for negative/null controls)
ARM_COLOR = {"crohn": _B.red, "uc": _B.blue, "alz": _B.green, "ild": _B.purple,
             "healthy": _B.grey, "negative": _B.grey}


def apply_style() -> None:
    """seaborn whitegrid background + top/right spines removed, applied globally."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "0.3",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 10,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
    })


def despine(ax) -> None:
    """Remove top/right spines for a single Axes (for use without a global apply_style)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
