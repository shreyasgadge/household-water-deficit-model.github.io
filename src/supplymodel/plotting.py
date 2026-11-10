"""Shared plotting style for the paper figures."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

CLUSTER_COLORS = ["#2A9D8F", "#E76F51", "#FFD23F"]   # bar charts / timing diagrams
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73"]        # line plots of selected e
FONT = 18
CMAP = "viridis"


def use_headless_backend():
    matplotlib.use("Agg")


def style_axes(ax, nbins=4, labelsize=FONT, x=True, y=True):
    """Fewer ticks and large tick labels, as used throughout the paper."""
    if x:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    if y:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.tick_params(axis="both", which="major", labelsize=labelsize)


def heat(ax, data, extent, vmin=0, vmax=1, cmap=CMAP):
    return ax.imshow(data, cmap=cmap, extent=extent, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)


def nearest_indices(grid, values):
    """Index of the grid point closest to each requested value (float-safe)."""
    grid = np.asarray(grid)
    idx = [int(np.abs(grid - v).argmin()) for v in values]
    return idx, grid[idx]


def save(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"saved {path}")
