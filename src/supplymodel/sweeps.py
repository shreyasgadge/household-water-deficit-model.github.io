"""Parameter sweeps that produce the heatmaps / response surfaces in the paper.

Every sweep returns *deficit frequencies* (fraction of time with empty storage),
shaped ``(n_clusters, n_rows, n_cols)`` with rows = extraction rate ``e`` and
columns = the second swept quantity, i.e. ``heatmap[j, z, y]`` in the notebooks.
"""

from __future__ import annotations

import numpy as np

from .data import ClusterParameters
from .model import DEFAULT_CONFIG, SimConfig, simulate

# Default grids used in the paper
E_GRID_M = np.arange(0, 2.1, 0.1)       # e vs M figures
M_GRID = np.arange(100, 2000, 100)
E_GRID_S = np.arange(0, 3.1, 0.1)       # e vs supply-duration figures
F_GRID = np.arange(0.5, 3.1, 0.1)       # supply duration = s / f
M_FIXED = 1200                          # M used in the supply-duration figures


def _grid(e_vals, x_vals):
    """Flattened (e, x) grid; index = row * len(x_vals) + col."""
    E, X = np.meshgrid(e_vals, x_vals, indexing="ij")
    return E.ravel(), X.ravel()


def sweep_e_vs_M(params: ClusterParameters, e_vals=E_GRID_M, M_vals=M_GRID,
                 start_days=(0, 0, 0), config: SimConfig = DEFAULT_CONFIG,
                 extraction_per_cluster=None) -> np.ndarray:
    """Deficit frequency over (e, M). Returns ``(nc, len(e_vals), len(M_vals))``."""
    nc = params.n_clusters
    E, X = _grid(e_vals, M_vals)
    e = np.tile(E[:, None], (1, nc))
    if extraction_per_cluster is not None:
        e = e * 0 + np.asarray(extraction_per_cluster)[None, :]
    res = simulate(extraction=e, municipal_water=X, start_days=list(start_days),
                   config=config, **params.model_kwargs()).deficit_frequency
    return res.T.reshape(nc, len(e_vals), len(M_vals))


def sweep_e_vs_supply(params: ClusterParameters, e_vals=E_GRID_S, f_vals=F_GRID,
                      municipal_water=M_FIXED, start_days=(0, 0, 0),
                      config: SimConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Deficit frequency over (e, f) where supply duration is ``s / f`` per cluster.

    Returns ``(nc, len(e_vals), len(f_vals))``. The supply-duration axis of
    cluster ``j`` is ``params.supply_days[j] / f_vals``.
    """
    nc = params.n_clusters
    E, F = _grid(e_vals, f_vals)
    kw = params.model_kwargs()
    kw["supply_days"] = params.supply_days[None, :] / F[:, None]
    res = simulate(extraction=np.tile(E[:, None], (1, nc)), municipal_water=municipal_water,
                   start_days=list(start_days), config=config, **kw).deficit_frequency
    return res.T.reshape(nc, len(e_vals), len(f_vals))


def sweep_relative_extraction(params: ClusterParameters, scenarios, M_vals=M_GRID,
                              start_days=(0, 0, 0), config: SimConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Per-cluster extraction rates (``scenarios``: list of length-nc tuples) vs M.

    Returns ``(nc, n_scenarios, len(M_vals))``.
    """
    nc = params.n_clusters
    sc = np.asarray(scenarios, dtype=float)                    # (S, nc)
    e = np.repeat(sc, len(M_vals), axis=0)                     # (S*Q, nc)
    M = np.tile(np.asarray(M_vals, dtype=float), len(sc))      # (S*Q,)
    res = simulate(extraction=e, municipal_water=M, start_days=list(start_days),
                   config=config, **params.model_kwargs()).deficit_frequency
    return res.T.reshape(nc, len(sc), len(M_vals))


def population_deficit_percent(freq: np.ndarray, weights) -> np.ndarray:
    """Weighted share (%) of households running out of water. ``freq``: ``(nc, ...)``."""
    w = np.asarray(weights, dtype=float).reshape(-1, *([1] * (freq.ndim - 1)))
    return (freq * w).sum(axis=0) / w.sum() * 100
