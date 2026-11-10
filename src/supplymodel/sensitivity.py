"""Global (Sobol) sensitivity analysis of the deficit frequency.

For a *focal* cluster, five inputs are varied jointly: extraction rate, municipal
water availability, storage size, daily demand and supply duration. All other
clusters stay at baseline (``e = 1``, ``M = 1100``, and cluster-mean ``K``, ``d``, ``s``).
The output is the focal cluster's deficit frequency.

Note: as in the original ``run_model``, the focal cluster has its *own* ``M``; the
shared-network term is ``sum_j n_j * extraction_j / M_j``.
"""

from __future__ import annotations

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from .data import ClusterParameters
from .model import DEFAULT_CONFIG, SimConfig, simulate

PARAMETER_NAMES = ["Extraction Rate", "Municipal Water", "Storage Size", "Daily Demand", "Supply Duration"]
PARAMETER_LABELS = ["Extraction rate", "Municipal water availability", "Storage size",
                    "Daily demand", "Supply duration"]

BASELINE_E = 1.0
BASELINE_M = 1100.0

#: Supply schedules compared in the paper (start day of cluster 1, 2, 3). Index + 1 = ``order_k`` key.
ORDERS = [[0, 0, 0], [0, 2, 4], [0, 4, 2], [2, 4, 0], [2, 0, 4], [4, 0, 2], [4, 2, 0]]


def make_problem(params: ClusterParameters) -> dict:
    """SALib problem definition; bounds as in the paper."""
    return {
        "num_vars": 5,
        "names": PARAMETER_NAMES,
        "bounds": [
            [0, 2],
            [100, 2000],
            [0.5 * np.min(params.storage), 1.5 * np.max(params.storage)],
            [0.5 * np.min(params.demand), 1.5 * np.max(params.demand)],
            [1, 7],
        ],
    }


def focal_deficit(X, params: ClusterParameters, start_days, cluster: int,
                  config: SimConfig = DEFAULT_CONFIG, chunk: int = 4096) -> np.ndarray:
    """Deficit frequency of ``cluster`` for each row of ``X`` (columns = ``PARAMETER_NAMES``).

    Non-focal clusters are always set to baseline for every row (no state is shared
    between calls).
    """
    X = np.atleast_2d(X)
    nc = params.n_clusters
    out = np.empty(len(X))
    for lo in range(0, len(X), chunk):
        x = X[lo : lo + chunk]
        B = len(x)
        e = np.full((B, nc), BASELINE_E)
        M = np.full((B, nc), BASELINE_M)
        K = np.tile(params.storage, (B, 1))
        d = np.tile(params.demand, (B, 1))
        s = np.tile(params.supply_days, (B, 1))
        e[:, cluster], M[:, cluster] = x[:, 0], x[:, 1]
        K[:, cluster], d[:, cluster], s[:, cluster] = x[:, 2], x[:, 3], x[:, 4]
        res = simulate(extraction=e, municipal_water=M, supply_days=s, demand=d, storage=K,
                       households=params.households, start_days=list(start_days), config=config)
        out[lo : lo + B] = res.deficit_frequency[:, cluster]
    return out


def _worker(args):
    X, params, start_days, cluster, config = args
    return focal_deficit(X, params, start_days, cluster, config)


def run_sobol(params: ClusterParameters, start_days, N: int = 1024, seed: int = 123,
              config: SimConfig = DEFAULT_CONFIG, n_jobs: int = 1, X=None) -> dict:
    """Sobol indices for every cluster under one supply schedule.

    Returns ``{"cluster_1": {"S1", "S1_conf", "ST", "ST_conf", "S2", "S2_conf", "N", "start_days"}, ...}``
    with plain lists (JSON-serialisable). One common Saltelli sample is used for all clusters.
    """
    problem = make_problem(params)
    if X is None:
        X = sobol_sample.sample(problem, N, calc_second_order=True, seed=seed)
    jobs = [(X, params, list(start_days), c, config) for c in range(params.n_clusters)]
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(n_jobs) as pool:
            Ys = list(pool.map(_worker, jobs))
    else:
        Ys = [_worker(j) for j in jobs]

    result = {}
    for c, Y in enumerate(Ys):
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=True, print_to_console=False, seed=seed)
        result[f"cluster_{c + 1}"] = {
            k: np.asarray(Si[k]).tolist() for k in ("S1", "S1_conf", "ST", "ST_conf", "S2", "S2_conf")
        } | {"N": N, "start_days": list(start_days)}
    return result
