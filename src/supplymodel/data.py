"""Cluster (representative-agent) parameters.

The model needs only *one set of aggregate values per cluster*:

=================  ===========================  ============================
column             symbol                       unit
=================  ===========================  ============================
``households``     ``n``                        number of households in the model
``demand``         ``d`` (``Actual Demand``)    m^3 / day, cluster mean
``storage``        ``K`` (``Storage Size``)     m^3, cluster mean
``supply_days``    ``s`` (``Supply Duration``)  days / week, cluster mean
=================  ===========================  ============================

Optional: ``households_total`` - household counts used as weights when the
population-level deficit is computed (the ``n1`` list in the notebooks).

Two ways to obtain them:

1. **Recommended / default** - read the published aggregates:
   :func:`load_cluster_parameters` (``data/cluster_parameters.csv``).
2. **From household-level data that are already clustered** -
   :func:`parameters_from_households`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "cluster_parameters.csv"

HOUSEHOLD_COLUMNS = {
    "supply_days": "Supply Duration",
    "demand": "Actual Demand",
    "storage": "Storage Size",
    "cluster": "Cluster",
}


@dataclass(frozen=True)
class ClusterParameters:
    """Aggregate values per cluster (arrays ordered by cluster 1, 2, 3, ...)."""

    cluster_ids: np.ndarray
    households: np.ndarray        # n  - used inside the simulation
    households_total: np.ndarray  # n1 - used to weight the population deficit
    demand: np.ndarray            # d
    storage: np.ndarray           # K
    supply_days: np.ndarray       # s

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_ids)

    @property
    def labels(self) -> list[str]:
        return [f"Cluster {c}" for c in self.cluster_ids]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cluster": self.cluster_ids,
                "households": self.households,
                "households_total": self.households_total,
                "demand": self.demand,
                "storage": self.storage,
                "supply_days": self.supply_days,
            }
        )

    def model_kwargs(self) -> dict:
        """Keyword arguments for :func:`supplymodel.simulate` (baseline values)."""
        return dict(
            demand=self.demand,
            storage=self.storage,
            households=self.households,
            supply_days=self.supply_days,
        )


def load_cluster_parameters(path: str | Path = DEFAULT_CSV) -> ClusterParameters:
    """Load aggregate cluster parameters from CSV (default: ``data/cluster_parameters.csv``)."""
    df = pd.read_csv(path).sort_values("cluster")
    total = df["households_total"] if "households_total" in df else df["households"]
    return ClusterParameters(
        cluster_ids=df["cluster"].to_numpy(),
        households=df["households"].to_numpy(dtype=float),
        households_total=total.to_numpy(dtype=float),
        demand=df["demand"].to_numpy(dtype=float),
        storage=df["storage"].to_numpy(dtype=float),
        supply_days=df["supply_days"].to_numpy(dtype=float),
    )


def parameters_from_households(
    df: pd.DataFrame,
    columns: dict[str, str] | None = None,
    households: list[int] | None = None,
    households_total: list[int] | None = None,
) -> ClusterParameters:
    """Aggregate household-level data (already assigned to clusters) to cluster means.

    Parameters
    ----------
    df
        One row per household with columns ``Supply Duration``, ``Actual Demand``,
        ``Storage Size`` and ``Cluster`` (override names via ``columns``).
        ``Storage Size`` is coerced to numeric; missing values are skipped when
        averaging (as in the paper notebooks).
    households, households_total
        Household counts per cluster. If omitted, the number of rows per cluster is
        used for both. Pass them explicitly if the model uses a different count.
    """
    cols = {**HOUSEHOLD_COLUMNS, **(columns or {})}
    d = df[[cols["supply_days"], cols["demand"], cols["storage"], cols["cluster"]]].copy()
    d[cols["storage"]] = pd.to_numeric(d[cols["storage"]], errors="coerce")
    grouped = d.groupby(cols["cluster"])
    means = grouped.mean(numeric_only=True)  # NaNs skipped
    counts = grouped.size()
    ids = means.index.to_numpy()
    n = np.asarray(households if households is not None else counts.to_numpy(), dtype=float)
    n_tot = np.asarray(households_total if households_total is not None else n, dtype=float)
    return ClusterParameters(
        cluster_ids=ids,
        households=n,
        households_total=n_tot,
        demand=means[cols["demand"]].to_numpy(),
        storage=means[cols["storage"]].to_numpy(),
        supply_days=means[cols["supply_days"]].to_numpy(),
    )
