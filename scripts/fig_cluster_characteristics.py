"""Cluster characteristics: mean demand, storage, supply duration (+/- SEM) and household counts.

Main code - "Data plotting with mean and range" (bars show mean; error bars are the
standard error of the mean, pandas ddof=1).

* With ``--household-data`` (CSV of already-clustered households) error bars are shown.
* Without it, the aggregate means from ``data/cluster_parameters.csv`` are plotted
  without error bars.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import parse_args
from supplymodel.data import HOUSEHOLD_COLUMNS
from supplymodel.plotting import CLUSTER_COLORS, save


def add_args(ap):
    ap.add_argument("--household-data", default=None, help="household-level CSV, already clustered")


args, P = parse_args(__doc__, add_args)
labels = P.labels
x = np.arange(P.n_clusters)

if args.household_data:
    df = pd.read_csv(args.household_data)
    df[HOUSEHOLD_COLUMNS["storage"]] = pd.to_numeric(df[HOUSEHOLD_COLUMNS["storage"]], errors="coerce")
    groups = [df[df[HOUSEHOLD_COLUMNS["cluster"]] == c] for c in P.cluster_ids]

    def mean_sem(col):
        return (np.array([g[col].mean() for g in groups]), np.array([g[col].sem() for g in groups]))

    demand, demand_err = mean_sem(HOUSEHOLD_COLUMNS["demand"])
    storage, storage_err = mean_sem(HOUSEHOLD_COLUMNS["storage"])
    supply, supply_err = mean_sem(HOUSEHOLD_COLUMNS["supply_days"])
else:
    demand, storage, supply = P.demand, P.storage, P.supply_days
    demand_err = storage_err = supply_err = None


def barplot(ax, values, ylabel, errors=None, ylim=None, decimal=True):
    bars = ax.bar(x, values, color=CLUSTER_COLORS, width=0.6, yerr=errors, capsize=6,
                  error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="black"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=18)
    if ylim:
        ax.set_ylim(ylim)
    for i, (bar, v) in enumerate(zip(bars, values)):
        err = errors[i] if errors is not None else 0
        ax.annotate(f"{v:.1f}" if decimal else f"{int(v)}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + err),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=14)


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
barplot(axes[0], demand, "Mean Daily Demand ($m^{3}/day$)", demand_err, ylim=(0, 1.3))
barplot(axes[1], storage, "Mean Storage Size ($m^{3}$)", storage_err, ylim=(0, 7))
barplot(axes[2], supply, "Mean Supply Duration (days/week)", supply_err, ylim=(0, 5))
barplot(axes[3], P.households, "no. of households", None, ylim=(0, 190), decimal=False)
plt.tight_layout()
out = Path(args.out)
save(fig, out / "cluster_characteristics.png")
fig.savefig(out / "cluster_characteristics.pdf", bbox_inches="tight")
if args.show:
    plt.show()
