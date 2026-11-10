"""Parameter-specific total-order Sobol indices (ST) across supply orders.

Main code - "Sensitivity analysis, parameter specific ST values across supply order".
Reads results/sobol_orders_N<N>.json (from run_sobol.py --mode orders).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import parse_args  # noqa: E402
from supplymodel import order_label  # noqa: E402
from supplymodel.plotting import save  # noqa: E402
from supplymodel.sensitivity import ORDERS, PARAMETER_LABELS  # noqa: E402


def add_args(ap):
    ap.add_argument("--N", type=int, default=1024)


args, P = parse_args(__doc__, add_args)
res = json.load(open(Path(args.out) / f"sobol_orders_N{args.N}.json"))["results"]
nc, npar, nord = P.n_clusters, len(PARAMETER_LABELS), len(ORDERS)
labels = [order_label(o) for o in ORDERS]

ST = np.zeros((nc, npar, nord))
for c in range(nc):
    for o in range(nord):
        ST[c, :, o] = res[f"cluster_{c + 1}"][f"order_{o + 1}"]["ST"]

fig, axes = plt.subplots(1, nc, figsize=(20, 6), sharey=True, constrained_layout=True)
vmax = ST.max()  # same colour scale for all clusters
for c, ax in enumerate(axes):
    im = ax.imshow(ST[c], cmap="viridis", vmin=0, vmax=vmax, aspect="auto", origin="upper")
    ax.set_xticks(np.arange(nord))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(npar))
    if c == 0:
        ax.set_yticklabels(PARAMETER_LABELS)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_title(f"Cluster {c + 1}", fontsize=18)
cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
cbar.set_label("Total-order Sobol index (ST)", fontsize=18)
fig.supxlabel("Supply order", fontsize=18)
save(fig, Path(args.out) / "Sobol_ST_all_clusters.png")
if args.show:
    plt.show()
