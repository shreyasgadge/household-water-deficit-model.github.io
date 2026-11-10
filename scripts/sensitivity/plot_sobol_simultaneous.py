"""First-order (S1) and total-order (ST) Sobol indices, simultaneous supply.

Reads results/sobol_simultaneous_N<N>.json (from run_sobol.py --mode simultaneous).
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import parse_args  # noqa: E402
from supplymodel.plotting import save  # noqa: E402
from supplymodel.sensitivity import PARAMETER_LABELS  # noqa: E402


def add_args(ap):
    ap.add_argument("--N", type=int, default=1024)


args, P = parse_args(__doc__, add_args)
res = json.load(open(Path(args.out) / f"sobol_simultaneous_N{args.N}.json"))["results"]
nc, npar = P.n_clusters, len(PARAMETER_LABELS)

g = lambda key: np.array([res[f"cluster_{c + 1}"][key] for c in range(nc)])  # noqa: E731
S1, ST, S1c, STc = g("S1"), g("ST"), g("S1_conf"), g("ST_conf")
xmax = np.max([S1 + S1c, ST + STc]) * 1.15

fig, axes = plt.subplots(1, nc, figsize=(20, 6), sharey=True, constrained_layout=True)
y, h = np.arange(npar), 0.35
for c, ax in enumerate(axes):
    ax.barh(y - h / 2, S1[c], height=h, xerr=S1c[c], capsize=4, label="S1")
    ax.barh(y + h / 2, ST[c], height=h, xerr=STc[c], capsize=4, label="ST")
    ax.set_yticks(y)
    if c == 0:
        ax.set_yticklabels(PARAMETER_LABELS)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=18)
    ax.set_title(f"Cluster {c + 1}", fontsize=18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(0, xmax)
axes[0].legend(fontsize=18, loc="lower right")
fig.supxlabel("Sobol index", fontsize=18)
save(fig, Path(args.out) / f"Sobol_S1_ST_simultaneous_N{args.N}.png")
if args.show:
    plt.show()
