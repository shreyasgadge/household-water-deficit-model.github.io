"""Sequential supply: relative extraction-rate permutations vs M.

Main supplementary code - "extraction rate permutations vs M (frequency of running out of
water and relative extraction rates across clusters)".
Extraction rates {1, 1.5, 2} (L / I / H = low / intermediate / high) are assigned to
clusters 1, 2, 3 in all 6 permutations, for three supply orders.
"""
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np

from _common import parse_args
from supplymodel import start_days_for_order
from supplymodel.plotting import save
from supplymodel.sweeps import M_GRID, sweep_relative_extraction

args, P = parse_args(__doc__)
GAP = 2
perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
perm_labels = ["1-2-3", "2-3-1", "3-1-2"]
mw = M_GRID

levels = {1: "L", 1.5: "I", 2: "H"}
scenarios = list(permutations([1, 1.5, 2]))
pressure_labels = ["".join(levels[v] for v in sc) for sc in scenarios]   # LIH, LHI, ILH, IHL, HLI, HIL

heatmaps = np.zeros((len(perms), P.n_clusters, len(scenarios), len(mw)))
for pi, perm in enumerate(perms):
    print(f"Running supply order {perm_labels[pi]}")
    heatmaps[pi] = sweep_relative_extraction(P, scenarios, mw, start_days=start_days_for_order(perm, gap=GAP))
np.save(f"{args.out}/relative_extraction_heatmaps.npy", heatmaps)

fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True, sharey=True, constrained_layout=True)
for pi, lab in enumerate(perm_labels):
    for j in range(P.n_clusters):
        ax = axes[pi, j]
        im = ax.imshow(heatmaps[pi, j], cmap="viridis", origin="lower", aspect="auto",
                       extent=[mw[0], mw[-1], 0, len(scenarios)], vmin=0, vmax=1)
        if pi == 0:
            ax.set_title(P.labels[j], fontsize=16)
        if j == 0:
            ax.set_ylabel(f"Order {lab}\n\nExtraction rate permutations", fontsize=14)
        ax.set_yticks(np.arange(len(scenarios)) + 0.5)
        ax.set_yticklabels(pressure_labels, fontsize=11)
        if pi == 2:
            ax.set_xlabel("Municipal water availability (M)", fontsize=14)
cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.95, pad=0.02)
cbar.set_label("Frequency of running out of water", fontsize=14)
save(fig, f"{args.out}/all_orders_relative_extraction.png")
if args.show:
    plt.show()
