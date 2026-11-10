"""Sequential supply: e vs M heatmaps for selected supply orders.

Main code - "water deficit, sequential supply (changing order of cluster 3): e vs M".
Clusters are supplied one after another, ``gap`` days apart (start days 0, 2, 4).
Orders shown: 1-2-3, 1-3-2, 3-1-2.
"""
import matplotlib.pyplot as plt
import numpy as np

from _common import parse_args
from supplymodel import start_days_for_order
from supplymodel.plotting import FONT, heat, save, style_axes
from supplymodel.sweeps import E_GRID_M, M_GRID, sweep_e_vs_M

args, P = parse_args(__doc__)
GAP = 2
perms = [(0, 1, 2), (0, 2, 1), (2, 0, 1)]
perm_labels = ["1-2-3", "1-3-2", "3-1-2"]
e, mw = E_GRID_M, M_GRID

heatmaps = np.zeros((len(perms), P.n_clusters, len(e), len(mw)))
for pi, perm in enumerate(perms):
    print(f"Running order {perm_labels[pi]}")
    heatmaps[pi] = sweep_e_vs_M(P, e, mw, start_days=start_days_for_order(perm, gap=GAP))
np.save(f"{args.out}/heatmaps_selected_orders.npy", heatmaps)

fig, axes = plt.subplots(3, 3, figsize=(20, 18), sharex=True, sharey=True, constrained_layout=True)
for pi, lab in enumerate(perm_labels):
    for j in range(P.n_clusters):
        ax = axes[pi, j]
        im = heat(ax, heatmaps[pi, j], [mw[0], mw[-1], e[0], e[-1]])
        style_axes(ax)
        if pi == 0:
            ax.set_title(P.labels[j], fontsize=FONT)
        if j == 0:
            ax.set_ylabel(f"Order {lab}", fontsize=FONT)
fig.supxlabel("Municipal water availability (M)", fontsize=FONT)
fig.supylabel("Extraction Rate (e)", fontsize=FONT)
cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.95, pad=0.02)
cbar.set_label("Frequency of running out of water", fontsize=FONT)
save(fig, f"{args.out}/heatmap_selected_permutationsticks.png")
if args.show:
    plt.show()
