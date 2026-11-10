"""Sequential supply: e vs supply duration for all 6 supply-order permutations.

Main supplementary code - "water deficit, sequential supply (all supply order permutations):
e vs Supply duration". M is fixed at 1200; supply duration of cluster j is s_j / f, f in 1..3.
Writes one figure per permutation plus a combined 6 x 3 figure.
"""
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np

from _common import parse_args
from supplymodel import start_days_for_order
from supplymodel.sweeps import E_GRID_S, sweep_e_vs_supply

args, P = parse_args(__doc__)
GAP = 2
e = E_GRID_S
f = np.arange(1, 3.1, 0.1)
perms = list(permutations(range(P.n_clusters)))

heatmaps = np.zeros((len(perms), P.n_clusters, len(e), len(f)))
for pi, perm in enumerate(perms):
    print(f"Running permutation {pi + 1}/{len(perms)}: order {[c + 1 for c in perm]}")
    heatmaps[pi] = sweep_e_vs_supply(P, e, f, start_days=start_days_for_order(perm, gap=GAP))
np.save(f"{args.out}/heatmaps_duration.npy", heatmaps)

# x-axis (supply duration) of each cluster. Columns of `heatmaps` run f = 1 -> 3, i.e.
# supply duration from large to small, so the columns are flipped to put small s on the left.
s_cols = [P.supply_days[j] / f for j in range(P.n_clusters)]


def draw(ax, pi, j, perm):
    im = ax.imshow(heatmaps[pi, j][:, ::-1], cmap="viridis", origin="lower", aspect="auto",
                   extent=[s_cols[j][-1], s_cols[j][0], e[0], e[-1]], vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Frequency (water running out)")
    ax.set_title(f"Order {[c + 1 for c in perm]} \u2014 {P.labels[j]}")
    ax.set_xlabel("Supply Duration (days/week)")
    ax.set_ylabel("Extraction Rate")


for pi, perm in enumerate(perms):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Supply Order: {[c + 1 for c in perm]}", fontsize=14)
    for j in range(P.n_clusters):
        draw(axes[j], pi, j, perm)
    fig.tight_layout()
    path = f"{args.out}/heatmap_duration_order{''.join(str(c + 1) for c in perm)}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)

fig, axes = plt.subplots(len(perms), 3, figsize=(18, 24))
fig.suptitle("All Supply Order Permutations \u2014 Duration vs Extraction Rate", fontsize=16, y=1.01)
for pi, perm in enumerate(perms):
    for j in range(P.n_clusters):
        draw(axes[pi, j], pi, j, perm)
fig.tight_layout()
fig.savefig(f"{args.out}/heatmap_duration_all_permutations.png", dpi=300, bbox_inches="tight")
print("saved combined figure")
if args.show:
    plt.show()
