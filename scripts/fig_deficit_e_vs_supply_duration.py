"""Water deficit: extraction rate (e) vs supply duration (s), simultaneous supply.

Main code - "Water Deficit: e vs Supply Duration (frequency of running out of water)".
Supply duration of cluster j is s_j / f for f in 0.5..3.0; M is fixed at 1200.
The red dashed line marks each cluster's actual (mean) supply duration.
"""
import matplotlib.pyplot as plt
import numpy as np

from _common import parse_args
from supplymodel.plotting import OKABE_ITO, FONT, heat, nearest_indices, save, style_axes
from supplymodel.sweeps import E_GRID_S, F_GRID, sweep_e_vs_supply

args, P = parse_args(__doc__)
e, f = E_GRID_S, F_GRID
freq = sweep_e_vs_supply(P, e, f, start_days=[0] * P.n_clusters)      # (nc, p, q)
np.save(f"{args.out}/deficit_e_vs_supply_duration.npy", freq)

sx = [P.supply_days[j] / f for j in range(P.n_clusters)]               # x-values per cluster (descending)
e_idx, e_sel = nearest_indices(e, [0.8, 1.6, 2.4])

fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=False, constrained_layout=True)
images = []
for j, ax in enumerate(axes[1]):
    # columns run from large s to small s; reverse so the x-axis increases
    im = heat(ax, freq[j][:, ::-1], [sx[j][-1], sx[j][0], e[0], e[-1]])
    images.append(im)
    style_axes(ax)
    ax.axvline(x=P.supply_days[j], color="red", linestyle="--", linewidth=2)

for j, ax in enumerate(axes[0]):
    for idx, ev, color in zip(e_idx, e_sel, OKABE_ITO):
        ax.plot(sx[j], freq[j, idx, :], marker="o", color=color, label=f"e = {ev:.1f}", linewidth=2)
    ax.set_title(P.labels[j], fontsize=FONT)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    style_axes(ax)
    ax.legend(title="Extraction rate", fontsize=FONT, title_fontsize=FONT)

axes[0, 0].set_ylabel("Frequency", fontsize=FONT)
axes[1, 0].set_ylabel("Extraction Rate (e)", fontsize=FONT)
fig.supxlabel("Supply Duration (s)", fontsize=FONT)
cbar = fig.colorbar(images[0], ax=axes[1], location="right", shrink=0.95, pad=0.02)
cbar.set_label("Frequency of running out of water", fontsize=FONT)
save(fig, f"{args.out}/Composite_S_vs_frequencyx.png")
if args.show:
    plt.show()
