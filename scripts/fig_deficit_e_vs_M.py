"""Water deficit: extraction rate (e) vs municipal water availability (M).

Main code - "Water Deficit: e vs M (frequency of running out of water)", simultaneous supply.
Top row: frequency vs M for selected e. Bottom row: heatmaps.
Also writes the supplementary population-level figure (% of households affected),
"Main supplementary code: Population deficit: e vs M".
"""
import matplotlib.pyplot as plt
import numpy as np

from _common import parse_args
from supplymodel.plotting import OKABE_ITO, FONT, heat, nearest_indices, save, style_axes
from supplymodel.sweeps import E_GRID_M, M_GRID, population_deficit_percent, sweep_e_vs_M

args, P = parse_args(__doc__)
e, mw = E_GRID_M, M_GRID
freq = sweep_e_vs_M(P, e, mw, start_days=[0] * P.n_clusters)          # (nc, p, q)
np.save(f"{args.out}/deficit_e_vs_M.npy", freq)

extent = [mw[0], mw[-1], e[0], e[-1]]
e_idx, e_sel = nearest_indices(e, [0.5, 1.2, 2])

# ---- composite: lines (top) + heatmaps (bottom) ----
fig = plt.figure(figsize=(20, 12), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.04])
axes = np.empty((2, 3), dtype=object)
for r in range(2):
    for c in range(3):
        axes[r, c] = fig.add_subplot(gs[r, c])
cax = fig.add_subplot(gs[1, 3])

for j, ax in enumerate(axes[0]):
    for idx, ev, color in zip(e_idx, e_sel, OKABE_ITO):
        ax.plot(mw, freq[j, idx, :], marker="o", color=color, label=f"e = {ev:.1f}", linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_title(P.labels[j], fontsize=FONT)
    ax.grid(alpha=0.3)
    style_axes(ax)
    ax.legend(title="Extraction rate", fontsize=FONT, title_fontsize=FONT)

images = []
for j, ax in enumerate(axes[1]):
    images.append(heat(ax, freq[j], extent))
    style_axes(ax)

cbar = fig.colorbar(images[0], cax=cax)
cbar.set_label("Frequency of running out of water", fontsize=FONT)
cbar.ax.tick_params(labelsize=FONT)
axes[0, 0].set_ylabel("Frequency", fontsize=FONT)
axes[1, 0].set_ylabel("Extraction Rate (e)", fontsize=FONT)
axes[1, 1].set_xlabel("Municipal water availability (M)", fontsize=FONT)
save(fig, f"{args.out}/Composite_PS_eMx.png")

# ---- supplementary: population-level deficit (% of households) ----
pop = population_deficit_percent(freq, P.households_total)
fig2, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
im = heat(ax, pop, extent, vmax=100)
style_axes(ax)
ax.set_xlabel("Municipal water availability (M)", fontsize=FONT)
ax.set_ylabel("Extraction Rate (e)", fontsize=FONT)
cb = fig2.colorbar(im, ax=ax, location="right", shrink=0.95, pad=0.02)
cb.set_label("% of total households affected", fontsize=FONT)
save(fig2, f"{args.out}/PS-eM-total.png")
if args.show:
    plt.show()
