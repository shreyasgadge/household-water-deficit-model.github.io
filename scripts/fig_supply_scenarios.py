"""Supply scenario visualisation: binary supply curves S_i(t), simultaneous vs sequential supply.

Main code - "Supply Scenario visualisation". One week, stacked tracks per cluster.
Supply durations default to the (rounded) cluster means.
"""
import matplotlib.pyplot as plt
import numpy as np

from _common import parse_args
from supplymodel.plotting import CLUSTER_COLORS, save

args, P = parse_args(__doc__)
durations = np.round(P.supply_days, 1)
labels = P.labels
scenarios = {
    "Simultaneous supply": [0, 0, 0],
    "Sequential supply (order 1 \u2192 2 \u2192 3)": [0, 1.5, 3],   # start days used for the schematic
}
t = np.linspace(0, 7, 4000)


def binary_signal(t, start, duration):
    """1 while supply is on; wraps around day 7 -> 0 if it runs past the week."""
    end = start + duration
    if end <= 7:
        return ((t >= start) & (t < end)).astype(float)
    return (((t >= start) & (t < 7)) | (t < end - 7)).astype(float)


offset_gap = 1.4
fig, axes = plt.subplots(len(scenarios), 1, figsize=(20, 6), sharex=True)
for ax, (title, start_days) in zip(axes, scenarios.items()):
    for i, (start, dur, color, label) in enumerate(zip(start_days, durations, CLUSTER_COLORS, labels)):
        y = binary_signal(t, start, dur)
        offset = (len(durations) - 1 - i) * offset_gap
        ax.step(t, y + offset, where="post", color=color, linewidth=2, label=label)
        ax.fill_between(t, offset, y + offset, step="post", color=color, alpha=0.25)
    ax.set_yticks([(len(durations) - 1 - i) * offset_gap + 0.5 for i in range(len(durations))])
    ax.set_yticklabels(labels, fontsize=18)
    ax.set_title(title, fontsize=18, loc="left")
    ax.set_xlim(0, 7)
    ax.set_xticks(np.arange(0, 8, 1))
    ax.tick_params(axis="x", labelsize=18)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
axes[-1].set_xlabel("Day of week", fontsize=18)
plt.tight_layout()
save(fig, f"{args.out}/binary_curves.png")
if args.show:
    plt.show()
