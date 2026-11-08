"""Dot strip plot of every stress match's final points, one row per config. Usage: python plot_points.py OUT.png"""
import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

d = json.load(open(Path(__file__).parent / "points.json"))
ROWS = [
    ("Before the merge, judge 6.5-13 s a call", None),
    ("main, v4 plan", "main_v4"),
    ("hybrid, hybrid plan", "hybrid_v4"),
    ("start-pose guard, v4 plan *", "guard_v4"),
    ("Fast judge, 4-6 s a call", None),
    ("hybrid, hybrid plan", "hybrid_v4_fast"),
    ("hybrid, long plan", "hybrid_long"),
    ("final, v4 plan + live guard", "final_v4_fast_live"),
    ("final, hybrid plan + live guard", "final_hybrid_fast_live"),
    ("final-harness as pushed, 6.5-13 s, 8 scenarios only", None),
    ("v4 plan", "build_v4"),
    ("hybrid plan", "build_hybrid"),
]
DOT, LOST, INK, INK2, MUTED, GRID, AXIS, SURF = ("#2a78d6", "#d03b3b", "#0b0b0b", "#52514e", "#898781",
                                                 "#e1e0d9", "#c3c2b7", "#fcfcfb")
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(11, 8.2), dpi=160)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)

y, ticks = 0.0, []
for label, key in ROWS:
    if key is None:
        y -= 0.55
        ax.text(-2, y, label, fontsize=10.5, color=INK2, fontweight="bold", ha="left", va="center",
                transform=ax.get_yaxis_transform() if False else ax.transData)
        y -= 0.75
        continue
    rows = d[key]
    by = collections.defaultdict(list)
    for p, lost, _ in rows:
        by[p].append(lost)
    ax.axhline(y, color=GRID, lw=0.8, zorder=0)
    for p, losts in by.items():
        losts.sort(reverse=True)
        n = len(losts)
        h = min(n * 0.028, 0.72)
        for i, lost in enumerate(losts):
            yy = y if n == 1 else y + h / 2 - h * i / (n - 1)
            if lost:
                ax.scatter(p, yy, marker="x", s=22, color=LOST, linewidths=1.4, zorder=3)
            else:
                ax.scatter(p, yy, s=9, color=DOT, alpha=0.6, linewidths=0, zorder=2)
    n = len(rows)
    nlost = sum(r[1] for r in rows)
    mean = sum(r[0] for r in rows) / n
    ax.plot([mean, mean], [y - 0.42, y + 0.42], color=INK, lw=2, zorder=4, solid_capstyle="butt")
    ax.text(124, y, f"{mean:.1f}", fontsize=11, color=INK, fontweight="bold", va="center", ha="center")
    ticks.append((y, label, f"{n} matches" + (f", {nlost} lost" if nlost else "")))
    y -= 1.0

ax.set_yticks([t[0] for t in ticks])
ax.set_yticklabels([f"{t[1]}\n{t[2]}" for t in ticks], fontsize=10, color=INK)
for lbl in ax.get_yticklabels():
    lbl.set_linespacing(1.25)
ax.tick_params(axis="y", length=0, pad=8)
ax.set_xlim(-4, 128)
ax.set_ylim(y + 0.4, 0.3)
ax.set_xticks(range(0, 121, 20))
ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
ax.grid(axis="x", color=GRID, lw=0.8, ls=(0, (2, 3)))
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.set_xlabel("Final points per match", color=INK2, fontsize=11, labelpad=8)
ax.text(124, 0.15, "mean", fontsize=10, color=MUTED, ha="center", va="bottom")
# section headers sit at the left edge of the axes, over the tick labels
for t in ax.texts:
    if t.get_text() in [r[0] for r in ROWS if r[1] is None]:
        t.set_x(-0.33)
        t.set_transform(matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))

fig.suptitle("Hill Climb stress sim: final points of every match", x=0.02, ha="left", fontsize=15, color=INK,
             fontweight="bold", y=0.985)
fig.text(0.02, 0.945, "Each dot is one simulated 5-minute match (virtual clock, 37 scenarios x 5 seeds unless noted). "
         "Red x = a match we'd lose (DQ or 8-ball in the bin).",
         fontsize=10, color=INK2, ha="left")
fig.text(0.02, 0.015, "* per-poll and live guard runs were identical at this judge speed (no ball stage reached), "
         "shown as one row.   Bottom rows: 8 hard scenarios only, not comparable to the 185-match rows.",
         fontsize=9, color=MUTED, ha="left")
fig.subplots_adjust(left=0.27, right=0.97, top=0.91, bottom=0.1)
fig.savefig(sys.argv[1], facecolor=SURF)
print("saved", sys.argv[1])
