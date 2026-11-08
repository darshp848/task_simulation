"""Stacked + small-multiple bar charts of mean points per task, per config. Usage: python plot_breakdown.py OUT.png"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

d = json.load(open(Path(__file__).parent / "breakdown.json"))
ROWS = [  # (label, key, section)
    ("main, v4 plan", "main_v4", "Judge 6.5-13 s"),
    ("hybrid, hybrid plan", "hybrid_v4", "Judge 6.5-13 s"),
    ("start-pose guard, v4 plan *", "guard_v4", "Judge 6.5-13 s"),
    ("hybrid, hybrid plan", "hybrid_v4_fast", "Fast judge 4-6 s"),
    ("hybrid, long plan", "hybrid_long", "Fast judge 4-6 s"),
    ("final, v4 + live guard", "final_v4_fast_live", "Fast judge 4-6 s"),
    ("final, hybrid + live guard", "final_hybrid_fast_live", "Fast judge 4-6 s"),
    ("pushed, v4 plan (8 scen.)", "build_v4", "final-harness, 6.5-13 s"),
    ("pushed, hybrid plan (8 scen.)", "build_hybrid", "final-harness, 6.5-13 s"),
]
TASKS = [  # key, label, colour, label ink
    ("unplug", "Unplug (+10, +5 each extra)", "#2a78d6", "#ffffff"),
    ("tower_pass", "Tower over 4 in (+20)", "#eb6834", "#ffffff"),
    ("tower_h", "Tower height (+3 per inch)", "#1baf7a", "#ffffff"),
    ("balls", "Balls in bin (+5 each)", "#eda100", "#0b0b0b"),
]
LOST_C = "#d3d1c7"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
plt.rcParams["hatch.color"] = "#888780"
plt.rcParams["hatch.linewidth"] = 0.8

# y positions with a gap between sections
ys, y, prev = [], 0.0, None
for _, _, sec in ROWS:
    if prev is not None and sec != prev:
        y += 0.55
    ys.append(y)
    prev = sec
    y += 1.0
H = 0.62

fig = plt.figure(figsize=(12, 11.5), dpi=160)
fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.22, wspace=0.12,
                      left=0.2, right=0.97, top=0.845, bottom=0.05)


def style(ax, xmax, step):
    ax.set_facecolor(SURF)
    ax.set_xlim(0, xmax)
    ax.set_ylim(ys[-1] + 0.6, -0.6)
    ax.set_xticks(range(0, int(xmax) + 1, step))
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)


# ---- stacked: the whole mean, split by task
ax = fig.add_subplot(gs[0, :])
style(ax, 80, 10)
for yy, (_, key, _) in zip(ys, ROWS):
    left = 0.0
    for tk, _, col, ink in TASKS:
        v = d[key][tk]
        if v <= 0:
            continue
        ax.barh(yy, v, left=left, height=H, color=col, edgecolor=SURF, linewidth=1.5)
        if v >= 6:
            ax.text(left + v / 2, yy, f"{v:.0f}", ha="center", va="center", fontsize=9, color=ink)
        left += v
    lv = d[key]["lost_pts"]
    if lv > 0.05:
        ax.barh(yy, lv, left=left, height=H, color=LOST_C, hatch="////", edgecolor=SURF, linewidth=1.5)
        if lv >= 6:
            ax.text(left + lv / 2, yy, f"{lv:.0f}", ha="center", va="center", fontsize=9, color=INK)
        left += lv
    ax.text(left + 1, yy, f"{left:.1f}", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=10, color=INK)
# section labels in the left margin
seen = set()
for yy, (_, _, sec) in zip(ys, ROWS):
    if sec not in seen:
        seen.add(sec)
        ax.text(-0.255, yy - 0.58, sec, transform=ax.get_yaxis_transform(), fontsize=9.5, color=INK2,
                fontweight="bold", ha="left", va="center")
ax.set_title("Mean points per match, split by task", loc="left", fontsize=12, color=INK, pad=34)
handles = [Patch(facecolor=c, label=lab) for _, lab, c, _ in TASKS]
handles.append(Patch(facecolor=LOST_C, hatch="////", edgecolor="#888780", label="Points in matches we'd lose (8-ball)"))
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, frameon=False, fontsize=9,
          handlelength=1.1, handleheight=1.1, columnspacing=1.4, labelcolor=INK2, borderaxespad=0.2)

# ---- small multiples: one task each, same rows
for i, (tk, lab, col, _) in enumerate(TASKS):
    a = fig.add_subplot(gs[1, i])
    xmax = {"unplug": 20, "tower_pass": 20, "tower_h": 35, "balls": 20}[tk]
    style(a, xmax, 5 if xmax <= 20 else 10)
    for yy, (_, key, _) in zip(ys, ROWS):
        v = d[key][tk]
        a.barh(yy, v, height=H, color=col)
        a.text(v + xmax * 0.02, yy, f"{v:.1f}", ha="left", va="center", fontsize=8.5, color=INK)
    a.set_yticks(ys)
    a.set_yticklabels([r[0] for r in ROWS] if i == 0 else [], fontsize=9, color=INK)
    a.set_title(lab, loc="left", fontsize=10.5, color=INK, pad=6)

fig.suptitle("Hill Climb stress sim: where the points come from", x=0.02, ha="left", fontsize=15, color=INK,
             fontweight="bold", y=0.985)
fig.text(0.02, 0.952, "Mean over every simulated match (185 per config: 37 scenarios x 5 seeds; the pushed-branch rows "
         "are 8 hard scenarios x 5 seeds, so compare them with each other only).",
         fontsize=10, color=INK2, ha="left")
fig.text(0.02, 0.937, "A DQ match counts 0 everywhere. Plugging a loose cable (+20) scored 0 in every config and is "
         "left out.", fontsize=10, color=INK2, ha="left")
fig.text(0.02, 0.012, "* per-poll and live guard gave identical results at this judge speed; one row.",
         fontsize=9, color=MUTED, ha="left")
fig.savefig(sys.argv[1], facecolor=SURF)
print("saved", sys.argv[1])
