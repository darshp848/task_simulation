"""Histograms of every stress match: total points per config (with a normal fit) and each task's points.
Usage: python plot_dist.py TOTAL.png TASKS.png"""
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).parent
runs = defaultdict(list)
for f in glob.glob(str(HERE / "out" / "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r["dq"]:
            parts = dict(unplug=0, tower_pass=0, tower_h=0, balls=0)
        else:
            u = 10 + 5 * (r["out"] - 1) if r["out"] >= 1 else 0
            b = 5 * r["balls"]
            t = r["points"] - u - 20 * min(r["plugged"], 4) - b
            parts = dict(unplug=u, tower_pass=20 if t >= 20 else 0, tower_h=t - 20 if t >= 20 else t, balls=b)
        runs[r["label"]].append(dict(points=r["points"], lost=bool(r["lost"]), **parts))

ROWS = [
    ("main, v4 plan", "main_v4", "judge 6.5-13 s"),
    ("hybrid, hybrid plan", "hybrid_v4", "judge 6.5-13 s"),
    ("start-pose guard, v4 plan *", "guard_v4", "judge 6.5-13 s"),
    ("hybrid, hybrid plan", "hybrid_v4_fast", "fast judge 4-6 s"),
    ("hybrid, long plan", "hybrid_long", "fast judge 4-6 s"),
    ("final, v4 + live guard", "final_v4_fast_live", "fast judge 4-6 s"),
    ("final, hybrid + live guard", "final_hybrid_fast_live", "fast judge 4-6 s"),
    ("final-harness pushed, v4 plan", "build_v4", "6.5-13 s, 8 hard scenarios"),
    ("final-harness pushed, hybrid plan", "build_hybrid", "6.5-13 s, 8 hard scenarios"),
]
DOT, LOST, INK, INK2, MUTED, GRID, AXIS, SURF = ("#2a78d6", "#d03b3b", "#0b0b0b", "#52514e", "#898781",
                                                 "#e1e0d9", "#c3c2b7", "#fcfcfb")
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
plt.rcParams["hatch.color"] = "#ffffff"
plt.rcParams["hatch.linewidth"] = 0.9


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


# ---------------------------------------------------------------- 1. total points, one histogram per config
W = 5
edges = list(range(0, 125, W))
fig, axes = plt.subplots(3, 3, figsize=(15, 11), dpi=150, sharex=True)
fig.patch.set_facecolor(SURF)
for ax, (lab, key, sec) in zip(axes.flat, ROWS):
    style(ax)
    rs = runs[key]
    n = len(rs)
    kept = [0] * len(edges)
    lost = [0] * len(edges)
    for r in rs:
        i = min(int(r["points"] // W), len(edges) - 1)
        (lost if r["lost"] else kept)[i] += 1
    kp = [100 * c / n for c in kept]
    lp = [100 * c / n for c in lost]
    ax.bar(edges, kp, width=W * 0.86, align="edge", color=DOT)
    ax.bar(edges, lp, width=W * 0.86, align="edge", bottom=kp, color=LOST, hatch="////")
    pts = [r["points"] for r in rs]
    mu, sd = statistics.mean(pts), statistics.pstdev(pts)
    xs = [x / 2 for x in range(0, 250)]
    ax.plot(xs, [100 * W / (sd * math.sqrt(2 * math.pi)) * math.exp(-0.5 * ((x - mu) / sd) ** 2) for x in xs],
            color=INK, lw=1.4, ls=(0, (4, 3)))
    ax.axvline(mu, color=INK, lw=1.6)
    med = statistics.median(pts)
    nl = sum(r["lost"] for r in rs)
    ax.set_title(lab, loc="left", fontsize=11, color=INK, pad=16)
    ax.text(0, 1.02, f"{sec}  |  {n} matches", transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.text(0.98, 0.95, f"mean {mu:.1f}\nmedian {med:.0f}\nsd {sd:.1f}\nlost {nl}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK2, linespacing=1.35)
    ax.set_xlim(-2, 122)
    ax.set_ylim(0, max(32, max(a + b for a, b in zip(kp, lp)) * 1.12))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
for ax in axes[-1]:
    ax.set_xlabel("Final points (5-point bins)", color=INK2, fontsize=10)
fig.legend(handles=[Patch(color=DOT, label="Matches kept"),
                    Patch(facecolor=LOST, hatch="////", edgecolor="#ffffff", label="Matches we'd lose (DQ or 8-ball)"),
                    Line2D([], [], color=INK, lw=1.6, label="Mean"),
                    Line2D([], [], color=INK, lw=1.4, ls=(0, (4, 3)), label="Normal curve with the same mean and sd")],
           loc="upper left", bbox_to_anchor=(0.012, 0.948), ncol=4, frameon=False, fontsize=9.5, labelcolor=INK2,
           handlelength=1.8)
fig.suptitle("Hill Climb stress sim: distribution of final points, every match", x=0.012, ha="left", fontsize=15,
             color=INK, fontweight="bold", y=0.99)
fig.text(0.012, 0.955, "Bar height = share of that config's matches. Not normal: a spike at 0-20 (cable never comes "
         "out, arm lost, API down, DQ), then a wide hump at 40-100.", fontsize=10, color=INK2, ha="left")
fig.text(0.012, 0.008, "* per-poll and live guard gave identical results at this judge speed.   Bottom row: 8 hard "
         "scenarios only, so compare those two with each other.", fontsize=8.5, color=MUTED, ha="left")
fig.subplots_adjust(left=0.045, right=0.985, top=0.87, bottom=0.07, hspace=0.42, wspace=0.12)
fig.savefig(sys.argv[1], facecolor=SURF)
print("saved", sys.argv[1])

# ---------------------------------------------------------------- 2. each task, every config
TASKS = [("unplug", "Unplug points", "#2a78d6", [0, 10, 15, 20, 25]),
         ("tower_pass", "Tower over 4 in", "#eb6834", [0, 20]),
         ("tower_h", "Tower height (+3/in)", "#1baf7a", None),
         ("balls", "Balls in bin (+5 each)", "#eda100", None)]
fig, axes = plt.subplots(len(ROWS), 4, figsize=(15, 17), dpi=140, sharex="col")
fig.patch.set_facecolor(SURF)
for j, (tk, tlab, col, cats) in enumerate(TASKS):
    vals_all = sorted({r[tk] for k in (r[1] for r in ROWS) for r in runs[k]})
    xs = cats or vals_all
    for i, (lab, key, sec) in enumerate(ROWS):
        ax = axes[i, j]
        style(ax)
        rs = runs[key]
        n = len(rs)
        cnt = defaultdict(int)
        for r in rs:
            cnt[r[tk]] += 1
        pos = list(range(len(xs)))
        hs = [100 * cnt[x] / n for x in xs]
        ax.bar(pos, hs, width=0.8, color=col)
        for p, h in zip(pos, hs):
            if h >= 1 and len(xs) <= 10:
                ax.text(p, h + 2, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 50, 100])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        ax.set_xticks(pos)
        ax.set_xticklabels([str(x) for x in xs], fontsize=7.5 if len(xs) > 10 else 8.5)
        mean = sum(r[tk] for r in rs) / n
        ax.text(0.99, 1.0, f"mean {mean:.1f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color=INK2)
        if j == 0:
            ax.set_ylabel(f"{lab}\n{sec}", rotation=0, ha="right", va="center", fontsize=9.5, color=INK,
                          labelpad=10)
        if i == 0:
            ax.set_title(tlab, loc="left", fontsize=12, color=INK, pad=10)
        if i == len(ROWS) - 1:
            ax.set_xlabel("points from this task", fontsize=9, color=INK2)
fig.suptitle("Hill Climb stress sim: points per task, distribution over every match", x=0.012, ha="left",
             fontsize=15, color=INK, fontweight="bold", y=0.997)
fig.text(0.012, 0.972, "Bar height = share of that config's matches scoring that many points on the task. "
         "A DQ match counts 0 on every task. Plugging (+20) scored 0 everywhere and is left out.",
         fontsize=10, color=INK2, ha="left")
fig.subplots_adjust(left=0.17, right=0.99, top=0.935, bottom=0.04, hspace=0.4, wspace=0.12)
fig.savefig(sys.argv[2], facecolor=SURF)
print("saved", sys.argv[2])
