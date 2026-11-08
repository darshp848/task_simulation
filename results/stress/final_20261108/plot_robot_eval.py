"""Stress results drawn the way robot-learning evaluation papers draw them.

1. Milestone success rates as Beta posteriors (uniform prior) per config, with compact letter display from pairwise
   Fisher exact tests, Bonferroni-corrected (TRI LBM team 2025, arXiv 2507.05331).
2. Match score: IQM and mean with stratified-bootstrap 95% CIs (strata = scenarios), and performance profiles
   (Agarwal et al. 2021, rliable).
3. Outcome grid per initial condition (scenario x seed), coloured by the furthest milestone reached
   (Kress-Gazit et al. 2024, arXiv 2409.09491).

A match we'd lose (DQ or 8-ball in the bin) counts as a loss: score 0, no milestones.
Usage: python plot_robot_eval.py OUTDIR
"""
import glob
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).parent
OUT = Path(sys.argv[1])
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ data: one record per match
runs = defaultdict(list)
for f in glob.glob(str(HERE / "out" / "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        lost = bool(r["lost"])
        u = 10 + 5 * (r["out"] - 1) if r["out"] >= 1 else 0
        t = r["points"] - u - 20 * min(r["plugged"], 4) - 5 * r["balls"]
        runs[r["label"]].append(dict(
            scenario=r["scenario"], seed=r["seed"], lost=lost,
            score=0 if lost else r["points"],
            unplug=(not lost) and r["out"] >= 1,
            tower=(not lost) and t >= 20,
            ball=(not lost) and r["balls"] >= 1,
            kept=not lost))

CONFIGS = [  # label, key, group
    ("main, v4", "main_v4", "Judge 6.5-13 s"),
    ("hybrid, hybrid plan", "hybrid_v4", "Judge 6.5-13 s"),
    ("start-pose guard, v4 *", "guard_v4", "Judge 6.5-13 s"),
    ("hybrid, hybrid plan", "hybrid_v4_fast", "Fast judge 4-6 s"),
    ("hybrid, long plan", "hybrid_long", "Fast judge 4-6 s"),
    ("final, v4 + live guard", "final_v4_fast_live", "Fast judge 4-6 s"),
    ("final, hybrid + live guard", "final_hybrid_fast_live", "Fast judge 4-6 s"),
    ("pushed, v4 plan", "build_v4", "final-harness branch, 8 hard scenarios"),
    ("pushed, hybrid plan", "build_hybrid", "final-harness branch, 8 hard scenarios"),
]
MAIN = [c for c in CONFIGS if not c[1].startswith("build")]   # same 37 scenarios: comparable with each other
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CRIT = "#d03b3b"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def style(ax, grid="x"):
    ax.set_facecolor(SURF)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def ypos(cfgs):
    ys, y, prev = [], 0.0, None
    for c in cfgs:
        if prev is not None and c[2] != prev:
            y += 0.6
        ys.append(y)
        prev = c[2]
        y += 1.0
    return ys


def group_labels(ax, cfgs, ys, x=-0.02):
    seen = set()
    for c, y in zip(cfgs, ys):
        if c[2] not in seen:
            seen.add(c[2])
            ax.text(x, y - 0.62, c[2], transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9,
                    color=INK2, fontweight="bold")


def cld(names, means, sig):
    """Compact letter display (insert-absorb). sig: set of frozenset({a, b}) pairs that differ."""
    groups = [set(names)]
    for pair in sig:
        a, b = tuple(pair)
        new = []
        for g in groups:
            if a in g and b in g:
                new += [g - {a}, g - {b}]
            else:
                new.append(g)
        groups = [g for g in new if not any(g < h for h in new)]
        groups = [g for i, g in enumerate(groups) if g not in groups[:i]]
    groups.sort(key=lambda g: -max(means[n] for n in g))
    letters = {n: "" for n in names}
    for i, g in enumerate(groups):
        for n in g:
            letters[n] += "abcdefghij"[i]
    return letters


# ------------------------------------------------------------------ figure 1: milestone success rates
MILESTONES = [("unplug", "Unplug: 1+ cable out"), ("tower", "Tower over 4 in, stands 3 s"),
              ("ball", "1+ ball in the bin"), ("kept", "Match not lost (no DQ, no 8-ball)")]
cfgs = CONFIGS
ys = ypos(cfgs)
fig, axes = plt.subplots(1, 4, figsize=(16, 8.4), dpi=150, sharey=True)
fig.patch.set_facecolor(SURF)
grid_p = np.linspace(0, 1, 801)
for j, ((mk, mlab), ax) in enumerate(zip(MILESTONES, axes)):
    style(ax)
    col = SERIES[j]
    k_n = {}
    for (lab, key, _), y in zip(cfgs, ys):
        rs = runs[key]
        k, n = sum(r[mk] for r in rs), len(rs)
        k_n[key] = (k, n)
        dens = stats.beta.pdf(grid_p, k + 1, n - k + 1)
        dens = dens / dens.max() * 0.42
        m = dens > 0.004
        ax.fill_between(100 * grid_p[m], y - dens[m], y + dens[m], color=col, alpha=0.85, lw=0)
        mean = (k + 1) / (n + 2)
        ax.plot([100 * mean] * 2, [y - 0.42, y + 0.42], color=INK, lw=1.4)
        lo, hi = stats.beta.ppf([0.025, 0.975], k + 1, n - k + 1)
        ax.text(103, y, f"{k}/{n}", va="center", ha="left", fontsize=8.5, color=INK2)
    # significance among the 7 configs on the same 37 scenarios
    names = [c[1] for c in MAIN]
    pairs = list(combinations(names, 2))
    alpha = 0.05 / len(pairs)
    sig = set()
    for a, b in pairs:
        (ka, na), (kb, nb) = k_n[a], k_n[b]
        p = stats.fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
        if p < alpha:
            sig.add(frozenset((a, b)))
    letters = cld(names, {n: k_n[n][0] / k_n[n][1] for n in names}, sig)
    for (lab, key, _), y in zip(cfgs, ys):
        if key in letters:
            ax.text(141, y, letters[key], va="center", ha="right", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlim(0, 142)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title(mlab, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("success rate", fontsize=9, color=INK2)
axes[0].set_yticks(ys)
axes[0].set_yticklabels([c[0] for c in cfgs], fontsize=10, color=INK)
axes[0].set_ylim(ys[-1] + 0.7, -0.7)
group_labels(axes[0], cfgs, ys, x=-0.04)
fig.suptitle("Milestone success rates per config (stress sim)", x=0.01, ha="left", fontsize=15, color=INK,
             fontweight="bold", y=0.985)
fig.text(0.01, 0.93, "Violin = posterior of the success rate (Beta, uniform prior); black line = posterior mean; "
         "k/n = matches that reached the milestone. A match we'd lose reaches none.\nLetters: configs that share "
         "no letter differ at 95% (Fisher's exact test, Bonferroni over 21 pairs), among the 7 configs run on the "
         "same 37 scenarios. The bottom two rows ran 8 hard scenarios and are not lettered.",
         fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.4)
fig.text(0.01, 0.01, "Method after the TRI LBM team (2025, arXiv 2507.05331).   * per-poll and live guard gave "
         "identical results at this judge speed.", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.2, right=0.985, top=0.83, bottom=0.08, wspace=0.08)
fig.savefig(OUT / "eval_1_milestones.png", facecolor=SURF)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: score IQM/mean + performance profiles
B = 2000


def strat_boot(rs, fn):
    by = defaultdict(list)
    for r in rs:
        by[r["scenario"]].append(r["score"])
    arrs = [np.array(v, float) for v in by.values()]
    out = np.empty(B)
    for b in range(B):
        out[b] = fn(np.concatenate([a[rng.integers(0, len(a), len(a))] for a in arrs]))
    return out


def iqm(x):
    return stats.trim_mean(x, 0.25)


fig = plt.figure(figsize=(16, 9), dpi=150)
fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], height_ratios=[1, 1], wspace=0.35, hspace=0.45,
                      left=0.16, right=0.985, top=0.84, bottom=0.08)
ys = ypos(MAIN)
for j, (fn, name) in enumerate([(iqm, "IQM (mean of the middle 50%)"), (np.mean, "Mean")]):
    ax = fig.add_subplot(gs[:, j])
    style(ax)
    for (lab, key, _), y in zip(MAIN, ys):
        sc = np.array([r["score"] for r in runs[key]], float)
        est, boot = fn(sc), strat_boot(runs[key], fn)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.add_patch(Rectangle((lo, y - 0.3), hi - lo, 0.6, color=SERIES[0], alpha=0.35, lw=0))
        ax.plot([est, est], [y - 0.3, y + 0.3], color=INK, lw=2)
        ax.text(hi + 1.2, y, f"{est:.1f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in MAIN] if j == 0 else [], fontsize=10, color=INK)
    ax.set_ylim(ys[-1] + 0.7, -0.7)
    ax.set_xlim(30, 95)
    ax.set_title(name, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("match score (lost match = 0)", fontsize=9, color=INK2)
    if j == 0:
        group_labels(ax, MAIN, ys, x=-0.04)
taus = np.arange(0, 121, 1)
for row, grp in enumerate(["Judge 6.5-13 s", "Fast judge 4-6 s"]):
    ax = fig.add_subplot(gs[row, 2])
    style(ax, grid="both")
    handles = []
    styles = ["-", (0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1.5, 2))]
    for i, (lab, key, _) in enumerate([c for c in MAIN if c[2] == grp]):
        sc = np.array([r["score"] for r in runs[key]], float)
        prof = np.array([(sc > t).mean() for t in taus])
        by = defaultdict(list)
        for r in runs[key]:
            by[r["scenario"]].append(r["score"])
        arrs = [np.array(v, float) for v in by.values()]
        boots = np.empty((400, len(taus)))
        for b in range(400):
            s = np.concatenate([a[rng.integers(0, len(a), len(a))] for a in arrs])
            boots[b] = (s[:, None] > taus[None, :]).mean(0)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(taus, lo, hi, color=SERIES[i], alpha=0.15, lw=0)
        ax.plot(taus, prof, color=SERIES[i], lw=2, ls=styles[i])
        handles.append(Line2D([], [], color=SERIES[i], lw=2, ls=styles[i], label=lab))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    ax.set_title(f"Performance profile: {grp}", loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("score threshold τ", fontsize=9, color=INK2)
    ax.set_ylabel("matches scoring > τ", fontsize=9, color=INK2)
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2, handlelength=2.6)
fig.suptitle("Match score per config, with uncertainty (stress sim, 37 scenarios x 5 seeds)", x=0.01, ha="left",
             fontsize=15, color=INK, fontweight="bold", y=0.985)
fig.text(0.01, 0.93, "Left: point estimate (black) and 95% stratified-bootstrap CI (band, 2,000 resamples of seeds "
         "within each scenario). Right: share of matches scoring above each threshold; band = 95% CI.\nA higher "
         "curve is better at every threshold. A lost match (DQ or 8-ball in the bin) scores 0, as it would in "
         "the real match.", fontsize=9.5, color=INK2, ha="left", va="top", linespacing=1.4)
fig.text(0.01, 0.01, "Method after Agarwal et al. (2021), 'Deep RL at the Edge of the Statistical Precipice' "
         "(rliable).   * per-poll and live guard gave identical results at this judge speed.",
         fontsize=8.5, color=MUTED)
fig.savefig(OUT / "eval_2_score_ci_profiles.png", facecolor=SURF)
plt.close(fig)

# ------------------------------------------------------------------ figure 3: outcome per scenario x seed
LEVELS = [("Lost (DQ or 8-ball)", CRIT), ("No cable out", "#e1e0d9"), ("Unplug only", "#b7d3f6"),
          ("Tower passed", "#5598e7"), ("Ball(s) in bin", "#184f95")]


def level(r):
    if r["lost"]:
        return 0
    if r["ball"]:
        return 4
    if r["tower"]:
        return 3
    if r["unplug"]:
        return 2
    return 1


scen = sorted({r["scenario"] for c in CONFIGS for r in runs[c[1]]})
# order scenarios by how far the best-known config gets, hardest at the bottom
avg = {s: np.mean([level(r) for c in MAIN for r in runs[c[1]] if r["scenario"] == s]) for s in scen}
scen.sort(key=lambda s: -avg[s])
fig, ax = plt.subplots(figsize=(15, 15.5), dpi=140)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
cw, gap = 5 * 0.9, 1.2
for ci, (lab, key, _) in enumerate(CONFIGS):
    x0 = ci * (cw + gap)
    cell = {(r["scenario"], r["seed"]): level(r) for r in runs[key]}
    for si, s in enumerate(scen):
        seeds = sorted(sd for (sc, sd) in cell if sc == s)
        for k, sd in enumerate(seeds):
            lv = cell[(s, sd)]
            ax.add_patch(Rectangle((x0 + k * 0.9, si + 0.08), 0.78, 0.84, color=LEVELS[lv][1], lw=0))
            if lv == 0:
                ax.plot([x0 + k * 0.9 + 0.15, x0 + k * 0.9 + 0.63], [si + 0.25, si + 0.75], color="#ffffff", lw=1)
                ax.plot([x0 + k * 0.9 + 0.63, x0 + k * 0.9 + 0.15], [si + 0.25, si + 0.75], color="#ffffff", lw=1)
    ax.text(x0 + cw / 2, -0.6, lab, rotation=35, ha="left", va="bottom", fontsize=9.5, color=INK,
            rotation_mode="anchor")
ax.set_xlim(-0.3, len(CONFIGS) * (cw + gap))
ax.set_ylim(len(scen) + 0.2, -0.2)
ax.set_yticks([i + 0.5 for i in range(len(scen))])
ax.set_yticklabels(scen, fontsize=9, color=INK)
ax.set_xticks([])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.legend(handles=[Patch(color=c, label=l) for l, c in LEVELS], loc="upper left", bbox_to_anchor=(0.01, 0.955),
           ncol=5, frameon=False, fontsize=10, labelcolor=INK2, handlelength=1.2, handleheight=1.2)
fig.suptitle("Outcome of every match: scenario (initial condition) x seed, per config", x=0.01, ha="left",
             fontsize=15, color=INK, fontweight="bold", y=0.99)
fig.text(0.01, 0.965, "Each square is one match (5 seeds per scenario); colour = furthest milestone reached. Rows "
         "sorted from easiest (top) to hardest. Blank = not run (the pushed-branch columns ran 8 scenarios).",
         fontsize=10, color=INK2, ha="left")
fig.text(0.01, 0.005, "Layout after Kress-Gazit et al. (2024), arXiv 2409.09491, Fig. 2 (repeated rollouts per "
         "initial condition).", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.18, right=0.99, top=0.83, bottom=0.02)
fig.savefig(OUT / "eval_3_outcome_grid.png", facecolor=SURF)
plt.close(fig)
print("saved 3 figures to", OUT)
