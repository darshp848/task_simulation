"""Build every data figure in the paper from the logged results (no new runs).

Sources (paths relative to the task_simulation root):
  data/station_nuc_20261108_late/robots-inference.jsonl   station Dream calls, 7-8 Nov
  results/stress/final_20261108/points.json               offline stress sim, per match
  results/judge_bench_20261108/bench_fair/*/summary.csv   judge benchmark
  FT diagnosis numbers are copied from the FT_DIAGNOSIS_2026-11-08 report (hill-climb-harness ft-diagnosis branch).
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TS = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, MUTED = "#9a9a94", "#1f1f1e", "#5c5c58"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3de",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


# ---------------------------------------------------------------- latency
def fig_latency():
    rows = [json.loads(l) for l in open(os.path.join(
        TS, "data/station_nuc_20261108_late/robots-inference.jsonl")) if l.strip()]
    inf = [r for r in rows if r.get("kind") == "inference" and r.get("rtt_ms")]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw={"width_ratios": [1.5, 1]})
    for mode, c in (("async", BLUE), ("sync", ORANGE)):
        v = np.sort([r["rtt_ms"] / 1000 for r in inf if r.get("mode") == mode])
        ax[0].plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=2,
                   label=f"{mode} chunks (n={len(v)})")
    v = np.sort([r["rtt_ms"] / 1000 for r in inf])
    ax[0].plot(v, np.arange(1, len(v) + 1) / len(v), color=INK, lw=1.2, ls="--",
               label=f"all (n={len(v)})")
    ax[0].axvline(0.9, color=MUTED, lw=1, ls=":")
    ax[0].text(0.93, 0.05, "0.9 s pacing\nthreshold", color=MUTED, fontsize=7)
    ax[0].set_xscale("log")
    ax[0].set_xticks([0.5, 1, 2, 4, 8]); ax[0].set_xticklabels(["0.5", "1", "2", "4", "8"])
    ax[0].set_xlabel("Dream-1 round trip (s, log)"); ax[0].set_ylabel("CDF")
    ax[0].legend(frameon=False, loc="lower right")
    ax[0].set_title("(a) Round trip by pacing mode")

    def med(k):
        return np.nanmedian([r.get(k) or np.nan for r in inf])
    serve, net, gw = med("serve_infer_ms"), med("network_ms"), med("gateway_ms")
    parts = [("model serve", serve, BLUE), ("gateway overhead", gw - serve, ORANGE),
             ("network", net, AQUA)]
    for i, (name, val, c) in enumerate(parts):
        ax[1].barh([2 - i], [val], color=c, height=0.55)
        ax[1].text(val + 15, 2 - i, f"{val:.0f} ms", va="center", fontsize=7, color=INK)
    ax[1].set_yticks([2, 1, 0]); ax[1].set_yticklabels([p[0] for p in parts])
    ax[1].set_xlim(0, 900)
    ax[1].set_xlabel("median per call (ms)"); ax[1].grid(axis="y", visible=False)
    ax[1].set_title("(b) Where the median goes")
    fig.savefig(os.path.join(OUT, "latency.pdf"))
    plt.close(fig)
    return len(inf)


# ---------------------------------------------------------------- stress
CONFIGS = [
    ("main_v4", "main, v4 plan"),
    ("hybrid_v4", "hybrid plan"),
    ("hybrid_v4_fast", "hybrid, fast judge"),
    ("hybrid_long", "hybrid long, fast judge"),
    ("final_v4_fast_live", "final, v4, fast + live guard"),
    ("final_hybrid_fast_live", "final, hybrid, fast + live guard"),
]


def fig_stress():
    d = json.load(open(os.path.join(TS, "results/stress/final_20261108/points.json")))
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True,
                           gridspec_kw={"width_ratios": [1.4, 1]})
    y = np.arange(len(CONFIGS))[::-1]
    rows = []
    for yi, (k, name) in zip(y, CONFIGS):
        pts = [m[0] for m in d[k]]
        lost8 = sum(1 for m in d[k] if m[1] == 1)
        mu, lo, hi = boot_ci(pts)
        live = "live" in k
        c = BLUE if live else GRAY
        ax[0].plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax[0].plot([mu], [yi], "o", ms=6, color=c, mec="white", mew=1.5)
        ax[0].text(hi + 1, yi, f"{mu:.1f}", va="center", fontsize=7, color=INK)
        rows.append((name, len(pts), mu, lo, hi, lost8))
    ax[0].set_yticks(y); ax[0].set_yticklabels([n for _, n in CONFIGS])
    ax[0].set_xlabel("points per match (mean, 95% bootstrap CI)")
    ax[0].set_title("(a) Score, 37 scenarios x 5 seeds")
    ax[0].set_xlim(45, 82)
    # losses: flag 1 = the referee's loss flag (DQ or 8-ball). Separate numbers from FINAL_HARNESS.md.
    eight = {"main_v4": 0, "hybrid_v4": 1, "hybrid_v4_fast": 28, "hybrid_long": 34,
             "final_v4_fast_live": 0, "final_hybrid_fast_live": 0}
    dq = {"main_v4": 4, "hybrid_v4": 4, "hybrid_v4_fast": 4, "hybrid_long": 5,
          "final_v4_fast_live": 6, "final_hybrid_fast_live": 4}
    for yi, (k, _) in zip(y, CONFIGS):
        ax[1].barh(yi + 0.17, dq[k], height=0.32, color=GRAY, edgecolor="white", linewidth=1)
        ax[1].barh(yi - 0.17, eight[k], height=0.32, color=ORANGE, edgecolor="white", linewidth=1)
        ax[1].text(eight[k] + 0.6, yi - 0.17, str(eight[k]), va="center", fontsize=7)
        ax[1].text(dq[k] + 0.6, yi + 0.17, str(dq[k]), va="center", fontsize=7, color=MUTED)
    ax[1].set_xlabel("matches lost (of 185)")
    ax[1].set_title("(b) Losses")
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=GRAY, label="DQ"), Patch(color=ORANGE, label="8-ball binned")],
                 frameon=False, loc="lower right")
    fig.savefig(os.path.join(OUT, "stress.pdf"))
    plt.close(fig)
    return rows


SCEN_ORDER = None


def fig_scenarios():
    d = json.load(open(os.path.join(TS, "results/stress/final_20261108/points.json")))
    keys = ["main_v4", "hybrid_v4", "hybrid_v4_fast", "final_hybrid_fast_live"]
    labels = ["main v4", "hybrid", "hybrid fast", "final hybrid\nfast + live"]
    scen = sorted({m[2] for m in d["main_v4"]})
    M = np.zeros((len(scen), len(keys)))
    for j, k in enumerate(keys):
        for i, s in enumerate(scen):
            M[i, j] = np.mean([m[0] for m in d[k] if m[2] == s])
    order = np.argsort(-M[:, -1])
    M = M[order]; scen = [scen[i] for i in order]
    fig, ax = plt.subplots(figsize=(3.3, 6.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=110, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.8,
                    color="white" if M[i, j] > 65 else INK)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=6.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(scen))); ax.set_yticklabels([s.replace("_", " ") for s in scen], fontsize=6)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("mean points (5 seeds)", fontsize=7); cb.ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(OUT, "scenarios.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- judges
def fig_judges():
    pts = []
    for f in glob.glob(os.path.join(TS, "results/judge_bench_20261108/bench_fair/*/summary.csv")):
        r = next(csv.DictReader(open(f)))
        if float(r["transport_failure_rate"]) > 0.3:
            continue  # gpt-6.1-sol rnone: every call failed
        name = os.path.basename(os.path.dirname(f))
        pts.append((name, float(r["stage_done_accuracy"]), float(r["guard_accuracy"]),
                    float(r["latency_p50_s"]), float(r["latency_p90_s"])))
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.5), sharex=True)
    hi = {"openai_gpt-6.1-terra_rnone": "GPT-6.1 Terra (none)",
          "openai_gpt-6.1-terra_rlow": "GPT-6.1 Terra (low)",
          "anthropic_claude-haiku-5.5_rlow": "Haiku 5.5 (low)",
          "subscription_claude-sonnet-5-5": "Sonnet 5.5 (CLI)",
          "google_gemini-3.1-flash-lite_rnone": "Gemini 3.1 Flash-Lite"}
    marks = {"openai_gpt-6.1-terra_rnone": "s", "openai_gpt-6.1-terra_rlow": "D",
             "anthropic_claude-haiku-5.5_rlow": "^", "subscription_claude-sonnet-5-5": "v",
             "google_gemini-3.1-flash-lite_rnone": "P"}
    cols = {"openai_gpt-6.1-terra_rnone": BLUE, "openai_gpt-6.1-terra_rlow": BLUE,
            "anthropic_claude-haiku-5.5_rlow": ORANGE, "subscription_claude-sonnet-5-5": ORANGE,
            "google_gemini-3.1-flash-lite_rnone": AQUA}
    for a, idx, title in ((ax[0], 1, "(a) Stage-done accuracy"), (ax[1], 2, "(b) Illegal-ball guard accuracy")):
        for p in sorted(pts, key=lambda p: p[0] in hi):
            k = p[0] in hi
            c = cols.get(p[0], GRAY)
            a.plot([p[3], p[4]], [p[idx] * 100] * 2, color=c, lw=1, alpha=0.6)
            a.plot(p[3], p[idx] * 100, marks.get(p[0], "o"), ms=6 if k else 3.5, color=c,
                   mec="white", mew=0.8, label=hi[p[0]] if (k and idx == 1) else None)
        a.set_xscale("log"); a.set_xticks([2, 4, 8, 16, 32]); a.set_xticklabels(["2", "4", "8", "16", "32"])
        a.set_xlabel("latency p50 (marker) to p90 (line end), s")
        a.set_title(title)
    fig.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.17), fontsize=7)
    ax[0].set_ylabel("accuracy (%)")
    ax[0].set_ylim(50, 103); ax[1].set_ylim(0, 105)
    fig.savefig(os.path.join(OUT, "judges.pdf"))
    plt.close(fig)
    return pts


# ---------------------------------------------------------------- fine-tune
def fig_finetune():
    # FT_DIAGNOSIS_2026-11-08.md, "Cause 1" table: median tool offset norm (mm), slow frames.
    folders = [("quest-split-v4-final", 55.9, "Quest"), ("unplug-split-v1-final", 37.2, "Quest"),
               ("unplug-v1-split", 50.9, "Quest"), ("leader-split-v1", 8.0, "Leader"),
               ("leader-split-v2", 6.1, "Leader"), ("leader-split-v4-final", 5.5, "Leader"),
               ("leader-split-v6", 7.8, "Leader"), ("plugin-split-v1-final", 7.6, "Leader")]
    # "What the live runs do": min tool z on the unplug prompt (cm).
    base = [6.0, 4.0, 0.0]
    base7 = [2.1]  # 7 Nov median only
    ft = [8.3, 9.9, 12.1, 12.6, 12.8, 13.2, 11.5]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.5, 1]})
    y = np.arange(len(folders))[::-1]
    for yi, (n, v, src) in zip(y, folders):
        ax[0].barh(yi, v, color=ORANGE if src == "Quest" else BLUE, height=0.6,
                   edgecolor="white", linewidth=2)
        ax[0].text(v + 1, yi, f"{v:.1f}", va="center", fontsize=7)
    ax[0].set_yticks(y); ax[0].set_yticklabels([f[0] for f in folders], fontsize=7)
    ax[0].set_xlabel("median |action - state| at the tool, slow frames (mm)"); ax[0].set_xlim(0, 66)
    ax[0].set_title("(a) Steady-state command offset per dataset")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=ORANGE, label="Quest (no gravity FF)"),
                          Patch(color=BLUE, label="Leader arm (gravity FF)")], frameon=False, loc="lower right")
    rng = np.random.default_rng(1)
    for x, vals, c in ((0, base, BLUE), (1, ft, ORANGE)):
        xs = x + rng.uniform(-0.12, 0.12, len(vals))
        ax[1].plot(xs, vals, "o", color=c, ms=6, mec="white", mew=1.5)
        ax[1].plot([x - 0.22, x + 0.22], [np.median(vals)] * 2, color=INK, lw=1.5)
    ax[1].axhspan(4, 6, color=AQUA, alpha=0.18, lw=0)
    ax[1].text(1.45, 5, "demo grasp\nheight", fontsize=6.5, color=MUTED, va="center", ha="right")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["base Dream-1\n(8 Nov, n=3)", "fine-tune\n(n=7)"])
    ax[1].set_xlim(-0.5, 1.5)
    ax[1].set_ylabel("min tool height (cm)")
    ax[1].set_title("(b) Live reach on the station")
    fig.savefig(os.path.join(OUT, "finetune.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    n = fig_latency()
    print("latency calls", n)
    for r in fig_stress():
        print("stress", r)
    fig_scenarios()
    for p in sorted(fig_judges(), key=lambda p: p[3]):
        print("judge", p)
    fig_finetune()
    print("figures in", OUT)
