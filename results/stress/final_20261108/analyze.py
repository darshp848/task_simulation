"""Aggregate out/*.jsonl: mean points / losses per scenario x config, plus per-config summaries."""
import collections
import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
rows = []
for f in glob.glob(str(Path(__file__).parent / "out" / "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        rows.append(json.loads(line))
labels = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted({r["label"] for r in rows})
by = collections.defaultdict(list)
for r in rows:
    by[(r["scenario"], r["label"])].append(r)
scen = sorted({r["scenario"] for r in rows if r["label"] in labels})


def cell(rs):
    if not rs:
        return "-"
    m = sum(r["points"] for r in rs) / len(rs)
    lost = sum(r["lost"] for r in rs)
    crash = sum(str(r["reason"]).startswith("CRASH") for r in rs)
    s = f"{m:5.1f}"
    if lost:
        s += f" L{lost}"
    if crash:
        s += f" C{crash}"
    return s


print("| scenario | " + " | ".join(labels) + " |")
print("|---|" + "---|" * len(labels))
for s in scen:
    print(f"| {s} | " + " | ".join(cell(by[(s, l)]) for l in labels) + " |")
print()
print("| config | matches | mean pts | lost (DQ / black) | crashes | unplug passed | tower >= 3 at end | mean balls | ended early |")
print("|---|---|---|---|---|---|---|---|---|")
for l in labels:
    rs = [r for r in rows if r["label"] == l]
    if not rs:
        continue
    n = len(rs)
    dq = sum(bool(r["dq"]) for r in rs)
    blk = sum(r["black_in_bin"] > 0 for r in rs)
    crash = sum(str(r["reason"]).startswith("CRASH") for r in rs)
    unp = sum(r.get("out", 0) >= 1 for r in rs)
    tow = sum(r.get("tower", 1) >= 3 for r in rs)
    balls = sum(r.get("balls", 0) for r in rs) / n
    early = sum(r["t_end"] < 280 for r in rs)
    print(f"| {l} | {n} | {sum(r['points'] for r in rs)/n:.1f} | {dq+blk} ({dq} / {blk}) | {crash} | {unp}/{n} | "
          f"{tow}/{n} | {balls:.2f} | {early} |")
if "--reasons" in sys.argv:
    for l in labels:
        c = collections.Counter(str(r["reason"])[:80] for r in rows if r["label"] == l)
        print(f"\n{l}:")
        for k, v in c.most_common(8):
            print(f"  {v:4d}  {k}")
