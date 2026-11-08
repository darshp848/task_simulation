"""Aggregate bench_judges results.json files: accuracy, latency, speed and cost per judge config, with 95% intervals.

    python analyze_judges.py RUN_DIR [RUN_DIR ...] [--md OUT.md] [--csv OUT.csv]

Latency = wall time of one judge decision (all hedged/raced copies until the winner). Speed = output tokens per
second of each successful model call (completion tokens / call time; includes time to first token, as the harness
sees it). Cost per match assumes a verdict every ~5 s over a 295 s match (~60 decisions).
"""
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
args = sys.argv[1:]
md_out = csv_out = None
if "--md" in args:
    md_out = args[args.index("--md") + 1]
    del args[args.index("--md"):args.index("--md") + 2]
if "--csv" in args:
    csv_out = args[args.index("--csv") + 1]
    del args[args.index("--csv"):args.index("--csv") + 2]
rows = []
for d in args:
    data = json.loads((Path(d) / "results.json").read_text(encoding="utf-8"))
    rows += data["trials"] if isinstance(data, dict) else data
rng = random.Random(0)


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def boot_mean(xs, b=2000):
    if len(xs) < 2:
        return (xs[0], xs[0], xs[0]) if xs else (None, None, None)
    ms = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(b))
    return statistics.fmean(xs), ms[int(0.025 * b)], ms[int(0.975 * b)]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


out = []
by = defaultdict(list)
for r in rows:
    by[r["judge"]].append(r)
for name, g in by.items():
    st = [r["stage_correct"] for r in g if r.get("stage_correct") is not None]
    gd = [r["guard_correct"] for r in g if r.get("guard_correct") is not None]
    held = [r["guard_correct"] for r in g if r.get("guard_correct") is not None and r.get("expected_guard") is True]
    lat = [r["latency_s"] for r in g if r.get("latency_s") is not None and not r.get("transport_failure")]
    calls = [a for r in g for a in r.get("attempts", []) if not a.get("error_type")]
    tps = [a["usage"]["completion_tokens"] / a["latency_s"] for a in calls
           if a.get("latency_s") and a.get("usage", {}).get("completion_tokens")]
    outtok = [a["usage"]["completion_tokens"] for a in calls if a.get("usage", {}).get("completion_tokens")]
    intok = [a["usage"]["prompt_tokens"] for a in calls if a.get("usage", {}).get("prompt_tokens")]
    costs = [a["cost_usd"] for a in calls if a.get("cost_usd") is not None]
    trial_cost = [r["known_cost_usd"] for r in g]
    a, alo, ahi = wilson(sum(st), len(st))
    ga, glo, ghi = wilson(sum(gd), len(gd))
    ha, _, _ = wilson(sum(held), len(held))
    lm, llo, lhi = boot_mean(lat)
    parse_fail = sum(bool(r.get("parse_failure")) for r in g) / len(g)
    xport_fail = sum(bool(r.get("transport_failure")) for r in g) / len(g)
    per_decision = statistics.fmean(trial_cost) if trial_cost else None
    out.append(dict(judge=name.replace("openrouter:", "").replace(";hedge=", " h"), n=len(g),
                    acc=a, acc_lo=alo, acc_hi=ahi, guard=ga, guard_lo=glo, guard_hi=ghi, held_8ball=ha,
                    parse_fail=parse_fail, transport_fail=xport_fail,
                    lat_mean=lm, lat_lo=llo, lat_hi=lhi, lat_p50=pct(lat, .5), lat_p90=pct(lat, .9),
                    lat_max=max(lat) if lat else None, lat_sd=statistics.pstdev(lat) if len(lat) > 1 else None,
                    tok_s=statistics.median(tps) if tps else None, out_tok=statistics.fmean(outtok) if outtok else None,
                    in_tok=statistics.fmean(intok) if intok else None,
                    cost_call=statistics.fmean(costs) if costs else None, cost_decision=per_decision,
                    cost_match=60 * per_decision if per_decision is not None else None,
                    spent=sum(trial_cost)))
out.sort(key=lambda r: (-(r["acc"] or 0), r["lat_mean"] or 1e9))


def pc(x):
    return None if x is None else 100 * x


def f(x, d=2, pre=""):
    return "-" if x is None else f"{pre}{x:.{d}f}"


lines = ["| judge | trials | done accuracy (95% CI) | guard accuracy (95% CI) | held 8-ball caught | parse fail | "
         "latency mean (95% CI) s | p50 / p90 / max s | output tok/s | out tok | in tok | $/call | $/match (~60) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in out:
    lines.append(
        f"| {r['judge']} | {r['n']} | {f(pc(r['acc']),0)}% ({f(pc(r['acc_lo']),0)}-"
        f"{f(pc(r['acc_hi']),0)}) | {f(pc(r['guard']),0)}% "
        f"({f(pc(r['guard_lo']),0)}-{f(pc(r['guard_hi']),0)}) | "
        f"{f(pc(r['held_8ball']),0)}% | {f(100*r['parse_fail'],0)}% | "
        f"{f(r['lat_mean'])} ({f(r['lat_lo'])}-{f(r['lat_hi'])}) | {f(r['lat_p50'])} / {f(r['lat_p90'])} / "
        f"{f(r['lat_max'])} | {f(r['tok_s'],0)} | {f(r['out_tok'],0)} | {f(r['in_tok'],0)} | "
        f"{f(r['cost_call'],5,'$')} | {f(r['cost_match'],3,'$')} |")
total = sum(r["spent"] for r in out)
lines.append(f"\nTotal spent in these runs: ${total:.3f} over {sum(r['n'] for r in out)} judge decisions.")
text = "\n".join(lines)
print(text)
if md_out:
    Path(md_out).write_text(text + "\n", encoding="utf-8")
if csv_out:
    import csv
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
