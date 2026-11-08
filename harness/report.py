"""`python -m harness.report runs/ [--last N]`: a table over past runs (DESIGN_v2.md section 7)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMMEDIATE_S = 15.0   # a stage that fails this soon after a successful one is a "false success" lead


def _jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    except OSError:
        pass
    return out


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def run_dirs(root: str | Path, last: int | None = None) -> list[Path]:
    dirs = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "summary.json").exists())
    return dirs[-last:] if last else dirs


def _normalise(reason: str) -> str:
    reason = reason.split(":")[0] if reason.startswith(("verifier said no", "monitor says", "stalled", "regressed")) else reason
    return re.sub(r"\d+(\.\d+)?", "N", reason).strip()[:80] or "(no reason)"


def _false_successes(stages: list[dict[str, Any]], segments: list[dict[str, Any]], events: list[dict[str, Any]]) -> int:
    count = 0
    for prev, cur in zip(stages, stages[1:]):
        if not prev.get("ok") or cur.get("ok"):
            continue
        seg = next((s for s in segments if s.get("stage") == cur["name"] and s.get("kind") == "stage"
                    and s.get("attempt") == 1), None)
        if seg is None or seg["t_end"] - seg["t_start"] > IMMEDIATE_S or seg.get("outcome") != "failed":
            continue
        if any(e.get("event") == "verdict" and e.get("stage") == cur["name"] and e.get("scene_ok") is False
               and seg["t_start"] <= e["t"] <= seg["t_end"] + 1 for e in events):
            count += 1
    return count


def build_report(dirs: list[Path]) -> dict[str, Any]:
    stage_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"runs": 0, "ok": 0, "attempts": 0, "seconds": 0.0})
    rec_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"used": 0, "ok": 0})
    reasons: Counter[str] = Counter()
    runs_ok = false_success = 0
    for d in dirs:
        summary, segments, events = _json(d / "summary.json"), _jsonl(d / "segments.jsonl"), _jsonl(d / "events.jsonl")
        runs_ok += bool(summary.get("ok"))
        stages = [s for s in summary.get("stages", []) if isinstance(s, dict) and "name" in s]
        for s in stages:
            st = stage_stats[s["name"]]
            st["runs"] += 1
            st["ok"] += bool(s.get("ok"))
            st["attempts"] += s.get("attempts", 0)
            st["seconds"] += s.get("seconds", 0.0)
        for seg in segments:
            if seg.get("kind") == "stage" and seg.get("outcome") == "failed":
                reasons[_normalise(str(seg.get("reason", "")))] += 1
        for i, seg in enumerate(segments):
            if seg.get("kind") != "recovery":
                continue
            stat = rec_stats[f"{seg['stage']}/{seg.get('name', '?')}"]
            stat["used"] += 1
            nxt = next((s for s in segments[i + 1:] if s.get("stage") == seg["stage"] and s.get("kind") == "stage"), None)
            stat["ok"] += bool(nxt and nxt.get("outcome") == "ok")   # worked if the attempt after it succeeded
        false_success += _false_successes(stages, segments, events)
    return {"runs": len(dirs), "runs_ok": runs_ok, "stages": dict(stage_stats), "recoveries": dict(rec_stats),
            "failure_reasons": reasons.most_common(5), "false_success_leads": false_success}


def format_report(rep: dict[str, Any]) -> str:
    lines = [f"runs: {rep['runs']}   ok: {rep['runs_ok']}/{rep['runs']}", "",
             f"{'stage':<24}{'ok/total':>10}{'rate':>7}{'attempts':>10}{'secs':>8}"]
    for name, st in rep["stages"].items():
        n = int(st["runs"])
        lines.append(f"{name:<24}{int(st['ok']):>5}/{n:<4}{st['ok'] / n:>7.0%}{st['attempts'] / n:>10.2f}{st['seconds'] / n:>8.1f}")
    lines += ["", "recoveries (worked = the next attempt of that stage succeeded):"]
    lines += [f"  {k:<30} used {v['used']}  worked {v['ok']} ({v['ok'] / v['used']:.0%})"
              for k, v in rep["recoveries"].items()] or ["  none"]
    lines += ["", "most common failure reasons:"]
    lines += [f"  {n:>3}  {r}" for r, n in rep["failure_reasons"]] or ["  none"]
    lines += ["", f"false-success leads (stage fails at once with scene_ok false after a success): "
                  f"{rep['false_success_leads']}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m harness.report", description="Summarise Hill Climb runs")
    p.add_argument("runs", help="the runs/ directory")
    p.add_argument("--last", type=int, default=None, help="only the newest N runs")
    args = p.parse_args(argv)
    try:
        dirs = run_dirs(args.runs, args.last)
    except OSError as exc:
        print(f"cannot read {args.runs}: {exc}", file=sys.stderr)
        return 2
    if not dirs:
        print("no finished runs found", file=sys.stderr)
        return 1
    print(format_report(build_report(dirs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
