"""`python -m harness.tasks PLAN DATASET... [--var NAME=VALUE] [--min N]`: does every prompt have demos?

A fine-tune follows the task texts it was trained on. This lists every prompt the plan can send (each stage's
prompt and each recovery prompt) with the episodes and minutes of LeRobot data recorded under exactly that text,
read from `meta/episodes.jsonl` (lengths) and `meta/info.json` (fps). A DATASET may be one LeRobot folder or a
parent of several (the layout the fine-tune upload takes); every `meta/episodes.jsonl` below it is read.

A prompt with no exact match gets the nearest dataset text, and a note when it differs only in case or spaces.
Exit status: 0 when every prompt has at least --min episodes (default 1), 1 when one has fewer, 2 on a bad
plan or no dataset found. Read only: it never touches the station.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .plan import Plan, load_plan

LOW_EPISODES = 30   # fewer than this is marked "low" (TELEOP_PLAN.md aims for 40-70 per task)


@dataclass
class TaskData:
    episodes: int = 0
    frames_s: float = 0.0
    datasets: set[str] = field(default_factory=set)


def plan_prompts(plan: Plan) -> dict[str, list[str]]:
    """Every prompt the plan can send -> where it is used ("stage" or "stage/recovery"), in plan order."""
    out: dict[str, list[str]] = {}
    for stage in plan.stages:
        out.setdefault(stage.prompt, []).append(stage.name)
        for rec in stage.menu():
            if rec.prompt:   # kind retry re-sends the stage prompt
                out.setdefault(rec.prompt, []).append(f"{stage.name}/{rec.name}")
    return out


def dataset_dirs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        p = p.expanduser()
        if (p / "meta" / "episodes.jsonl").is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(m.parent.parent for m in p.rglob("meta/episodes.jsonl")))
    return list(dict.fromkeys(found))


def read_tasks(dirs: list[Path]) -> dict[str, TaskData]:
    """Task text -> episodes and seconds recorded under it, over every dataset folder."""
    tasks: dict[str, TaskData] = defaultdict(TaskData)
    for d in dirs:
        try:
            fps = float(json.loads((d / "meta" / "info.json").read_text(encoding="utf-8")).get("fps") or 30)
        except (OSError, ValueError):
            fps = 30.0
        for line in (d / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                ep = json.loads(line)
            except ValueError:
                continue
            for text in ep.get("tasks") or []:
                t = tasks[text]
                t.episodes += 1
                t.frames_s += float(ep.get("length") or 0) / fps
                t.datasets.add(d.name)
        try:   # a task listed with no episodes still shows up as 0
            for line in (d / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
                text = json.loads(line).get("task") if line.strip() else None
                if text:
                    tasks.setdefault(text, TaskData())
        except (OSError, ValueError):
            pass
    return dict(tasks)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def report(plan: Plan, tasks: dict[str, TaskData], min_episodes: int = 1) -> tuple[str, bool]:
    """The table and whether every prompt has at least min_episodes."""
    lines = [f"checkpoint in plan: {plan.checkpoint or '(none: base Dream-1)'}", ""]
    ok = True
    by_norm = {_norm(t): t for t in tasks}
    for prompt, used in plan_prompts(plan).items():
        data = tasks.get(prompt)
        n = data.episodes if data else 0
        if n < min_episodes:
            ok = False
            mark = "MISSING"
        elif n < LOW_EPISODES:
            mark = "low"
        else:
            mark = "ok"
        minutes = f"{data.frames_s / 60:.1f} min" if data else "-"
        lines.append(f"[{mark:>7}] {n:4d} eps {minutes:>9}  {prompt!r}")
        lines.append(f"{'':10}used by: {', '.join(used)}")
        if not data or n == 0:
            near = by_norm.get(_norm(prompt))
            if near and near != prompt:
                lines.append(f"{'':10}differs only in case/spaces from: {near!r}")
            else:
                close = difflib.get_close_matches(prompt, list(tasks), n=1, cutoff=0.6)
                if close:
                    lines.append(f"{'':10}nearest recorded text: {close[0]!r}")
    unused = [t for t in tasks if t not in plan_prompts(plan)]
    if unused:
        lines += ["", "recorded but never prompted by this plan:"]
        lines += [f"  {tasks[t].episodes:4d} eps  {t!r}" for t in unused]
    return "\n".join(lines), ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m harness.tasks", description=__doc__.split("\n\n")[0])
    p.add_argument("plan")
    p.add_argument("datasets", nargs="+", type=Path, help="LeRobot dataset folders, or parents of several")
    p.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--min", type=int, default=1, dest="min_episodes", help="episodes a prompt needs (default 1)")
    args = p.parse_args(argv)
    variables = dict(v.split("=", 1) for v in args.var if "=" in v)
    try:
        plan = load_plan(args.plan, variables)
    except (ValueError, OSError) as exc:
        print(f"plan error: {exc}", file=sys.stderr)
        return 2
    dirs = dataset_dirs(args.datasets)
    if not dirs:
        print("no LeRobot dataset (meta/episodes.jsonl) found under: " + ", ".join(map(str, args.datasets)),
              file=sys.stderr)
        return 2
    print("datasets: " + ", ".join(str(d) for d in dirs))
    text, ok = report(plan, read_tasks(dirs), args.min_episodes)
    print(text)
    print("" if ok else "\nSome prompts have no demos: record them, or change the plan's prompt to a recorded text.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
