"""Plan files: YAML in, validated dataclasses out."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any

import yaml

VERIFIERS = ("manual", "claude", "time", "claude+manual", "monitor", "monitor+manual")
RECOVERY_KINDS = ("retry", "reset")
ON_FAIL = ("retry_in_place", "home_and_retry", "skip", "abort")
MAX_PROMPT_CHARS = 512

_PLAN_KEYS = {"task", "checkpoint", "time_budget_s", "check_every_s", "verifier", "claude", "cameras_for_check", "stages", "confirm", "done_message", "vars", "guard", "tower_target", "tower_prompts"}
TOWER_STAGES = ("stack", "tower", "higher")   # stage-name prefixes the tower target applies to
_STAGE_KEYS = {"name", "prompt", "success", "timeout_s", "max_attempts", "min_run_s", "verifier", "recovery", "on_fail",
               "stall_s", "settle_s", "skip_if_done", "skip_unless", "confirm", "expect",
               "cameras", "camera_notes", "history_cameras", "watch"}
_RECOVERY_KEYS = {"name", "prompt", "when", "kind", "settle_s", "max_uses"}


@dataclass
class Recovery:
    """One entry of a stage's recovery menu. A v1 plain-string item has legacy=True (hold, then re-issue)."""

    name: str
    prompt: str = ""
    when: str = ""
    kind: str = "reset"  # retry: re-issue the stage prompt; reset: send `prompt`, check the scene, then re-issue
    settle_s: float = 8.0
    max_uses: int = 1
    legacy: bool = False


@dataclass
class Stage:
    name: str
    prompt: str
    success: str
    timeout_s: float = 60.0
    max_attempts: int = 3
    min_run_s: float = 4.0
    verifier: str = "manual"
    recovery: list[str | Recovery] = field(default_factory=list)
    on_fail: str = "retry_in_place"
    stall_s: float | None = None
    settle_s: float | None = None  # None = min_run_s
    skip_if_done: bool = False
    # A yes/no question asked once before the prompt is sent; the stage is skipped unless the answer is a confident
    # yes. E.g. plug-in stages: "are both grippers empty?", so a still-held UNPLUGGED cable is never pushed back in.
    skip_unless: str | None = None
    confirm: int | None = None    # None = the plan's confirm; 1 for stages where a false "done" costs nothing
    expect: str | None = None
    cameras: list[str] = field(default_factory=list)          # roles this stage's checks see; empty = plan's
    camera_notes: dict[str, str] = field(default_factory=dict)  # role substring -> what that view shows
    history_cameras: list[str] = field(default_factory=list)  # extra roles whose frame history the monitor keeps
    watch: Watch | None = None

    def note_for(self, role: str) -> str:
        return next((n for k, n in self.camera_notes.items() if k in role), "")

    def menu(self) -> list[Recovery]:
        """The recovery list with v1 string items wrapped as legacy Recovery entries."""
        return [r if isinstance(r, Recovery) else Recovery(name=f"recovery{i + 1}", prompt=r, legacy=True)
                for i, r in enumerate(self.recovery)]

    def effective_settle(self) -> float:
        return self.min_run_s if self.settle_s is None else self.settle_s


_GUARD_KEYS = {"question", "stages", "confirm", "message", "max_blind_s"}


@dataclass
class Guard:
    """A yes/no safety question the monitor answers on every check of the stages it covers. `confirm` yes answers
    in a row stop the run at once (POST /stop: the arms hold where they are, so a held object stays held)."""

    question: str
    stages: list[str] = field(default_factory=list)   # stage-name prefixes it covers; empty = every stage
    confirm: int = 1
    message: str = "GUARD TRIPPED: RUN STOPPED. CALL STOP NOW"
    # Stop as if tripped when a covered stage has had no yes/no answer for this long (monitor down, bad frames):
    # a robot binning balls that nobody is watching for a black one. None = never.
    max_blind_s: float | None = None

    def covers(self, stage_name: str) -> bool:
        return not self.stages or any(stage_name.startswith(p) for p in self.stages)


@dataclass
class Watch:
    """A watcher arm whose wrist camera must keep looking at the work while another arm does the stage.

    Each monitor check also answers `check`; after `confirm` "no" answers in a row the runner sends `prompt`
    (put the watcher back), holds `settle_s`, then re-sends the stage prompt. At most `max_fixes` per attempt;
    a fix does not count as a failed attempt."""

    camera: str
    check: str
    prompt: str
    settle_s: float = 5.0
    max_fixes: int = 2


@dataclass
class Plan:
    task: str
    stages: list[Stage]
    checkpoint: str | None = None
    time_budget_s: float = 900.0
    check_every_s: float = 3.0
    verifier: str = "manual"
    claude: dict[str, Any] = field(default_factory=lambda: {"model": "claude-haiku-5-5", "max_tokens": 200})
    cameras_for_check: list[str] = field(default_factory=list)  # empty = all cameras
    confirm: int = 1
    claude_user: dict[str, Any] = field(default_factory=dict)  # only the keys the plan file set
    done_message: str = ""  # printed large (with a bell) when every stage passed, e.g. "CALL STOP NOW"
    guard: Guard | None = None
    # The tower to build, bottom to top, one piece per entry ("blue cube, hole facing sideways"). The monitor gets it
    # as the expected result on stacking stages and reports the tower it sees; the runner logs whether the order
    # matches. It does not gate a stage: the stack prompt is one recorded text, so Dream can't be told which piece.
    tower_target: list[str] = field(default_factory=list)
    # Optional: name the next piece in the prompt instead of the stage's generic one. Shape word -> template with
    # <piece>, e.g. {"cube": "put the <piece> on top of the stack"}. The piece is the target's next entry after the
    # tower the monitor last reported. Off (empty) for a fine-tune that only knows its recorded texts.
    tower_prompts: dict[str, str] = field(default_factory=dict)


def _roles(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(c, str) and c.strip() for c in value):
        raise ValueError(f"{where}: must be a list of camera role names")
    return [c.strip() for c in value]


def _num(value: Any, where: str, *, integer: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (integer and not isinstance(value, int)):
        raise ValueError(f"{where}: must be {'an integer' if integer else 'a number'}, got {value!r}")
    if value <= 0:
        raise ValueError(f"{where}: must be positive, got {value!r}")
    return value


def _prompt(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: must be a non-empty string")
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise ValueError(f"{where}: must be a single line")
    if len(value) > MAX_PROMPT_CHARS:
        raise ValueError(f"{where}: {len(value)} chars, the limit is {MAX_PROMPT_CHARS}")
    return value


def _enum(value: Any, allowed: tuple[str, ...], where: str) -> str:
    if value not in allowed:
        raise ValueError(f"{where}: must be one of {', '.join(allowed)}, got {value!r}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{where}: must be true or false, got {value!r}")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: must be a non-empty string")
    return value.strip()


def _recovery_item(raw: Any, j: int, label: str, seen: set[str]) -> str | Recovery:
    """A v1 string item stays a plain str in Stage.recovery (Stage.menu() wraps it)."""
    where = f"{label}.recovery[{j}]"
    if isinstance(raw, str):
        item = Recovery(name=f"recovery{j + 1}", prompt=_prompt(raw, where), legacy=True)
        if item.name in seen:
            raise ValueError(f"{where}: name clash with another recovery item")
        seen.add(item.name)
        return item.prompt
    elif isinstance(raw, dict):
        unknown = set(raw) - _RECOVERY_KEYS
        if unknown:
            raise ValueError(f"{where}: unknown field(s) {sorted(unknown)}")
        has_prompt = raw.get("prompt") is not None
        kind = _enum(raw.get("kind", "reset" if has_prompt else "retry"), RECOVERY_KINDS, f"{where}.kind")
        if kind == "reset" and not has_prompt:
            raise ValueError(f"{where}.prompt: required for kind reset")
        if kind == "retry" and has_prompt:
            raise ValueError(f"{where}.prompt: kind retry re-issues the stage prompt; remove prompt or use kind reset")
        name = raw.get("name", f"recovery{j + 1}")
        item = Recovery(
            name=_text(name, f"{where}.name"),
            prompt=_prompt(raw["prompt"], f"{where}.prompt") if has_prompt else "",
            when=_text(raw["when"], f"{where}.when") if raw.get("when") is not None else "",
            kind=kind,
            settle_s=float(_num(raw.get("settle_s", 8.0), f"{where}.settle_s")),
            max_uses=int(_num(raw.get("max_uses", 1), f"{where}.max_uses", integer=True)),
        )
    else:
        raise ValueError(f"{where}: must be a prompt string or a mapping")
    if item.name in seen:
        raise ValueError(f"{where}.name: duplicate recovery name {item.name!r}")
    seen.add(item.name)
    return item


def parse_plan(data: Any) -> Plan:
    """Validates an already-loaded mapping."""
    if not isinstance(data, dict):
        raise ValueError("plan: top level must be a mapping")
    unknown = set(data) - _PLAN_KEYS
    if unknown:
        raise ValueError(f"plan: unknown field(s) {sorted(unknown)}")
    default_verifier = _enum(data.get("verifier", "manual"), VERIFIERS, "plan.verifier")
    raw_stages = data.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("plan.stages: must be a non-empty list")

    stages: list[Stage] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_stages):
        label = f"stage #{i + 1}"
        if not isinstance(raw, dict):
            raise ValueError(f"{label}: must be a mapping")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.name: must be a non-empty string")
        label = f"stage {name!r}"
        if name in seen:
            raise ValueError(f"{label}.name: duplicate stage name")
        seen.add(name)
        unknown = set(raw) - _STAGE_KEYS
        if unknown:
            raise ValueError(f"{label}: unknown field(s) {sorted(unknown)}")
        success = raw.get("success", f"Has the stage '{name}' been completed?")
        if not isinstance(success, str) or not success.strip():
            raise ValueError(f"{label}.success: must be a non-empty string")
        recovery = raw.get("recovery") or []
        if not isinstance(recovery, list):
            raise ValueError(f"{label}.recovery: must be a list of prompts or mappings")
        rec_seen: set[str] = set()
        recs = [_recovery_item(r, j, label, rec_seen) for j, r in enumerate(recovery)]
        stall = raw.get("stall_s")
        settle = raw.get("settle_s")
        expect = raw.get("expect")
        if expect is not None:
            expect = _text(expect, f"{label}.expect")
        cams = _roles(raw.get("cameras"), f"{label}.cameras")
        hist = _roles(raw.get("history_cameras"), f"{label}.history_cameras")
        notes = raw.get("camera_notes") or {}
        if not isinstance(notes, dict) or not all(isinstance(k, str) and isinstance(v, str) and v.strip()
                                                  for k, v in notes.items()):
            raise ValueError(f"{label}.camera_notes: must map camera role names to non-empty text")
        watch = None
        if raw.get("watch") is not None:
            w = raw["watch"]
            if not isinstance(w, dict) or set(w) - {"camera", "check", "prompt", "settle_s", "max_fixes"}:
                raise ValueError(f"{label}.watch: must be a mapping of camera, check, prompt[, settle_s, max_fixes]")
            watch = Watch(camera=_text(w.get("camera"), f"{label}.watch.camera"),
                          check=_text(w.get("check"), f"{label}.watch.check"),
                          prompt=_prompt(w.get("prompt"), f"{label}.watch.prompt"),
                          settle_s=float(_num(w.get("settle_s", 5.0), f"{label}.watch.settle_s")),
                          max_fixes=int(_num(w.get("max_fixes", 2), f"{label}.watch.max_fixes", integer=True)))
            if not any(watch.camera in h for h in hist):
                hist.append(watch.camera)   # the monitor needs the watcher's frame history to judge it
        stages.append(
            Stage(
                name=name,
                prompt=_prompt(raw.get("prompt"), f"{label}.prompt"),
                success=success.strip(),
                timeout_s=float(_num(raw.get("timeout_s", 60.0), f"{label}.timeout_s")),
                max_attempts=int(_num(raw.get("max_attempts", 3), f"{label}.max_attempts", integer=True)),
                min_run_s=float(_num(raw.get("min_run_s", 4.0), f"{label}.min_run_s")),
                verifier=_enum(raw.get("verifier", default_verifier), VERIFIERS, f"{label}.verifier"),
                recovery=recs,
                on_fail=_enum(raw.get("on_fail", "retry_in_place"), ON_FAIL, f"{label}.on_fail"),
                stall_s=None if stall is None else float(_num(stall, f"{label}.stall_s")),
                settle_s=None if settle is None else float(_num(settle, f"{label}.settle_s")),
                skip_if_done=_bool(raw.get("skip_if_done", False), f"{label}.skip_if_done"),
                skip_unless=None if raw.get("skip_unless") is None else _text(raw["skip_unless"], f"{label}.skip_unless"),
                confirm=None if raw.get("confirm") is None else int(_num(raw["confirm"], f"{label}.confirm", integer=True)),
                expect=expect,
                cameras=cams,
                camera_notes={k: v.strip() for k, v in notes.items()},
                history_cameras=hist,
                watch=watch,
            )
        )

    checkpoint = data.get("checkpoint")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ValueError("plan.checkpoint: must be a string or null")
    claude = {"model": "claude-haiku-5-5", "max_tokens": 200}
    claude_user: dict[str, Any] = {}
    if data.get("claude") is not None:
        if not isinstance(data["claude"], dict):
            raise ValueError("plan.claude: must be a mapping")
        claude.update(data["claude"])
        claude_user = dict(data["claude"])
    cams = data.get("cameras_for_check") or []
    if not isinstance(cams, list) or not all(isinstance(c, str) for c in cams):
        raise ValueError("plan.cameras_for_check: must be a list of role names")
    task = data.get("task", "")
    if not isinstance(task, str):
        raise ValueError("plan.task: must be a string")
    done_message = data.get("done_message", "") or ""
    if not isinstance(done_message, str):
        raise ValueError("plan.done_message: must be a string")
    guard = _guard(data.get("guard"))
    if guard is not None and default_verifier not in ("monitor", "monitor+manual"):
        raise ValueError("plan.guard: needs verifier monitor or monitor+manual (only the monitor answers it)")
    if guard is not None:
        names = [st.name for st in stages]
        for prefix in guard.stages:
            if not any(n.startswith(prefix) for n in names):
                raise ValueError(f"plan.guard.stages: {prefix!r} matches no stage name")
    tower = data.get("tower_target") or []
    if not isinstance(tower, list) or not all(isinstance(x, str) and x.strip() for x in tower):
        raise ValueError("plan.tower_target: must be a list of pieces, bottom to top")
    return Plan(
        task=task,
        guard=guard,
        tower_target=[x.strip() for x in tower],
        tower_prompts=_tower_prompts(data.get("tower_prompts"), [x.strip() for x in tower]),
        done_message=done_message,
        stages=stages,
        checkpoint=checkpoint,
        time_budget_s=float(_num(data.get("time_budget_s", 900.0), "plan.time_budget_s")),
        check_every_s=float(_num(data.get("check_every_s", 3.0), "plan.check_every_s")),
        verifier=default_verifier,
        claude=claude,
        cameras_for_check=list(cams),
        confirm=int(_num(data.get("confirm", 1), "plan.confirm", integer=True)),
        claude_user=claude_user,
    )


def _tower_prompts(raw: Any, target: list[str]) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) and "<piece>" in v
                                            for k, v in raw.items()):
        raise ValueError("plan.tower_prompts: must map a shape word to a prompt containing <piece>")
    if not target:
        raise ValueError("plan.tower_prompts: needs tower_target")
    for piece in target:
        name = piece.split(",")[0].strip()
        if not any(name.endswith(k) for k in raw):
            raise ValueError(f"plan.tower_prompts: no template for {name!r}")
        prompt = next(v for k, v in raw.items() if name.endswith(k)).replace("<piece>", name)
        _prompt(prompt, "plan.tower_prompts")
    return {k: v.strip() for k, v in raw.items()}


def _guard(raw: Any) -> Guard | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("plan.guard: must be a mapping")
    extra = set(raw) - _GUARD_KEYS
    if extra:
        raise ValueError(f"plan.guard: unknown key(s): {', '.join(sorted(extra))}")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("plan.guard.question: must be a non-empty yes/no question")
    stages = raw.get("stages") or []
    if not isinstance(stages, list) or not all(isinstance(x, str) and x.strip() for x in stages):
        raise ValueError("plan.guard.stages: must be a list of stage-name prefixes")
    message = raw.get("message", Guard.message)
    if not isinstance(message, str) or not message.strip():
        raise ValueError("plan.guard.message: must be non-empty text")
    blind = raw.get("max_blind_s")
    return Guard(question=question.strip(), stages=[x.strip() for x in stages],
                 confirm=int(_num(raw.get("confirm", 1), "plan.guard.confirm", integer=True)), message=message.strip(),
                 max_blind_s=None if blind is None else float(_num(blind, "plan.guard.max_blind_s")))


_VAR = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute(text: str, variables: dict[str, str] | None) -> str:
    """Replaces {name} placeholders (no spaces inside the braces) with --var values.

    Raises ValueError for a placeholder with no value, so a plan never reaches the robot with "{color}" in it.
    YAML flow mappings such as {name: x} are untouched (they contain a colon and a space)."""
    variables = variables or {}
    missing = sorted({m for m in _VAR.findall(text) if m not in variables})
    if missing:
        raise ValueError(f"plan needs --var for: {', '.join(missing)}")
    return _VAR.sub(lambda m: variables[m.group(1)], text)


def load_plan(path: str | Path, variables: dict[str, str] | None = None) -> Plan:
    """Reads and validates a plan YAML file. Raises ValueError (or OSError if unreadable).

    {name} placeholders are filled from the plan's own `vars:` defaults, then from `variables` (--var wins)."""
    text = Path(path).read_text(encoding="utf-8")
    merged: dict[str, str] = {}
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        raw = None  # placeholders may sit where YAML can't parse them; the substituted text is checked below
    if isinstance(raw, dict) and raw.get("vars") is not None:
        if not isinstance(raw["vars"], dict) or not all(isinstance(k, str) for k in raw["vars"]):
            raise ValueError("plan.vars: must map names to values")
        merged.update({k: str(v) for k, v in raw["vars"].items()})
    merged.update(variables or {})
    text = substitute(text, merged)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from None
    return parse_plan(data)
