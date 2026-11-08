"""Estimate the force a Metal arm was commanding, from Dream's per-tick run trace (no robot access needed).

    python -m harness.force last-run.csv [--arm r] [--tcp 0 0 0.12] [--from 10 --to 25] [--csv out.csv]

Dream's station writes `~/.dreamlabs/last-run.csv` after every run (robots/control.py trace_columns): per 20 Hz tick
the measured and commanded joint angles of both arms (radians, the URDF frame), the measured and commanded gripper
(0 closed .. 1 open), the end-effector speed each arm asked for and a `clamped` flag. Copy it off the station
after an unplug/plug attempt and run this on any machine.

The Metal joints run Damiao MIT control: torque = kp * (target - angle) + kd * (0 - velocity) + gravity feedforward
(robots/metal.py, dreamlabs 1.0.37). The gravity feedforward only holds the arm up, so the PD part is what pushes or
pulls on the world. This tool rebuilds that PD torque per joint, maps it to a force at the hand with the arm's
Jacobian (built from the same link table Dream's gravity model uses), and estimates the gripper's squeeze.

It is an estimate, and it says so:
- the trace has no measured motor torque or current (the driver reads them but the trace does not keep them);
- velocity is a finite difference of the measured angles at 20 Hz;
- Dream's gravity factors (1.1-1.2) over-compensate a little, so a constant PD offset is subtracted as a baseline
  (the median over the first `--baseline-s` seconds, when the arm is moving freely);
- the tool point defaults to the wrist-roll joint; pass `--tcp` (metres, in the last link's frame) for the jaw tip.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

# robots/metal.py JOINTS: (name, kp N*m/rad, kd N*m*s/rad, motor model)
JOINTS = (("shoulder_pan", 100.0, 5.0, "metal_jlo"), ("shoulder_lift", 100.0, 5.0, "metal_j2"),
          ("elbow_flex", 100.0, 5.0, "metal_jlo"), ("wrist_flex", 50.0, 5.0, "metal_jlo"),
          ("wrist_yaw", 20.0, 0.6, "metal_jhi"), ("wrist_roll", 20.0, 0.6, "metal_jhi"))
# robots/damiao.py MOTOR_LIMITS: the TMAX each motor's MIT frame is scaled to, N*m (an upper bound, not a rating).
TMAX = {"metal_jlo": 30.0, "metal_j2": 120.0, "metal_jhi": 20.0}
# robots/metal.py LINKS: per joint, origin xyz in the previous link (m), rpy (rad), axis.
LINKS = (((0.0, 0.0, 0.06675), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
         ((0.00075, 0.0, 0.055), (1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
         ((-0.28, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
         ((0.215, 0.063, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
         ((0.080817, 0.05072, 0.00025), (1.5708, 0.0, -0.015918), (0.0, 0.0, 1.0)),
         ((0.0235, 0.0, 0.044), (-3.14159265330655, 0.0, 0.0), (1.0, 0.0, 0.0)))
# Gripper: motor kp 20 N*m/rad (metal.py), stroke 0..137.5 motor degrees (metal_gripper.LIMITS_DEG), held at
# the arm's `gripper_hold_torque_nm` (0.1..2.0 N*m at the motor, makers_hold.py; 0.5 is the default).
GRIP_KP = 20.0
SUSTAIN_S = 0.5
GRIP_STROKE_RAD = math.radians(137.5)


# -- small 3-vector / 3x3 algebra (stdlib only, like the rest of the harness) --------------------------------------
def _mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _rot_axis(axis, angle):
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def _rpy(r, p, y):
    return _mm(_rot_axis((0, 0, 1), y), _mm(_rot_axis((0, 1, 0), p), _rot_axis((1, 0, 0), r)))


def _inv3(m):
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ZeroDivisionError("singular")
    return [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]


def tool_point(q: list[float], tcp=(0.0, 0.0, 0.0)) -> list[float]:
    """Forward kinematics of the Metal arm to the tool point, in the arm's base frame, metres."""
    rot = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    org = [0.0, 0.0, 0.0]
    for (xyz, rpy, axis), angle in zip(LINKS, q):
        org = [o + d for o, d in zip(org, _mv(rot, xyz))]
        rot = _mm(_mm(rot, _rpy(*rpy)), _rot_axis(axis, angle))
    return [o + d for o, d in zip(org, _mv(rot, tcp))]


def jacobian(q: list[float], tcp=(0.0, 0.0, 0.0), eps: float = 1e-6) -> list[list[float]]:
    """3x6 positional Jacobian, by central differences."""
    cols = []
    for j in range(6):
        hi, lo = list(q), list(q)
        hi[j] += eps
        lo[j] -= eps
        a, b = tool_point(hi, tcp), tool_point(lo, tcp)
        cols.append([(x - y) / (2 * eps) for x, y in zip(a, b)])
    return [[cols[j][i] for j in range(6)] for i in range(3)]


def hand_force(q: list[float], tau: list[float], tcp=(0.0, 0.0, 0.0), damping: float = 1e-4) -> list[float]:
    """The hand force F (N, base frame) whose J^T F best matches the joint torques tau (damped least squares)."""
    jac = jacobian(q, tcp)
    jjt = [[sum(jac[i][k] * jac[j][k] for k in range(6)) + (damping if i == j else 0.0) for j in range(3)]
           for i in range(3)]
    jt = [sum(jac[i][k] * tau[k] for k in range(6)) for i in range(3)]
    return _mv(_inv3(jjt), jt)


# -- the trace ------------------------------------------------------------------------------------------------------
def load_trace(path: str | Path) -> list[dict[str, float]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for k, v in raw.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = float("nan")
            rows.append(row)
    if not rows or "t_s" not in rows[0]:
        raise ValueError(f"{path}: not a Dream run trace (no t_s column)")
    return rows


def analyse(rows, arm="r", tcp=(0.0, 0.0, 0.0), baseline_s=1.0, hold_torque_nm=0.5):
    """Per tick: PD torque per joint, % of TMAX, hand force, gripper squeeze estimate."""
    meas = [f"measured_{arm}{i}" for i in range(6)]
    cmd = [f"commanded_{arm}{i}" for i in range(6)]
    missing = [c for c in meas + cmd if c not in rows[0]]
    if missing:
        raise ValueError(f"trace has no columns {missing[:3]}... for arm {arm!r}")
    out = []
    for n, row in enumerate(rows):
        q = [row[c] for c in meas]
        if any(math.isnan(x) for x in q) or any(math.isnan(row[c]) for c in cmd):
            continue
        prev = rows[n - 1] if n else row
        dt = row["t_s"] - prev["t_s"]
        vel = [(row[c] - prev[c]) / dt if dt > 0 else 0.0 for c in meas]
        tau = [kp * (row[c] - row[m]) - kd * v for (_, kp, kd, _), c, m, v in zip(JOINTS, cmd, meas, vel)]
        out.append({"t_s": row["t_s"], "q": q, "tau": tau, "clamped": row.get("clamped", 0.0),
                    "grip": row.get(f"gripper_{arm}"), "cmd_grip": row.get(f"cmd_gripper_{arm}"),
                    "ee_speed": row.get(f"ee_speed_{arm}_m_s")})
    if not out:
        raise ValueError("no usable ticks in the trace")
    t0 = out[0]["t_s"]
    base_rows = [r for r in out if r["t_s"] - t0 <= baseline_s] or out[:1]
    base = [statistics.median(r["tau"][j] for r in base_rows) for j in range(6)]
    for r in out:
        r["tau_ext"] = [t - b for t, b in zip(r["tau"], base)]
        r["pct"] = [100.0 * abs(t) / TMAX[jt[3]] for t, jt in zip(r["tau"], JOINTS)]
        try:
            f = hand_force(r["q"], r["tau_ext"], tcp)
        except ZeroDivisionError:
            f = [float("nan")] * 3
        r["force"] = f
        r["force_n"] = math.sqrt(sum(x * x for x in f))
        g, cg = r["grip"], r["cmd_grip"]
        if g is not None and cg is not None and not math.isnan(g) and not math.isnan(cg):
            raw = GRIP_KP * GRIP_STROKE_RAD * max(0.0, g - cg)   # commanded more closed than the jaw is
            r["grip_nm"] = min(raw, hold_torque_nm)
            r["grip_capped"] = raw >= hold_torque_nm
        else:
            r["grip_nm"], r["grip_capped"] = None, False
    return out, base


def _pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))] if s else float("nan")


def report(out, base, *, arm, hold_torque_nm, t_from=None, t_to=None) -> str:
    t0 = out[0]["t_s"]
    sel = [r for r in out if (t_from is None or r["t_s"] - t0 >= t_from) and (t_to is None or r["t_s"] - t0 <= t_to)]
    if not sel:
        return "no ticks in that time window"
    lines = [f"arm {arm}: {len(sel)} ticks, t = {sel[0]['t_s'] - t0:.1f} .. {sel[-1]['t_s'] - t0:.1f} s",
             f"baseline PD torque subtracted (N*m): " + ", ".join(f"{b:+.2f}" for b in base), "",
             f"{'joint':<14}{'kp':>6}{'peak |tau| N*m':>16}{'p90':>8}{'% of TMAX':>11}"]
    for j, (name, kp, _, model) in enumerate(JOINTS):
        mags = [abs(r["tau"][j]) for r in sel]
        lines.append(f"{name:<14}{kp:>6.0f}{max(mags):>16.2f}{_pct(mags, 90):>8.2f}"
                     f"{100 * max(mags) / TMAX[model]:>10.0f}%")
    forces = [r["force_n"] if not math.isnan(r["force_n"]) else 0.0 for r in sel]
    # A one-tick spike (a fast move opening a big tracking gap for a moment) is not a pull: the headline peak is
    # the highest force held for SUSTAIN_S (a rolling median), and the raw one-tick peak is shown beside it.
    k = max(1, int(round(SUSTAIN_S / max(1e-3, statistics.median(
        b["t_s"] - a["t_s"] for a, b in zip(sel, sel[1:])) if len(sel) > 1 else 0.05))))
    rolled = [statistics.median(forces[i:i + k]) for i in range(max(1, len(forces) - k + 1))]
    i_best = max(range(len(rolled)), key=rolled.__getitem__)
    held = sel[i_best + k // 2] if i_best + k // 2 < len(sel) else sel[i_best]
    lines += ["", f"hand force estimate: median {statistics.median(forces):.1f} N, p90 {_pct(forces, 90):.1f} N, "
                  f"highest held for {SUSTAIN_S:g} s: {rolled[i_best]:.1f} N at t = {held['t_s'] - t0:.1f} s "
                  f"(x {held['force'][0]:+.1f}, y {held['force'][1]:+.1f}, z {held['force'][2]:+.1f} N, base frame); "
                  f"one-tick peak {max(forces):.1f} N (fast moves spike this; not a pull)"]
    grips = [r for r in sel if r["grip_nm"] is not None]
    if grips:
        capped = sum(r["grip_capped"] for r in grips)
        lines.append(f"gripper squeeze estimate: peak {max(r['grip_nm'] for r in grips):.2f} N*m at the motor, "
                     f"at the hold limit ({hold_torque_nm:g} N*m) on {100 * capped / len(grips):.0f}% of ticks")
    clamped = sum(1 for r in sel if r["clamped"] and not math.isnan(r["clamped"]))
    lines.append(f"clamped ticks: {clamped} of {len(sel)} ({100 * clamped / len(sel):.0f}%)")
    lines += ["", "Reading it:"]
    grip_capped = grips and sum(r["grip_capped"] for r in grips) / len(grips) > 0.3
    if grip_capped:
        lines.append("- the gripper sits at its hold limit much of the time: the cable most likely slips in the jaws. "
                     "The knob is the arm's gripper_hold_torque_nm in the station config (not firmware).")
    if clamped / len(sel) > 0.2:
        lines.append("- many ticks were clamped: Dream's speed/step limits cut the command, so the target cannot "
                     "lead the hand far enough to pull hard.")
    if forces and _pct(forces, 90) < 5 and not grip_capped:
        lines.append("- the commanded force stays small: the model is not asking to pull (its targets track the "
                     "hand closely). Demos that pull harder, or a longer, straighter pull, are the fix.")
    if len(lines) and lines[-1] == "Reading it:":
        lines.append("- no single limit stands out; compare this window with a window where the pull worked.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m harness.force", description=__doc__.split("\n\n")[0])
    p.add_argument("trace", help="Dream's run trace, e.g. a copy of ~/.dreamlabs/last-run.csv")
    p.add_argument("--arm", choices=("l", "r"), default="r")
    p.add_argument("--tcp", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="tool point in the last link's frame, metres (default: the wrist-roll joint)")
    p.add_argument("--baseline-s", type=float, default=1.0, help="seconds at the start used as the zero-force PD offset")
    p.add_argument("--hold-torque", type=float, default=0.5, help="the arm's gripper_hold_torque_nm (default 0.5)")
    p.add_argument("--from", dest="t_from", type=float, default=None, help="window start, s from the trace start")
    p.add_argument("--to", dest="t_to", type=float, default=None, help="window end, s")
    p.add_argument("--csv", default=None, help="also write per-tick results here")
    a = p.parse_args(argv)
    try:
        out, base = analyse(load_trace(a.trace), a.arm, tuple(a.tcp), a.baseline_s, a.hold_torque)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report(out, base, arm=a.arm, hold_torque_nm=a.hold_torque, t_from=a.t_from, t_to=a.t_to))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", *[f"tau_{n}" for n, *_ in JOINTS], "fx", "fy", "fz", "force_n", "grip_nm", "clamped"])
            for r in out:
                w.writerow([f"{r['t_s']:.3f}", *[f"{t:.3f}" for t in r["tau"]], *[f"{f:.2f}" for f in r["force"]],
                            f"{r['force_n']:.2f}", "" if r["grip_nm"] is None else f"{r['grip_nm']:.3f}",
                            int(r["clamped"] or 0)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
