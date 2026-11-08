"""Measured joint torque on a Metal arm, from a passive CAN capture (read only: it never sends a frame).

On the station (SocketCAN links, e.g. can_left / can_right), while Dream's agent runs the arm:

    candump -L -n 4000 can_right > right.log          # listen only; Ctrl+C or -n stops it
    python -m harness.can_torque right.log [--csv out.csv]

Every Damiao motor answers each MIT command with a state frame from id 0x10 + motor id (robots/damiao.py,
dreamlabs 1.0.37): byte 0 = status << 4 | id, then position (16 bit), velocity (12 bit), torque (12 bit), MOSFET
and rotor temperature. The torque is the motor's own estimate from its phase current, so unlike
`harness.force` (which rebuilds the commanded torque from the trace) this is what the joint actually produced.

No candump on the station? `--listen` does the same listen with the Python standard library (a raw SocketCAN
socket; it only receives, never sends, and needs nothing installed):

    python3 harness/can_torque.py --listen can_follower_r --seconds 15 > right.log

If the arms are on a CANable in slcan mode (a /dev/tty... serial device), the port is exclusive to Dream's agent
and cannot be listened to alongside it; this tool then has nothing to read.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from pathlib import Path

REPLY_OFFSET = 0x10
_STATUSES = {0, 1, *range(8, 15)}  # disabled, enabled, faults 8..14
# robots/metal.py JOINTS (CAN id -> name, motor model) and robots/damiao.py MOTOR_LIMITS (PMAX rad, VMAX rad/s, TMAX)
MOTORS = {1: ("shoulder_pan", "metal_jlo"), 2: ("shoulder_lift", "metal_j2"), 3: ("elbow_flex", "metal_jlo"),
          4: ("wrist_flex", "metal_jlo"), 5: ("wrist_yaw", "metal_jhi"), 6: ("wrist_roll", "metal_jhi"),
          7: ("gripper", "metal_jhi")}
LIMITS = {"metal_jlo": (6.28, 10.0, 30.0), "metal_j2": (6.28, 10.0, 120.0), "metal_jhi": (6.28, 30.0, 20.0)}

# candump -L: "(1699999999.123456) can0 011#0102030405060708"
_LOG = re.compile(r"^\((?P<t>[\d.]+)\)\s+(?P<ch>\S+)\s+(?P<id>[0-9A-Fa-f]+)#(?P<data>[0-9A-Fa-f]*)")
# candump default / -t a: "(…) can0  011   [8]  01 02 03 04 05 06 07 08" or "  can0  011   [8]  01 02 …"
_TXT = re.compile(r"^(?:\((?P<t>[\d.\-: ]+)\)\s+)?(?P<ch>\S+)\s+(?P<id>[0-9A-Fa-f]{1,8})\s+\[(?P<n>\d)\]\s+(?P<data>(?:[0-9A-Fa-f]{2}\s*)*)")


def _u2f(x: int, lo: float, hi: float, bits: int) -> float:
    return x * (hi - lo) / ((1 << bits) - 1) + lo


def decode(data: bytes, model: str) -> dict[str, float]:
    pmax, vmax, tmax = LIMITS[model]
    q = (data[1] << 8) | data[2]
    dq = (data[3] << 4) | (data[4] >> 4)
    tau = ((data[4] & 0x0F) << 8) | data[5]
    return {"status": data[0] >> 4, "pos_deg": math.degrees(_u2f(q, -pmax, pmax, 16)),
            "vel_deg_s": math.degrees(_u2f(dq, -vmax, vmax, 12)), "torque_nm": _u2f(tau, -tmax, tmax, 12),
            "temp_mos_c": float(data[6]), "temp_rotor_c": float(data[7])}


def parse(lines) -> list[dict]:
    """State frames from candump output (either format). Command, query and register frames are skipped."""
    out, n = [], 0
    for line in lines:
        m = _LOG.match(line.strip()) or _TXT.match(line.strip())
        if not m:
            continue
        n += 1
        arb = int(m.group("id"), 16)
        data = bytes.fromhex(re.sub(r"\s+", "", m.group("data")))
        motor = arb - REPLY_OFFSET
        if motor not in MOTORS or len(data) < 8 or (data[0] & 0x0F) != motor:
            continue  # not a state frame (a command to a motor, a query, or a register reply)
        if data[0] >> 4 not in _STATUSES or (data[1] == 0 and data[2] in (0x33, 0x55, 0xAA)):
            continue  # a register read/write/save reply shares the id (robots/damiao.is_state_frame)
        try:
            t = float(m.group("t")) if m.group("t") else float(n)
        except ValueError:
            t = float(n)
        name, model = MOTORS[motor]
        out.append({"t": t, "channel": m.group("ch"), "motor": motor, "joint": name, **decode(data, model)})
    return out


def report(frames: list[dict]) -> str:
    if not frames:
        return ("no Damiao state frames found. Check the capture is from the arm's SocketCAN link while the arm "
                "was running (replies come from ids 0x11..0x17).")
    lines = []
    for ch in sorted({f["channel"] for f in frames}):
        fs = [f for f in frames if f["channel"] == ch]
        span = max(f["t"] for f in fs) - min(f["t"] for f in fs)
        lines += [f"{ch}: {len(fs)} state frames over {span:.1f} s",
                  f"{'joint':<14}{'n':>6}{'median N*m':>12}{'peak |tau|':>12}{'% of TMAX':>11}{'max temp C':>12}{'faults':>8}"]
        for motor, (name, model) in MOTORS.items():
            js = [f for f in fs if f["motor"] == motor]
            if not js:
                continue
            taus = [f["torque_nm"] for f in js]
            peak = max(abs(x) for x in taus)
            faults = sum(1 for f in js if f["status"] >= 8)
            temp = max(max(f["temp_mos_c"], f["temp_rotor_c"]) for f in js)
            lines.append(f"{name:<14}{len(js):>6}{statistics.median(taus):>12.2f}{peak:>12.2f}"
                         f"{100 * peak / LIMITS[model][2]:>10.0f}%{temp:>12.0f}{faults:>8}")
        lines.append("")
    return "\n".join(lines).rstrip()


def listen(channel: str, seconds: float, out=sys.stdout) -> int:
    """Receive-only capture on a SocketCAN link, written in candump -L format. Sends nothing."""
    import socket
    import struct
    import time

    sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((channel,))
    sock.settimeout(0.5)
    end, n = time.time() + seconds, 0
    try:
        while time.time() < end:
            try:
                frame = sock.recv(16)
            except socket.timeout:
                continue
            can_id, dlc = struct.unpack("=IB3x", frame[:8])
            out.write(f"({time.time():.6f}) {channel} {can_id & 0x1FFFFFFF:03X}#{frame[8:8 + dlc].hex().upper()}\n")
            n += 1
    finally:
        sock.close()
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m harness.can_torque", description=__doc__.split("\n\n")[0])
    p.add_argument("capture", nargs="?", help="candump output (candump -L can_right > right.log)")
    p.add_argument("--listen", metavar="CHANNEL", help="capture from this SocketCAN link instead (receive only)")
    p.add_argument("--seconds", type=float, default=15.0, help="how long --listen captures")
    p.add_argument("--csv", default=None, help="also write every decoded state frame here")
    a = p.parse_args(argv)
    if a.listen:
        n = listen(a.listen, a.seconds)
        print(f"captured {n} frames from {a.listen}", file=sys.stderr)
        return 0 if n else 1
    if not a.capture:
        p.error("give a capture file, or --listen CHANNEL")
    try:
        frames = parse(Path(a.capture).read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report(frames))
    if a.csv and frames:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(frames[0]))
            w.writeheader()
            w.writerows(frames)
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
