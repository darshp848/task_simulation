"""harness.force: Metal arm kinematics, torque -> hand force, and the trace report (synthetic traces)."""

import csv
import math

import pytest

from harness import force


Q = [0.2, -0.9, 1.3, -0.4, 0.3, 0.5]


def test_tool_point_at_zero_matches_link_sum():
    # All joints zero: upper arm (-0.28 m) and forearm (+0.215 m) fold back on each other, so the wrist sits close
    # to the shoulder; unfolding the elbow by 180 deg puts it about 0.28 + 0.215 m out.
    folded = force.tool_point([0.0] * 6)
    assert 0.15 < math.dist(folded, (0, 0, 0)) < 0.25
    opened = force.tool_point([0.0, 0.0, math.pi, 0.0, 0.0, 0.0])
    assert math.dist(opened, folded) > 0.35


def test_force_round_trip_through_jacobian_transpose():
    f = [3.0, -12.0, 5.0]
    jac = force.jacobian(Q)
    tau = [sum(jac[i][j] * f[i] for i in range(3)) for j in range(6)]
    got = force.hand_force(Q, tau, damping=0.0)
    assert got == pytest.approx(f, abs=1e-3)


def _trace(path, rows):
    cols = (["t_s", "chunk_id", "row_index", "ramp_alpha", "clamped"]
            + [f"measured_l{i}" for i in range(6)] + [f"measured_r{i}" for i in range(6)] + ["gripper_l", "gripper_r"]
            + [f"commanded_l{i}" for i in range(6)] + [f"commanded_r{i}" for i in range(6)]
            + ["cmd_gripper_l", "cmd_gripper_r", "ee_speed_l_m_s", "ee_speed_r_m_s", "commands_written"])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, 0.0) for c in cols})


def test_trace_report_finds_a_pull_and_a_slipping_grip(tmp_path, capsys):
    # 2 s of free motion, then 3 s where the hand is stuck: commanded leads measured by a known joint offset,
    # and the gripper is commanded closed (0) with the jaw at 0.3 open (squeezing at the hold limit).
    want_f = [0.0, 0.0, 15.0]   # 15 N upward pull at the hand
    jac = force.jacobian(Q)
    tau = [sum(jac[i][j] * want_f[i] for i in range(3)) for j in range(6)]
    kp = [j[1] for j in force.JOINTS]
    rows = []
    for n in range(100):
        t = n * 0.05
        stuck = t >= 2.0
        r = {"t_s": t, "gripper_r": 0.3 if stuck else 0.8, "cmd_gripper_r": 0.0 if stuck else 0.8}
        for i in range(6):
            r[f"measured_r{i}"] = Q[i]
            r[f"commanded_r{i}"] = Q[i] + (tau[i] / kp[i] if stuck else 0.0)
        rows.append(r)
    path = tmp_path / "last-run.csv"
    _trace(path, rows)
    out, base = force.analyse(force.load_trace(path), "r")
    assert base == pytest.approx([0.0] * 6, abs=1e-9)
    late = [r for r in out if r["t_s"] > 2.1]
    assert late[0]["force"] == pytest.approx(want_f, abs=0.05)
    assert all(r["grip_capped"] for r in late)
    assert force.main([str(path), "--arm", "r", "--from", "2.5", "--csv", str(tmp_path / "o.csv")]) == 0
    text = capsys.readouterr().out
    assert "hand force estimate: median 15.0 N" in text
    assert "slips in the jaws" in text and "gripper_hold_torque_nm" in text
    assert (tmp_path / "o.csv").read_text().count("\n") == 101


def test_bad_trace(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    assert force.main([str(p)]) == 2
