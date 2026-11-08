"""harness.can_torque: decoding Damiao state frames from candump captures."""

from harness import can_torque as ct


def _encode(motor, status, pos_rad, vel, tau, model):
    pmax, vmax, tmax = ct.LIMITS[model]
    f = lambda x, lim, bits: round((x + lim) * ((1 << bits) - 1) / (2 * lim))  # noqa: E731
    q, dq, t = f(pos_rad, pmax, 16), f(vel, vmax, 12), f(tau, tmax, 12)
    return bytes([status << 4 | motor, q >> 8, q & 0xFF, dq >> 4, ((dq & 0xF) << 4) | (t >> 8), t & 0xFF, 40, 45])


def test_decode_round_trip_and_both_candump_formats(tmp_path, capsys):
    a = _encode(2, 1, -0.5, 0.2, 12.5, "metal_j2").hex().upper()
    b = _encode(7, 1, 1.0, 0.0, -0.8, "metal_jhi")
    lines = [f"(1699999999.100000) can_right 012#{a}",
             "  can_right  007   [8]  01 02 03 04 05 06 07 08",                       # a command to motor 7: skipped
             "  can_right  017   [8]  " + " ".join(f"{x:02X}" for x in b),
             f"(1699999999.200000) can_right 012#{_encode(2, 1, 0, 0, 0, 'metal_j2')[:1].hex()}0033000000000000"]
    frames = ct.parse(lines)
    assert [f["joint"] for f in frames] == ["shoulder_lift", "gripper"]
    assert abs(frames[0]["torque_nm"] - 12.5) < 0.1 and abs(frames[1]["torque_nm"] + 0.8) < 0.02
    p = tmp_path / "cap.log"
    p.write_text("\n".join(lines))
    assert ct.main([str(p), "--csv", str(tmp_path / "o.csv")]) == 0
    out = capsys.readouterr().out
    assert "shoulder_lift" in out and "gripper" in out


def test_empty_capture(tmp_path):
    p = tmp_path / "e.log"
    p.write_text("nothing here\n")
    assert ct.main([str(p)]) == 1
