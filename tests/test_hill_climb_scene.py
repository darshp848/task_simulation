"""Ground-truth referee for the simulated Hill Climb table: place objects by hand and check rubric v4 scoring."""

import importlib.util
import sys
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
spec = importlib.util.find_spec("dream_examples.yam_mujoco")
if spec is None or not spec.submodule_search_locations:
    pytest.skip("Dream's yam_mujoco example is not installed", allow_module_level=True)
sys.path.insert(0, str(Path(list(spec.submodule_search_locations)[0])))
import yam_env  # noqa: E402

from sim_scenes import hill_climb_table as h  # noqa: E402


@pytest.fixture(scope="module")
def env():
    h.patch(yam_env)
    return yam_env.YamEnv()


def place(env, body, xyz, quat=(1, 0, 0, 0)):
    m, d = env.model, env.data
    adr = m.jnt_qposadr[m.body_jntadr[m.body(body).id]]
    d.qpos[adr:adr + 3] = xyz
    d.qpos[adr + 3:adr + 7] = quat
    d.qvel[m.jnt_dofadr[m.body_jntadr[m.body(body).id]]:][:6] = 0


def settle(env, ref, seconds):
    hold = env.state()
    for _ in range(round(seconds / 0.05)):
        env.step(hold["left"], hold["right"])
        ref.update(env.model, env.data)


def test_start_scene_scores_nothing_and_is_stable(env):
    env.reset()
    ref = h.HillClimbReferee(env.model)
    settle(env, ref, 3)
    r = ref.report(env.data)
    assert r["unplugged"] == 0 and r["seated"] == 0 and r["golf_in_bin"] == 0 and r["eight_in_bin"] == 0
    assert r["dq"] is None and not r["loses"] and r["points"] == 0
    assert r["tower_in"] == pytest.approx(2 * h.CUBE_HALF / h.INCH, abs=0.1)


def test_full_relay_scores_by_rubric(env):
    env.reset()
    ref = h.HillClimbReferee(env.model)
    hx, hy = h.HUB_POS
    # Stage 1: pull one plugged cable out, seat one loose cable in a free port.
    place(env, "plugged_0", (h.HUB_POS[0] - 0.18, -0.16, h.TABLE_Z + 0.03))   # out, clear of everything
    free_port = next(p for p in range(4) if p not in h.PLUGGED_PORTS)
    place(env, "loose_0", (hx + h.PORT_X[free_port], hy, ref.hub_top + 0.0005))
    settle(env, ref, 1)
    assert ref.stage1_at is not None
    # Stage 2: two cubes stacked is 12 cm, over the 4-inch bar; free-standing for 3 s.
    place(env, "cube_1", (*h.CUBES[0], h.TABLE_Z + 3 * h.CUBE_HALF + 0.001))   # on top of cube_0
    settle(env, ref, h.FREE_STANDING_S + 1)
    assert ref.stage2_at is not None
    # Stage 3: two golf balls in the bin.
    for name, dx in (("golf_0", -0.04), ("golf_1", 0.04)):
        place(env, name, (h.BIN_POS[0] + dx, h.BIN_POS[1], h.TABLE_Z + h.GOLF_R + 0.01))
    settle(env, ref, 1)
    r = ref.report(env.data)
    assert (r["unplugged"], r["seated"], r["golf_in_bin"], r["dq"], r["loses"]) == (1, 1, 2, None, False)
    assert r["tower_in"] == pytest.approx(4.72, abs=0.1)
    assert r["points"] == 10 + 20 + 20 + 3 * 4 + 5 * 2


def test_eight_ball_in_bin_loses(env):
    env.reset()
    ref = h.HillClimbReferee(env.model)
    place(env, "eight_0", (h.BIN_POS[0], h.BIN_POS[1], h.TABLE_Z + h.EIGHT_R + 0.01))
    settle(env, ref, 1)
    r = ref.report(env.data)
    assert r["eight_in_bin"] == 1 and r["loses"]


def test_touching_a_block_before_unplugging_is_a_dq(env):
    env.reset()
    ref = h.HillClimbReferee(env.model)
    tip = env.data.body("left_tip_left").xpos.copy()
    place(env, "cube_0", (tip[0], tip[1], tip[2]))   # overlap a fingertip: a gripper-block contact
    env.step(env.state()["left"], env.state()["right"])
    ref.update(env.model, env.data)
    r = ref.report(env.data)
    assert r["dq"] and "block" in r["dq"] and r["points"] == 0 and r["loses"]
