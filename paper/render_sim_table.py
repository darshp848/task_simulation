"""Render the cleaned-up Hill Climb sim table (cloud-sims 9c89ae3) for the paper. No policy, no network.

usage: python render_sim_table.py <cloud-sims dir> <out dir>
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(repo))
import sim_station  # noqa: E402
from sim_scenes import hill_climb_table  # noqa: E402

yam_env, _dc = sim_station.load_example(None)
hill_climb_table.patch(yam_env)
env = yam_env.YamEnv()
env.reset()
for _ in range(200):           # let the balls and cables settle
    mujoco.mj_step(env.model, env.data)

m, d = env.model, env.data
m.vis.global_.offwidth, m.vis.global_.offheight = 1600, 900
print("cameras:", [m.camera(i).name for i in range(m.ncam)])

# The policy's own cameras, at the station's 640x360.
r = mujoco.Renderer(m, 360, 640)
for i in range(m.ncam):
    name = m.camera(i).name
    r.update_scene(d, camera=name)
    Image.fromarray(r.render()).save(out / f"cam_{name}.png")
r.close()

# A high-resolution free view of the whole table, for the paper.
r = mujoco.Renderer(m, 900, 1600)
names = ["hub", "cube_0", "cube_1", "cube_2", "bin", "golf_0", "golf_1", "golf_2", "eight_0", "left_base", "right_base"]
c = np.mean([d.body(n).xpos for n in names], axis=0)
print("centre", c, {n: d.body(n).xpos.round(2).tolist() for n in names})
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(m, cam)
cam.lookat[:] = [c[0] - 0.08, c[1] - 0.02, 0.76]
for tag, dist, az, el in (("persp", 1.05, 270, -38), ("persp2", 1.0, 250, -30), ("top", 0.95, 90, -89)):
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    r.update_scene(d, camera=cam)
    Image.fromarray(r.render()).save(out / f"view_{tag}.png")
r.close()
print("bodies:", [m.body(i).name for i in range(m.nbody)][:80])
