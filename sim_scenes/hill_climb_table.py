"""Hill Climb relay table for Dream's YAM MuJoCo sim, plus a ground-truth referee for rubric v4.

patch(yam_env) wraps yam_env.build_xml(): the fruit, bowl, plate and bottle are removed and the relay objects
added, keeping Dream's calibrated arms, cameras, lights and table. Sizes are from the 7 Nov station photos
(task_photos/, IMG_2680) and standard parts; positions are a plausible layout, not a measured one:

  Stage 1  a 4-port USB hub (ports on top) with 2 cables plugged in and 2 loose cables on the table
  Stage 2  10 foam blocks: 6 cubes of 6 cm and 4 cylinders of 2.5 cm x 10 cm (one alone is under 4 in)
  Stage 3  6 white golf balls, 2 black pool 8-balls and a wire bin by the right arm

World frame: table top at z = 0.76, arm bases at x = +-0.305 (left arm at +x), y = 0.24; objects sit in front
of the arms at y < 0.2. The hub and cables are in front of the RIGHT arm and the bin by the left one, because
plans/hill_climb_v4.yaml unplugs "the USB cable at the end of the hub nearest the right arm" (with the hub in
front of the left arm, Dream-1 reached across the blocks and balls: 3 DQs in 3 sim matches). Mirror the x
coordinates below if your station is the other way round. HillClimbReferee reads contacts and poses each physics step and reports the score,
the pass bars and any DQ (a gripper touching a later stage's object before the current pass bar).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any

TABLE_Z = 0.76
INCH = 0.0254
TOWER_BAR_M = 4 * INCH
FREE_STANDING_S = 3.0

HUB_POS = (-0.12, 0.02)
HUB_HALF = (0.06, 0.02, 0.015)          # x, y, z half sizes
PORT_X = (-0.0375, -0.0125, 0.0125, 0.0375)
SLOT_HALF = (0.007, 0.00325)            # USB-A opening with clearance
SLOT_DEPTH = 0.012
PLUGGED_PORTS = (0, 2)
LOOSE_PLUGS = ((-0.24, -0.06), (-0.2, -0.12))

CUBE_HALF = 0.03
CUBES = ((0.02, -0.02), (-0.04, -0.06), (0.08, -0.08), (-0.02, -0.13), (0.04, 0.06), (-0.08, -0.17))
CYL_R, CYL_HALF = 0.0125, 0.05
CYLINDERS = ((0.12, -0.02), (0.06, -0.15), (0.14, -0.12), (0.0, -0.19))

BIN_POS = (0.27, 0.08)
BIN_HALF = (0.11, 0.075, 0.04)          # inner x, y half sizes and wall half height
GOLF_R, EIGHT_R = 0.02135, 0.028575
GOLF = ((0.12, 0.1), (0.18, -0.06), (0.3, -0.1), (0.22, -0.16), (0.1, -0.2), (0.35, -0.02))
EIGHT = ((0.24, -0.06), (0.16, -0.24))

REMOVED = ("apple", "banana", "orange", "water_bottle", "bowl", "plate")
GRIPPER_BODIES = tuple(f"{side}_{part}" for side in ("left", "right")
                       for part in ("link6", "gripper", "tip_left", "tip_right"))


def _n(*values: float) -> str:
    return " ".join(f"{v:.5g}" for v in values)


def _plug(world: ET.Element, name: str, pos: tuple[float, float, float], quat: str = "1 0 0 0") -> None:
    """USB-A plug: metal tongue below the body origin, overmold above it, a short stub of cable on top."""
    body = ET.SubElement(world, "body", name=name, pos=_n(*pos), quat=quat)
    ET.SubElement(body, "freejoint", name=f"{name}_free")
    ET.SubElement(body, "geom", name=f"{name}_metal", type="box", size=_n(0.006, 0.00225, 0.006),
                  pos=_n(0, 0, -0.006), mass="0.002", rgba=".75 .75 .78 1")
    ET.SubElement(body, "geom", name=f"{name}_mold", type="box", size=_n(0.008, 0.004, 0.0125),
                  pos=_n(0, 0, 0.0125), mass="0.006", rgba=".08 .08 .09 1")
    ET.SubElement(body, "geom", name=f"{name}_tail", type="capsule", fromto=_n(0, 0, 0.025, 0, 0, 0.06),
                  size="0.0025", mass="0.004", rgba=".9 .45 .1 1")


def _hub(world: ET.Element) -> None:
    hx, hy, hz = HUB_HALF
    body = ET.SubElement(world, "body", name="hub", pos=_n(HUB_POS[0], HUB_POS[1], TABLE_Z))
    rgba = ".85 .86 .88 1"
    floor_h = 2 * hz - SLOT_DEPTH
    ET.SubElement(body, "geom", type="box", size=_n(hx, hy, floor_h / 2), pos=_n(0, 0, floor_h / 2), rgba=rgba)
    top_z = floor_h + SLOT_DEPTH / 2
    sx, sy = SLOT_HALF
    strip = (hy - sy) / 2
    for sign in (-1, 1):   # front and back strips beside the slots
        ET.SubElement(body, "geom", type="box", size=_n(hx, strip, SLOT_DEPTH / 2),
                      pos=_n(0, sign * (sy + strip), top_z), rgba=rgba)
    edges = [-hx] + [e for c in PORT_X for e in (c - sx, c + sx)] + [hx]
    for lo, hi in zip(edges[::2], edges[1::2]):   # separators between the slots
        ET.SubElement(body, "geom", type="box", size=_n((hi - lo) / 2, sy, SLOT_DEPTH / 2),
                      pos=_n((lo + hi) / 2, 0, top_z), rgba=rgba)


def _bin(world: ET.Element) -> None:
    bx, by, bh = BIN_HALF
    t = 0.0015
    body = ET.SubElement(world, "body", name="bin", pos=_n(BIN_POS[0], BIN_POS[1], TABLE_Z))
    rgba = ".95 .95 .95 1"
    ET.SubElement(body, "geom", type="box", size=_n(bx + t, by + t, t), pos=_n(0, 0, t), rgba=rgba)
    for sign in (-1, 1):
        ET.SubElement(body, "geom", type="box", size=_n(t, by + t, bh), pos=_n(sign * (bx + t), 0, bh), rgba=rgba)
        ET.SubElement(body, "geom", type="box", size=_n(bx + t, t, bh), pos=_n(0, sign * (by + t), bh), rgba=rgba)


def add_objects(world: ET.Element) -> None:
    _hub(world)
    hub_top = TABLE_Z + 2 * HUB_HALF[2]
    for i, port in enumerate(PLUGGED_PORTS):
        _plug(world, f"plugged_{i}", (HUB_POS[0] + PORT_X[port], HUB_POS[1], hub_top + 0.0005))
    lying = "0.7071068 0.7071068 0 0"   # plug axis along -y, lying on its wide face
    for i, (x, y) in enumerate(LOOSE_PLUGS):
        _plug(world, f"loose_{i}", (x, y, TABLE_Z + 0.0045), lying)
    for i, (x, y) in enumerate(CUBES):
        body = ET.SubElement(world, "body", name=f"cube_{i}", pos=_n(x, y, TABLE_Z + CUBE_HALF + 0.0005))
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="box", size=_n(CUBE_HALF, CUBE_HALF, CUBE_HALF), mass="0.008",
                      rgba=(".15 .5 .85 1" if i % 2 else ".7 .25 .75 1"))
    for i, (x, y) in enumerate(CYLINDERS):
        body = ET.SubElement(world, "body", name=f"cylinder_{i}", pos=_n(x, y, TABLE_Z + CYL_R + 0.0005),
                             quat="0.7071068 0 0.7071068 0")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="cylinder", size=_n(CYL_R, CYL_HALF), mass="0.004",
                      rgba=(".95 .85 .15 1" if i % 2 else ".7 .35 .8 1"))
    _bin(world)
    for i, (x, y) in enumerate(GOLF):
        body = ET.SubElement(world, "body", name=f"golf_{i}", pos=_n(x, y, TABLE_Z + GOLF_R + 0.0005))
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="sphere", size=str(GOLF_R), mass="0.0459", rgba=".97 .97 .95 1")
    for i, (x, y) in enumerate(EIGHT):
        body = ET.SubElement(world, "body", name=f"eight_{i}", pos=_n(x, y, TABLE_Z + EIGHT_R + 0.0005))
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "geom", type="sphere", size=str(EIGHT_R), mass="0.16", rgba=".03 .03 .03 1")


def build_xml(original_xml: str) -> str:
    root = ET.fromstring(original_xml)
    root.set("model", "Dream YAM Sim: Hill Climb relay table")
    world = root.find("worldbody")
    for body in list(world):
        if body.tag == "body" and body.get("name") in REMOVED:
            world.remove(body)
    add_objects(world)
    return ET.tostring(root, encoding="unicode")


def patch(yam_env: Any) -> None:
    """Make yam_env.YamEnv() build the relay table. Idempotent."""
    if getattr(yam_env, "_hill_climb_patched", False):
        return
    original = yam_env.build_xml
    yam_env.build_xml = lambda: build_xml(original())
    yam_env._hill_climb_patched = True


# ----------------------------------------------------------------------------------------------- referee
class HillClimbReferee:
    """Ground-truth rubric v4 scoring from the sim state. Call update(model, data) after each physics step."""

    def __init__(self, model: Any) -> None:
        import mujoco
        self._mj = mujoco
        self._model = model
        names = [model.body(i).name for i in range(model.nbody)]
        self.id = {n: i for i, n in enumerate(names)}
        self.gripper = {self.id[n] for n in GRIPPER_BODIES if n in self.id}
        self.groups = {
            1: {self.id[n] for n in names if n.startswith(("plugged_", "loose_"))},
            2: {self.id[n] for n in names if n.startswith(("cube_", "cylinder_"))},
            3: {self.id[n] for n in names if n.startswith(("golf_", "eight_"))},
        }
        self.hub_top = TABLE_Z + 2 * HUB_HALF[2]
        self.stage1_at: float | None = None
        self.stage2_at: float | None = None
        self.dq: str | None = None
        self.first_touch: dict[str, float] = {}
        self._last_block_touch = -math.inf
        self._t = 0.0

    def _pos(self, data: Any, name: str) -> tuple[float, float, float]:
        p = data.xpos[self.id[name]]
        return float(p[0]), float(p[1]), float(p[2])

    def _unplugged(self, data: Any) -> int:
        out = 0
        for i, port in enumerate(PLUGGED_PORTS):
            x, y, z = self._pos(data, f"plugged_{i}")
            in_slot = math.hypot(x - HUB_POS[0] - PORT_X[port], y - HUB_POS[1]) < 0.006
            if not in_slot or z - 0.012 > self.hub_top + 0.001:   # metal tongue fully clear of the slot
                out += 1
        return out

    def _seated(self, data: Any) -> int:
        free = [p for p in range(len(PORT_X)) if p not in PLUGGED_PORTS]
        seated = 0
        for i in range(len(LOOSE_PLUGS)):
            x, y, z = self._pos(data, f"loose_{i}")
            if any(math.hypot(x - HUB_POS[0] - PORT_X[p], y - HUB_POS[1]) < 0.003 for p in free) \
                    and abs(z - self.hub_top) < 0.004:
                seated += 1
        return seated

    def _tower_m(self, data: Any) -> float:
        top = TABLE_Z
        for b in self.groups[2]:
            # A block's highest point, from its geom's world AABB (cheap and orientation-proof).
            gid = self._geom_of(b)
            half = self._half_height(data, gid)
            top = max(top, float(data.geom_xpos[gid][2]) + half)
        return top - TABLE_Z

    def _geom_of(self, body: int) -> int:
        if not hasattr(self, "_geom_cache"):
            self._geom_cache: dict[int, int] = {}
        return self._geom_cache.setdefault(body, int(self._model.body_geomadr[body]))

    def _half_height(self, data: Any, gid: int) -> float:
        size, kind = self._model.geom_size[gid], self._model.geom_type[gid]
        R = data.geom_xmat[gid].reshape(3, 3)
        if kind == self._mj.mjtGeom.mjGEOM_BOX:
            return float(sum(abs(R[2, k]) * size[k] for k in range(3)))
        r, h = float(size[0]), float(size[1])   # cylinder: radius, half-length along local z
        axis_z = abs(R[2, 2])
        return h * axis_z + r * math.sqrt(max(0.0, 1 - axis_z * axis_z))

    def _in_bin(self, data: Any, prefix: str) -> int:
        bx, by, bh = BIN_HALF
        count = 0
        for name in self.id:
            if name.startswith(prefix):
                x, y, z = self._pos(data, name)
                if abs(x - BIN_POS[0]) < bx and abs(y - BIN_POS[1]) < by and z < TABLE_Z + 2 * bh:
                    count += 1
        return count

    def update(self, model: Any, data: Any) -> None:
        self._model = model
        self._t = float(data.time)
        touched: set[int] = set()
        for k in range(data.ncon):
            c = data.contact[k]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            for a, b in ((b1, b2), (b2, b1)):
                if a in self.gripper:
                    touched.add(b)
        for b in touched:
            self.first_touch.setdefault(model.body(b).name, round(self._t, 2))
        if touched & self.groups[2]:
            self._last_block_touch = self._t
        if self.stage1_at is None and self._unplugged(data) >= 1:
            self.stage1_at = self._t
        if self.dq is None:
            if touched & self.groups[2] and self.stage1_at is None:
                self.dq = f"block touched at {self._t:.1f} s before a cable was unplugged"
            elif touched & self.groups[3] and self.stage2_at is None:
                self.dq = f"ball touched at {self._t:.1f} s before the tower passed"
        if (self.stage1_at is not None and self.stage2_at is None and self._tower_m(data) >= TOWER_BAR_M
                and self._t - self._last_block_touch >= FREE_STANDING_S):
            self.stage2_at = self._t

    def report(self, data: Any) -> dict[str, Any]:
        unplugged, seated = self._unplugged(data), self._seated(data)
        tower = self._tower_m(data)
        golf, eight = self._in_bin(data, "golf_"), self._in_bin(data, "eight_")
        points = 0
        if unplugged:
            points += 10 + 5 * (unplugged - 1)
        points += 20 * min(seated, 4)
        if self.stage2_at is not None:
            points += 20 + 3 * int(tower / INCH)
        points += 5 * golf
        return {
            "t": round(self._t, 2), "points": 0 if self.dq else points, "dq": self.dq,
            "loses": bool(self.dq or eight), "unplugged": unplugged, "seated": seated,
            "tower_in": round(tower / INCH, 2), "stage1_at": self.stage1_at, "stage2_at": self.stage2_at,
            "golf_in_bin": golf, "eight_in_bin": eight, "first_touch": dict(self.first_touch),
        }
