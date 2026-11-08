"""Local colour-blob detector for the relay table: golf balls, the black 8-balls, blocks and cylinders.

It runs on the laptop in a few milliseconds per frame, with no API call, so it serves two jobs the Claude monitor
is too slow or too fragile for:

1. A fast black-ball guard. Every harness poll on a guarded stage looks at the wrist cameras; a black ball between
   the fingers counts as a guard "yes" at once. This also keeps the guard answering when the Claude API is down.
2. Boxes on the frames the monitor sees, labelled with class, colour and (for cylinders) upright/lying, so the
   monitor counts balls in the bin, black balls and tower blocks from marked objects instead of guessing.

It does NOT change what Dream-1 sees or is told: the station builds the policy's camera input itself, the fine-tune
was trained on plain frames, and prompts must stay the recorded task texts word for word.

Colours live in a YAML file (plans/table_colors.yaml). Tune them on a real frame of the table:
    python -m harness.detect plans/table_colors.yaml frame.jpg --out boxed.jpg
"""

from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

try:
    from scipy import ndimage as _ndi
except ImportError:   # pragma: no cover
    _ndi = None

SMALL_BALL_FRAC = 0.004  # a golf ball in the overhead view is ~0.05-0.1% of the image
WORK_W = 640           # frames are shrunk to this width before detection
COLORS = {"avoid": (230, 20, 20), "target": (20, 200, 60), "info": (30, 120, 230)}


@dataclass
class ClassSpec:
    name: str
    hsv_lo: tuple[int, int, int]
    hsv_hi: tuple[int, int, int]
    min_frac: float = 0.0005       # blob area as a fraction of the image
    max_frac: float = 0.15
    label: str = ""                # colour name, e.g. "blue"
    # which shapes of this colour to report, and how: kind (ball | cube | stick) -> role (avoid | target | info).
    # Empty = every kind, role info. Black lists only ball, so the black gripper fingers are never reported.
    kinds: dict[str, str] = field(default_factory=dict)
    small_ball: bool = False       # this colour only comes as balls: read a small compact blob as a ball


@dataclass
class Box:
    cls: str
    x0: float
    y0: float
    x1: float
    y1: float                      # fractions of the image, 0-1
    frac: float
    orientation: str | None = None
    role: str = "info"

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def text(self) -> str:
        return self.cls + (f" {self.orientation}" if self.orientation else "")


@dataclass
class DetectConfig:
    classes: list[ClassSpec]
    guard_class: str = "black_ball"
    wrist_roles: list[str] = field(default_factory=lambda: ["wrist"])
    grip_roi: tuple[float, float, float, float] = (0.25, 0.35, 0.75, 1.0)   # x0 y0 x1 y1 between the fingers
    grip_min_frac: float = 0.01   # a ball this big in the wrist view is in or at the fingers
    table_roi: tuple[float, float, float, float] | None = None   # overhead view: ignore blobs outside (people, chairs)
    annotate: bool = True


def load_config(path: str | Path) -> DetectConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    classes = []
    for name, c in (raw.get("classes") or {}).items():
        lo, hi = c["hsv"]
        classes.append(ClassSpec(name=name, hsv_lo=tuple(lo), hsv_hi=tuple(hi),
                                 min_frac=float(c.get("min_frac", 0.0005)), max_frac=float(c.get("max_frac", 0.15)),
                                 label=c.get("label", name), kinds=dict(c.get("kinds") or {}),
                                 small_ball=bool(c.get("small_ball", False))))
    g = raw.get("guard") or {}
    return DetectConfig(classes=classes, guard_class=g.get("class", "black_ball"),
                        wrist_roles=list(g.get("wrist_roles", ["wrist"])),
                        grip_roi=tuple(g.get("grip_roi", (0.25, 0.35, 0.75, 1.0))),
                        grip_min_frac=float(g.get("grip_min_frac", 0.01)), annotate=bool(raw.get("annotate", True)),
                        table_roi=tuple(raw["table_roi"]) if raw.get("table_roi") else None)


def _load(jpeg: bytes) -> Image.Image | None:
    try:
        im = Image.open(io.BytesIO(jpeg))
        im.load()
        return im.convert("RGB")
    except Exception:   # noqa: BLE001 - a bad frame is simply not looked at
        return None


def _hsv(im: Image.Image) -> np.ndarray:
    w, h = im.size
    small = im.resize((WORK_W, max(1, round(h * WORK_W / w))))
    return np.asarray(small.convert("HSV"), dtype=np.int16)


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if _ndi is not None:
        return _ndi.label(mask)
    lab = np.zeros(mask.shape, dtype=np.int32)   # small 4-connected flood fill for machines without scipy
    n = 0
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x]:
            continue
        n += 1
        stack = [(y, x)]
        lab[y, x] = n
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not lab[ny, nx]:
                    lab[ny, nx] = n
                    stack.append((ny, nx))
    return lab, n


def _in_range(hsv: np.ndarray, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = (h >= lo[0]) & (h <= hi[0]) if lo[0] <= hi[0] else (h >= lo[0]) | (h <= hi[0])   # red wraps around
    return hue & (s >= lo[1]) & (s <= hi[1]) & (v >= lo[2]) & (v <= hi[2])


def _kind(ys: np.ndarray, xs: np.ndarray) -> tuple[str | None, str | None]:
    """(kind, orientation) of one blob from its pixels, independent of rotation.
    ball: compact and fills its circumscribed circle; cube: compact but square (or an upright stick seen from above);
    stick: long and thin, i.e. lying on its side."""
    n = len(xs)
    if n < 6:
        return None, None
    pts = np.stack([xs, ys], axis=1).astype(float)
    c = pts.mean(axis=0)
    ev = np.linalg.eigvalsh(np.cov((pts - c).T)) if n > 2 else np.array([1.0, 1.0])
    elong = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
    rmax = float(np.sqrt(((pts - c) ** 2).sum(axis=1).max())) + 0.5
    roundness = n / (np.pi * rmax * rmax)          # disc ~1, square ~0.64, thin bar small
    if elong >= 2.2:
        return "stick", "lying"
    if elong <= 1.5 and roundness >= 0.72:
        return "ball", None
    if elong <= 1.6:
        return "cube", None
    return None, None


def detect(jpeg: bytes, cfg: DetectConfig, *, overhead: bool = False) -> list[Box]:
    im = _load(jpeg)
    if im is None:
        return []
    hsv = _hsv(im)
    H, W = hsv.shape[:2]
    total = float(H * W)
    out: list[Box] = []
    for spec in cfg.classes:
        mask = _in_range(hsv, spec.hsv_lo, spec.hsv_hi)
        if _ndi is not None:   # close specular highlights and the cubes' holes, so shapes read whole
            mask = _ndi.binary_fill_holes(_ndi.binary_closing(mask, iterations=2))
        lab, n = _label(mask)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        for i in range(1, n + 1):
            area = sizes[i]
            frac = area / total
            if not spec.min_frac <= frac <= spec.max_frac:
                continue
            ys, xs = np.nonzero(lab == i)
            kind, orient = _kind(ys, xs)
            if kind == "cube" and spec.small_ball and frac <= SMALL_BALL_FRAC:
                kind = "ball"   # a colour that only comes as balls: a small, half-lit ball reads square-ish
                                # (the size cap keeps the white bin out; a held ball fills the wrist view and is round)
            if kind is None or (spec.kinds and kind not in spec.kinds):
                continue
            role = spec.kinds.get(kind, "info")
            label = f"{spec.label or spec.name} {kind}"
            out.append(Box(label, xs.min() / W, ys.min() / H, (xs.max() + 1) / W, (ys.max() + 1) / H, frac,
                           orient, role))
    if overhead and cfg.table_roi:
        x0, y0, x1, y1 = cfg.table_roi
        out = [b for b in out if x0 <= b.cx <= x1 and y0 <= b.cy <= y1]
    return out


def held(boxes: list[Box], cfg: DetectConfig) -> Box | None:
    """A ball the plan must never bin (role avoid) between the fingers of a wrist camera, big enough to be held."""
    rx0, ry0, rx1, ry1 = cfg.grip_roi
    for b in boxes:
        if b.role == "avoid" and b.frac >= cfg.grip_min_frac and rx0 <= b.cx <= rx1 and ry0 <= b.cy <= ry1:
            return b
    return None


def annotate(jpeg: bytes, boxes: list[Box]) -> bytes:
    im = _load(jpeg)
    if im is None or not boxes:
        return jpeg
    d = ImageDraw.Draw(im)
    W, H = im.size
    width = max(2, W // 200)
    for b in boxes:
        col = COLORS.get(b.role, COLORS["info"])
        xy = (b.x0 * W, b.y0 * H, b.x1 * W, b.y1 * H)
        d.rectangle(xy, outline=col, width=width)
        d.text((xy[0] + 2, max(0, xy[1] - 12)), b.text().upper() if b.role == "avoid" else b.text(), fill=col)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


class TableDetector:
    """What the runner and the monitor use."""

    def __init__(self, cfg: DetectConfig) -> None:
        self.cfg = cfg

    @classmethod
    def from_file(cls, path: str | Path) -> TableDetector:
        return cls(load_config(path))

    def _overhead(self, role: str) -> bool:
        return not any(w in role for w in self.cfg.wrist_roles)

    def boxes(self, frames: dict[str, bytes]) -> dict[str, list[Box]]:
        return {role: detect(j, self.cfg, overhead=self._overhead(role)) for role, j in frames.items()}

    def guard(self, frames: dict[str, bytes]) -> tuple[bool | None, str]:
        """(True, why) if a wrist camera shows the guard class between the fingers; (False, ...) if the wrist views
        were readable and showed none; (None, ...) if no wrist view could be read."""
        looked = False
        for role, jpeg in frames.items():
            if not any(w in role for w in self.cfg.wrist_roles):
                continue
            boxes = detect(jpeg, self.cfg)
            if _load(jpeg) is None:
                continue
            looked = True
            b = held(boxes, self.cfg)
            if b is not None:
                return True, f"local detector: {b.cls} between the fingers in {role} ({b.frac:.1%} of the view)"
        return (False, "local detector: none held") if looked else (None, "local detector: no wrist view")

    def annotate_frames(self, frames: dict[str, bytes]) -> dict[str, bytes]:
        if not self.cfg.annotate:
            return frames
        return {role: annotate(j, detect(j, self.cfg, overhead=self._overhead(role))) for role, j in frames.items()}

    def legend(self) -> str:
        avoid = [f"{c.label or c.name} {k}" for c in self.cfg.classes for k, r in c.kinds.items() if r == "avoid"]
        return ("Boxes on the images come from a local colour detector and can miss or mislabel objects: "
                f"RED boxes = {', '.join(avoid) or 'objects to avoid'} (never pick these); green = targets; "
                "blue = other objects (blocks: colour, cylinders: upright or lying).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the table detector on one image and draw the boxes.")
    ap.add_argument("config")
    ap.add_argument("image")
    ap.add_argument("--out", default="boxed.jpg")
    ap.add_argument("--wrist", action="store_true", help="the image is a wrist view (no table_roi)")
    args = ap.parse_args(argv)
    det = TableDetector.from_file(args.config)
    jpeg = Path(args.image).read_bytes()
    boxes = detect(jpeg, det.cfg, overhead=not args.wrist)
    for b in boxes:
        print(f"{b.text():24s} x {b.x0:.2f}-{b.x1:.2f} y {b.y0:.2f}-{b.y1:.2f} area {b.frac:.2%}")
    Path(args.out).write_bytes(annotate(jpeg, boxes))
    print(f"{len(boxes)} boxes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
