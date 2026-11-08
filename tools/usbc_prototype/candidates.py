"""Candidate cable ends: compact orange blobs (the anodised ferrules), and square crops around them.

Shared by the training-data tools and the harness classifier (harness/cable_end.py copies the same logic, numpy +
OpenCV only). A ferrule is a filled, roughly rectangular orange blob; the cable itself is a long thin run and the
ribs of the strain relief are small slivers.
"""

from __future__ import annotations

import numpy as np
import cv2

WORK_W = 640
CROP = 96          # classifier input size (px)
CROP_SCALE = 2.6   # crop side = this x the blob's long side, so the tip and the relief are both in view


def ferrules(bgr: np.ndarray, work_w: int = WORK_W) -> list[dict]:
    """[{cx, cy, side} in the input image's pixels], one per candidate ferrule."""
    h0, w0 = bgr.shape[:2]
    s = work_w / w0
    im = cv2.resize(bgr, (work_w, max(1, round(h0 * s))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((H >= 3) & (H <= 22) & (S >= 110) & (V >= 60)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    total = im.shape[0] * im.shape[1]
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 30 or area > 0.08 * total:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (cx, cy), (w, h), _ang = cv2.minAreaRect(pts)
        L, W = max(w, h), max(1.0, min(w, h))
        fill = area / max(L * W, 1.0)
        if L / W > 4.0 or fill < 0.5:
            continue
        out.append({"cx": cx / s, "cy": cy / s, "side": max(L * CROP_SCALE, 24) / s, "fill": fill, "aspect": L / W})
    return out


def crop(bgr: np.ndarray, cand: dict, size: int = CROP) -> np.ndarray:
    side = cand["side"]
    x0, y0 = cand["cx"] - side / 2, cand["cy"] - side / 2
    M = np.float32([[size / side, 0, -x0 * size / side], [0, size / side, -y0 * size / side]])
    return cv2.warpAffine(bgr, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
