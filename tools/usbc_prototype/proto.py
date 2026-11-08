"""Prototype: find orange ferrules and measure the metal tip beyond each."""
import sys
import numpy as np, cv2

def ends(bgr, work_w=800, debug=None):
    h0, w0 = bgr.shape[:2]
    s = work_w / w0
    im = cv2.resize(bgr, (work_w, round(h0 * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    orange = ((H >= 4) & (H <= 22) & (S >= 120) & (V >= 70)).astype(np.uint8)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(orange)
    out = []
    dark = (V < 70)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 60:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1).astype(float)
        c = pts.mean(0)
        ev, evec = np.linalg.eigh(np.cov((pts - c).T))
        ax = evec[:, 1]; perp = evec[:, 0]
        proj = (pts - c) @ ax; pp = (pts - c) @ perp
        L = proj.max() - proj.min(); W = pp.max() - pp.min()
        fill = area / max(L * W, 1)
        if not (1.1 <= L / max(W, 1) <= 3.5 and fill > 0.6):
            continue        # cable runs are long and thin; ferrules are compact, filled rectangles
        # which end has the black strain relief?
        def darkness(sign):
            vals = []
            for t in np.linspace(0.05, 0.4, 8) * L:
                for u in np.linspace(-0.3, 0.3, 5) * W:
                    p = c + ax * sign * (L / 2 + t) + perp * u
                    x, y = int(round(p[0])), int(round(p[1]))
                    if 0 <= x < im.shape[1] and 0 <= y < im.shape[0]:
                        vals.append(dark[y, x])
            return np.mean(vals) if vals else 0
        sign = 1 if darkness(1) < darkness(-1) else -1   # tip on the less dark side
        # table brightness: ring around the ferrule
        ring = []
        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            p = c + (ax * np.cos(a) * L + perp * np.sin(a) * W) * 1.6
            x, y = int(p[0]), int(p[1])
            if 0 <= x < im.shape[1] and 0 <= y < im.shape[0]:
                ring.append(V[y, x])
        vt = np.median(ring) if ring else 200
        # walk outwards from the ferrule's tip end; at each step the width of "not table, not orange" pixels
        widths = []
        for t in np.arange(1, 1.5 * L, 1.0):
            row = 0
            for u in np.arange(-0.8 * W, 0.8 * W + 1, 1.0):
                p = c + ax * sign * (proj.max() if sign > 0 else -proj.min()) + ax * sign * t + perp * u
                x, y = int(round(p[0])), int(round(p[1]))
                if not (0 <= x < im.shape[1] and 0 <= y < im.shape[0]):
                    continue
                metal = abs(V[y, x] - vt) > 35 and S[y, x] < 110
                row += metal
            widths.append(row)
        widths = np.array(widths, float)
        on = widths >= 0.3 * W
        # tip length = first run of "on" steps starting within the first few steps
        tl = 0
        started = False
        for k, o in enumerate(on):
            if o:
                started = True; tl = k + 1
            elif started:
                break
            elif k > 0.15 * L:
                break
        tw = np.median(widths[:tl]) if tl else 0
        out.append(dict(c=(c / s).round().tolist(), L=L, W=W, tip_len=tl / L, tip_w=tw / W, ax=ax, sign=sign,
                        box=(c, ax, perp, L, W, tl)))
    if debug is not None:
        for e in out:
            c, ax, perp, L, W, tl = e['box']
            tipc = c + ax * e['sign'] * (L / 2 + tl / 2)
            cv2.circle(im, tuple(int(v) for v in tipc), 6, (0, 255, 0), 2)
            cv2.putText(im, f"l{e['tip_len']:.2f} w{e['tip_w']:.2f}", tuple(int(v) for v in c + 12), 0, 0.5,
                        (255, 0, 0), 2)
        cv2.imwrite(debug, im)
    return out

if __name__ == "__main__":
    for path in sys.argv[1:]:
        res = ends(cv2.imread(path), debug=path.rsplit('.', 1)[0] + '_ends.jpg')
        for e in res:
            print(path.split('/')[-1], [round(v) for v in e['c']], f"L{e['L']:.0f} W{e['W']:.0f} tip_len {e['tip_len']:.2f} tip_w {e['tip_w']:.2f}")
