import sys, av, glob, random, json, cv2, numpy as np
sys.path.insert(0, 'usbc')
from candidates import ferrules, crop
random.seed(1)
vids = sorted(v for v in glob.glob('nuc_data/hackathon_data/*/videos/chunk-000/observation.images.*/episode_*.mp4') if '-eef' not in v)
random.shuffle(vids)
recs = []
n = 0
for v in vids[:80]:
    try:
        with av.open(v) as c:
            frames = []
            for i, f in enumerate(c.decode(video=0)):
                if i % 45 == 7:
                    frames.append((i, f.to_ndarray(format='bgr24')))
                if len(frames) >= 4:
                    break
    except Exception:
        continue
    for i, img in frames:
        for cd in ferrules(img):
            cv2.imwrite(f'usbc/real/cands/{n:05d}.png', crop(img, cd))
            recs.append({"id": n, "src": v.replace(chr(92), '/'), "frame": i, **{k: round(float(x), 2) for k, x in cd.items()}})
            n += 1
for p in sorted(glob.glob('table_photos/IMG_270*.jpg')) + ['table_photos/IMG_2697.jpg', 'table_photos/IMG_2698.jpg', 'table_photos/IMG_2699.jpg']:
    img = cv2.imread(p)
    for cd in ferrules(img, work_w=800):
        cv2.imwrite(f'usbc/real/cands/{n:05d}.png', crop(img, cd))
        recs.append({"id": n, "src": p, "frame": 0, **{k: round(float(x), 2) for k, x in cd.items()}})
        n += 1
json.dump(recs, open('usbc/real/cands.json', 'w'))
print(n, 'candidates')
for sheet in range(0, n, 100):
    g = np.full((10 * 110, 10 * 100, 3), 255, np.uint8)
    for k in range(sheet, min(sheet + 100, n)):
        r, c = divmod(k - sheet, 10)
        im = cv2.imread(f'usbc/real/cands/{k:05d}.png')
        g[r*110+12:r*110+108, c*100+2:c*100+98] = im
        cv2.putText(g, str(k), (c*100+3, r*110+10), 0, 0.35, (0, 0, 0), 1)
    cv2.imwrite(f'usbc/real/sheet_{sheet//100:02d}.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 85])
