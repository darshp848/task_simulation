import sys, cv2, numpy as np
ids = [int(x) for x in sys.argv[2:]]
cols = 6
rows = (len(ids) + cols - 1) // cols
g = np.full((rows * 210, cols * 200, 3), 255, np.uint8)
for k, i in enumerate(ids):
    r, c = divmod(k, cols)
    im = cv2.resize(cv2.imread(f'usbc/real/cands/{i:05d}.png'), (192, 192), interpolation=cv2.INTER_CUBIC)
    g[r*210+16:r*210+208, c*200+4:c*200+196] = im
    cv2.putText(g, str(i), (c*200+6, r*210+13), 0, 0.45, (0, 0, 0), 1)
cv2.imwrite(sys.argv[1], g)
