"""Time the harness's local perception on real station frames: colour detector, black-ball guard, cable-end
classifier, and the full annotate pass the monitor runs before each Claude call. Usage: python bench_local.py TREE"""
import statistics
import sys
import time
from pathlib import Path

TREE = Path(sys.argv[1])
sys.path.insert(0, str(TREE))
import cv2  # noqa: E402

from harness.cable_end import CableEndClassifier  # noqa: E402
from harness.detect import TableDetector, detect  # noqa: E402

det = TableDetector.from_file(TREE / "plans" / "table_colors.yaml")
clf = CableEndClassifier(TREE / "models" / "cable_end.onnx")
data = TREE / "tests" / "data"
top = (data / "table_c_topdown.jpg").read_bytes()
im = cv2.imdecode(__import__("numpy").frombuffer(top, "uint8"), cv2.IMREAD_COLOR)
print("table_c_topdown.jpg", im.shape)
ok, b640 = cv2.imencode(".jpg", cv2.resize(im, (640, round(im.shape[0] * 640 / im.shape[1]))),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
f640 = b640.tobytes()
frames3 = {"overhead": f640, "wrist_l": (data / "cable_usba_end.jpg").read_bytes(),
           "wrist_r": (data / "cable_usbc_end.jpg").read_bytes()}
print({k: len(v) for k, v in frames3.items()}, "bytes")


def bench(name, fn, n=60):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(1000 * (time.perf_counter() - t))
    ts.sort()
    print(f"{name:52s} median {statistics.median(ts):7.1f} ms   p90 {ts[int(0.9 * len(ts))]:7.1f} ms")


bench("JPEG decode, 640 px overhead", lambda: cv2.imdecode(__import__("numpy").frombuffer(f640, "uint8"), 1))
bench("colour detector, one 640 px frame", lambda: detect(f640, det.cfg, overhead=True))
bench("black-ball guard, 2 wrist frames", lambda: det.guard({k: v for k, v in frames3.items() if k != "overhead"}))
bench("cable-end classifier, one 640 px frame", lambda: clf.ends_jpeg(f640))
bench("cable-end classifier, wrist crop (USB-A)", lambda: clf.ends_jpeg(frames3["wrist_l"]))
det.cable_ends = clf
bench("annotate 3 cameras (detector + classifier + re-encode)", lambda: det.annotate_frames(frames3), n=30)
print("cable ends found on the 640 px overhead:", [(e.label, round(e.prob, 2)) for e in clf.ends_jpeg(f640)])
