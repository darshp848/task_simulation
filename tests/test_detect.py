"""harness.detect on station C's real table (top-down photo crop, 7 Nov) and on synthetic wrist frames."""

import io
from pathlib import Path

from PIL import Image, ImageDraw

from harness.detect import TableDetector, detect, load_config
from harness.verifiers import valid_jpeg

ROOT = Path(__file__).resolve().parents[1]
COLORS = ROOT / "plans" / "table_colors.yaml"
TABLE_PHOTO = ROOT / "tests" / "data" / "table_c_topdown.jpg"   # Drive IMG_2698, cropped to the table
TABLE = (232, 229, 220)


def jpeg(draw_fn, size=(640, 480)) -> bytes:
    im = Image.new("RGB", size, TABLE)
    draw_fn(ImageDraw.Draw(im))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def fingers(d):
    d.rectangle((180, 150, 230, 480), fill=(20, 20, 22))     # two long black fingers, as on the Metal gripper
    d.rectangle((410, 150, 460, 480), fill=(20, 20, 22))


def ball(colour):
    return lambda d: (fingers(d), d.ellipse((260, 290, 380, 410), fill=colour))


def test_real_table_balls_and_blocks():
    found = {(b.text(), b.role) for b in detect(TABLE_PHOTO.read_bytes(), load_config(COLORS), overhead=False)}
    assert ("white ball", "avoid") in found                    # the illegal ball
    for c in ("green", "yellow", "orange"):
        assert (f"{c} ball", "target") in found
    assert ("pink cube", "info") in found and ("blue cube", "info") in found
    assert ("pink stick lying", "info") in found and ("yellow stick lying", "info") in found
    assert not any(t.startswith("black") for t, _ in found)    # the black gripper fingers are not balls


def test_white_ball_between_the_fingers_trips_the_local_guard():
    yes, why = TableDetector.from_file(COLORS).guard({"wrist_r": jpeg(ball((206, 223, 241)))})
    assert yes is True and "white ball" in why


def test_black_ball_between_the_fingers_trips_the_local_guard():
    yes, _ = TableDetector.from_file(COLORS).guard({"wrist_r": jpeg(ball((12, 12, 12)))})
    assert yes is True


def test_coloured_ball_or_empty_gripper_is_not_a_guard_yes():
    det = TableDetector.from_file(COLORS)
    assert det.guard({"wrist_r": jpeg(ball((250, 130, 60)))})[0] is False   # orange: legal
    assert det.guard({"wrist_r": jpeg(fingers), "scene": jpeg(lambda d: None)})[0] is False


def test_unreadable_wrist_frame_is_no_answer():
    assert TableDetector.from_file(COLORS).guard({"wrist_r": b"\xff\xd8garbage\xff\xd9"})[0] is None


def test_annotation_and_legend():
    det = TableDetector.from_file(COLORS)
    frame = TABLE_PHOTO.read_bytes()
    boxed = det.annotate_frames({"scene": frame})["scene"]
    assert boxed != frame and valid_jpeg(boxed)
    assert "white ball" in det.legend() and "RED boxes" in det.legend()
