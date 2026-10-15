"""Tests for Seal Print.  Run from the repository folder:

    QT_QPA_PLATFORM=offscreen python -m pytest -q
"""
import json
import math
import random
import re
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6 import QtCore  # noqa: E402
import sealprint as sp  # noqa: E402

IMAGES = Path(__file__).resolve().parent / "images"     # the tests' own seal images
PT = 72 / 25.4                     # PDF points per mm


@pytest.fixture(scope="session", autouse=True)
def qt():
    yield QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


def box(corners):
    """(x, y, width, height) of a frame, in whole pixels."""
    c = sp.order_corners(corners)
    return (round(c[0][0] + 0.5), round(c[0][1] + 0.5),
            round(float(np.linalg.norm(c[1] - c[0]))), round(float(np.linalg.norm(c[3] - c[0]))))


def placements(pdf: bytes):
    """Every (width_mm, height_mm) a seal is drawn at, read from the page contents."""
    sizes = []
    for m in re.finditer(rb"<< /Length (\d+) /Filter /FlateDecode >>\nstream\n", pdf):
        ops = zlib.decompress(pdf[m.end():m.end() + int(m.group(1))]).decode()
        sizes += [(float(w) / PT, float(h) / PT) for w, h in re.findall(r"q ([\d.]+) 0 0 ([\d.]+) [\d.]+ [\d.]+ cm", ops)]
    return sizes


def test_finds_a_red_seal_in_a_screenshot():
    corners, mode, found, _ = sp.detect_frame(sp.load_image(str(IMAGES / "red-square.png")))
    assert found and mode == "red"
    assert box(corners) == (300, 130, 360, 360)


def test_finds_a_black_seal_next_to_text():
    corners, mode, found, _ = sp.detect_frame(sp.load_image(str(IMAGES / "black-on-a-page.png")))
    assert found and mode == "dark"
    assert box(corners) == (60, 120, 300, 300)


def test_other_shapes_keep_their_proportions():
    job = sp.SealJob(sp.load_image(str(IMAGES / "oval.png")), "oval", size=30.0, square=False)
    job.detect()
    w, h = job.dims()
    assert h == pytest.approx(30.0)
    assert w == pytest.approx(30.0 * 300 / 420, abs=0.05)


def test_result_is_pure_black_and_white():
    job = sp.SealJob(sp.load_image(str(IMAGES / "red-square.png")), "red", size=20.0)
    job.detect()
    ink = job.render(mirror=False)
    assert ink.dtype == bool and ink.shape == (sp.export_px(20.0), sp.export_px(20.0))
    assert 0.05 < ink.mean() < 0.95


def test_packing_never_overlaps_or_leaves_the_page():
    rnd = random.Random(1)
    for _ in range(200):
        seals = [(rnd.uniform(5, 60), rnd.uniform(5, 60), rnd.randint(1, 30)) for _ in range(rnd.randint(1, 4))]
        gap = rnd.choice([0.0, 2.0, 5.0])
        pages = sp.pack(seals, gap)
        assert sum(len(p) for p in pages) == sum(n for *_, n in seals)
        for page in pages:
            for i, (_, x, y, w, h) in enumerate(page):
                assert sp.PAGE_MARGIN - 1e-6 <= x and x + w <= sp.A4_W - sp.PAGE_MARGIN + 1e-6
                assert sp.PAGE_MARGIN - 1e-6 <= y and y + h <= sp.A4_H - sp.PAGE_MARGIN + 1e-6
                for _, x2, y2, w2, h2 in page[i + 1:]:
                    assert (x + w + gap <= x2 + 1e-6 or x2 + w2 + gap <= x + 1e-6
                            or y + h + gap <= y2 + 1e-6 or y2 + h2 + gap <= y + 1e-6)


@pytest.mark.parametrize("scale", [1.0, 1.0361])
def test_pdf_draws_every_seal_at_its_exact_size(scale):
    job = sp.SealJob(sp.load_image(str(IMAGES / "red-square.png")), "red", size=25.0, copies=3)
    job.detect()
    pdf, pages = sp.make_pdf([job], gap=5.0, mirror=False, title="test", scale=scale)
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    assert b"/BitsPerComponent 1" in pdf and b"/MediaBox [0 0 595.2756 841.8898]" in pdf
    sizes = placements(pdf)
    assert len(sizes) == 3
    for w, h in sizes:
        assert w == pytest.approx(25.0 * scale, abs=1e-3) and h == pytest.approx(25.0 * scale, abs=1e-3)


def test_command_line_specs():
    oval = str(IMAGES / "oval.png")
    assert sp.parse_spec(oval, 20.0, 1) == (oval, 20.0, 1, True)
    assert sp.parse_spec(oval + ":30:2:other", 20.0, 1) == (oval, 30.0, 2, False)
    with pytest.raises(ValueError):
        sp.parse_spec("missing.png:20:2", 20.0, 1)


# ───────────────────────────── characters and builds

def glyph_cross(size=100, bar=12):
    """A plus sign: one vertical and one horizontal stroke."""
    g = np.zeros((size, size), bool)
    g[:, size // 2 - bar // 2:size // 2 + bar // 2] = True
    g[size // 2 - bar // 2:size // 2 + bar // 2, :] = True
    return g


def test_lengthening_a_selected_stroke_grows_its_free_end():
    g = np.pad(glyph_cross(200, 20), 80)                 # bar rows 170-189, x 80-279
    sel, centre = sp.select_stroke(g, np.column_stack([np.linspace(205, 276, 40), np.full(40, 180.0)]))
    end = sp.free_end(g, sel, centre)
    assert end == 1                                      # the right end is free, the left joins the cross
    out, _, _ = sp.lengthen_stroke(g, sel, centre, 0.5, end)
    assert abs(np.nonzero(out[180])[0].max() - 313) <= 2   # 50% of the ~68 px selected
    assert [int(out[:, x].sum()) for x in (240, 290, 310)] == [20, 20, 20]   # no thicker
    assert (out[:, :195] == g[:, :195]).all()            # the rest is untouched


def test_build_frame_shape_and_white_text():
    b = sp.Build("rect", 20, 20, frame_mm=1.0)
    p = sp.Placed("g", "十", glyph_cross())
    b.chars.append(p)
    sp.auto_arrange(b)
    ink = sp.render_build(b, 10)                         # 200 x 200 px
    assert ink.shape == (200, 200)
    assert ink[:10].all() and ink[:, -10:].all()         # a 1 mm frame is 10 px thick
    assert not ink[10:13, 10:13].any()                   # paper just inside it
    assert ink[100, 100]                                 # the character sits in the middle
    b.invert_all = True
    white = sp.render_build(b, 10)
    assert not white[100, 100] and white[30, 30]         # white text: character white, ground black
    assert white[5, 5] and not white[15, 15]             # solid rim, then the frame as a white line
    b.shape, b.invert_all, b.frame_mm = "ellipse", False, 0.0
    oval = sp.render_build(b, 10)
    assert not oval[0, 0] and not oval[3, 3]             # nothing drawn outside the circle


def test_auto_arrange_reads_right_to_left():
    b = sp.Build("rect", 20, 20, frame_mm=1.0)
    b.chars = [sp.Placed(str(i), str(i), glyph_cross()) for i in range(4)]
    sp.auto_arrange(b)
    first, second, third, fourth = (sp.char_rect(p) for p in b.chars)
    assert first[0] > third[0] and second[0] > fourth[0]        # first column is on the right
    assert first[1] < second[1] and third[1] < fourth[1]        # each column top to bottom
    for x, y, w, h in (first, second, third, fourth):
        assert 1.0 < x and x + w < 19.0 and 1.0 < y and y + h < 19.0   # inside the frame
        assert abs(w - h) < 0.05                                # kept their proportions


def test_characters_and_builds_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "LIBRARY", tmp_path / "library")
    monkeypatch.setattr(sp, "CHARACTERS", tmp_path / "characters")
    glyph = sp.tidy_glyph(np.pad(glyph_cross(), 30))
    cid = sp.save_character(glyph, "十", "parent-1", "Test seal", (0.1, 0.1, 0.4, 0.4))
    [entry] = sp.character_entries()
    assert entry["name"] == "十" and entry["parent_name"] == "Test seal"
    assert (sp.png_to_ink(entry["dir"] / "glyph.png") == glyph).all()
    b = sp.Build("ellipse", 18, 24, frame_mm=0.8)
    op = {"kind": "lengthen", "path": [[60.0, 59.0], [95.0, 59.0]], "amount": 0.4, "end": 1}
    b.chars = [sp.Placed(cid, "十", glyph, sq_x=0.8, use_squish=False, crop=[0, 0, 90, 130], ops=[op], invert=True)]
    sp.auto_arrange(b)
    before = sp.render_build(b, 10)
    sid = sp.save_build(b, "十字", None)
    [seal] = sp.library_entries()
    assert seal["name"] == "十字" and seal["square"] is False and seal["size"] == 24
    again = sp.load_build(seal)
    assert (again.shape, again.w_mm, again.h_mm, again.frame_mm) == ("ellipse", 18, 24, 0.8)
    c = again.chars[0]
    assert c.invert and c.sq_x == 0.8 and not c.use_squish and c.crop == [0, 0, 90, 130] and c.ops == [op]
    assert (sp.render_build(again, 10) == before).all()
    job = sp.job_from_library(seal)
    assert job.lib_id == sid and job.dims() == pytest.approx((18, 24), abs=0.05)


# ───────────────────────────── single strokes

def plus_padded():
    """A plus sign with room around it: bar rows 170-189, x 80-279; vertical stroke x 170-189."""
    return np.pad(glyph_cross(200, 20), 80)


@pytest.mark.parametrize("off", [0, 7, -8])
def test_drawing_along_part_of_a_stroke_selects_just_that_part(off):
    g = plus_padded()
    path = np.column_stack([np.linspace(205, 276, 40), np.full(40, 180.0 + off)])   # even along an edge
    sel, centre = sp.select_stroke(g, path)
    assert sel[:, 250].sum() == 20                       # the whole width of the bar there
    assert not sel[180, 120] and not sel[100, 180]       # not the other half, not the other stroke
    assert abs(np.median(centre[:, 1]) - 179.5) < 1.5    # centreline in the middle of the bar


def test_pulling_a_stroke_lengthens_it_without_thickening():
    g = plus_padded()
    sel, centre = sp.select_stroke(g, np.column_stack([np.linspace(205, 276, 40), np.full(40, 180.0)]))
    drag = sp.StrokeDrag(g, sel, centre, (275, 180))
    ink, _, _ = drag.apply((40, 0))
    assert np.nonzero(ink[180])[0].max() == 319          # was 279
    assert ink[:, 300].sum() == 20                       # same thickness as before
    assert (ink[:, :170] == g[:, :170]).all()            # the rest of the character is untouched
    bent, _, _ = drag.apply((0, 50))
    rows = np.nonzero(bent[:, 272])[0]
    assert rows.min() >= 215 and rows.max() <= 245       # the end moved down about 50
    assert (bent[:, 160:200] == g[:, 160:200]).all()     # the junction and the other stroke stay put


def test_every_edit_can_be_switched_off_and_on_again():
    g = glyph_cross(200, 20)                             # bar rows 90-109, x 0-199
    p = sp.Placed("g", "十", g, cx=10, cy=10, scale=0.05)
    plain = sp.char_rect(p)
    p.ops = [{"kind": "lengthen", "path": [[125.0, 99.5], [196.0, 99.5]], "amount": 0.5, "end": 1}]
    p.sq_x, p.weight, p.crop = 0.7, 2, [20, 0, 200, 200]
    edited = sp.char_rect(p)
    assert edited != plain
    for flag in ("use_strokes", "use_lengthen", "use_squish", "use_weight", "use_crop"):
        setattr(p, flag, False)
    assert sp.char_rect(p) == pytest.approx(plain)       # every Original switch on: as picked
    for flag in ("use_strokes", "use_lengthen", "use_squish", "use_weight", "use_crop"):
        setattr(p, flag, True)
    assert sp.char_rect(p) == pytest.approx(edited)      # and back


def test_builds_from_earlier_versions_still_open(tmp_path):
    d = tmp_path / "seal"
    (d / "glyphs").mkdir(parents=True)
    (d / "glyphs" / "g1.png").write_bytes(sp.ink_to_png(glyph_cross()))
    old = {"version": 1, "shape": "rect", "w_mm": 20, "h_mm": 20, "frame_mm": 1, "invert_all": False,
           "chars": [{"glyph_id": "g1", "name": "十", "cx": 10, "cy": 10, "w": 8, "h": 12, "weight": 1,
                      "stretch_v": 0.5, "stretch_v_at": 0.2, "stretch_h": 0, "stretch_h_at": 0.5, "invert": False}]}
    (d / "build.json").write_text(json.dumps(old), "utf-8")
    b = sp.load_build({"dir": d})
    p = b.chars[0]
    assert p.src.shape == (150, 100) and p.weight == 1   # the old lengthening is baked into the picture
    x, y, w, h = sp.char_rect(p)
    assert abs(x + w / 2 - 10) < 0.3 and abs(y + h / 2 - 10) < 0.3 and h > w


# ───────────────────────────── outlines, eraser, overlap, round corners

def test_a_hand_drawn_outline_picks_only_what_is_inside():
    ink = np.zeros((100, 100), bool)
    ink[20:80, 20:30] = True                             # the character's stroke
    ink[20:80, 60:70] = True                             # a neighbour inside the bounding box
    poly = np.array([[10, 10], [45, 10], [45, 90], [10, 90]])
    mask = np.zeros((100, 100), np.uint8)
    cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 1)
    g = sp.tidy_glyph(ink & (mask > 0))
    assert g.sum() == 60 * 10                            # just the stroke inside the outline
    edge = np.zeros((100, 100), np.uint8)
    cv2.polylines(edge, [poly.reshape(-1, 1, 2)], True, 1, 3)
    ink2 = ink.copy()
    ink2[50:55, 5:15] = True                             # a bit of something the outline cuts through
    kept = sp.tidy_glyph(ink2 & (mask > 0), True, edge > 0)
    assert kept.sum() == 60 * 10                         # dropped when asked


def test_erasing_can_be_switched_off():
    g = glyph_cross(200, 20)
    p = sp.Placed("g", "十", g, cx=10, cy=10, scale=0.05)
    whole = sp.process_char(p)[0].sum()
    p.ops = [{"kind": "erase", "points": [[150.0, 99.5], [199.0, 99.5]], "radius": 15}]
    p.cache = None
    erased = sp.process_char(p)[0].sum()
    assert erased < whole - 900                          # the end of the bar is gone
    p.use_erase = False
    assert sp.process_char(p)[0].sum() == whole


def test_overlapping_characters_show_through_each_other():
    ring = np.zeros((100, 100), bool)                    # a square outline: paper inside it
    ring[:10], ring[-10:], ring[:, :10], ring[:, -10:] = True, True, True, True
    bar = np.zeros((100, 100), bool)
    bar[45:55, :] = True
    build = sp.Build("rect", 20, 20, frame_mm=0)
    a = sp.Placed("a", "a", ring, cx=10, cy=10, scale=0.1)    # px 50-150 at 10 px/mm
    b = sp.Placed("b", "b", bar, cx=14, cy=10, scale=0.1)     # px 90-190, rows 95-105
    build.chars = [a, b]
    ink = sp.render_build(build, 10)
    assert ink[100, 145] and ink[100, 120] and ink[100, 175] and not ink[80, 100]   # both show
    a.invert, a.z = True, 1                              # inverted: a solid tile with a white outline,
    ink = sp.render_build(build, 10)                     # on top it covers the bar
    assert ink[80, 100] and not ink[100, 145] and ink[100, 175]
    a.z = -1                                             # underneath, the bar shows across it
    assert sp.render_build(build, 10)[100, 145]


def test_round_corners():
    b = sp.Build("rect", 20, 20, frame_mm=1.0, corner_mm=4.0)
    ink = sp.render_build(b, 10)                         # 200 x 200 px, 40 px radius, 10 px frame
    assert not ink[2, 2] and not ink[8, 8]               # the corner itself is cut away
    assert ink[0, 100].all() and ink[:10, 100].all()     # straight edges: a 10 px frame
    assert ink[100, :10].all() and not ink[100, 12]
    d = 40 - 40 / math.sqrt(2)                           # on the diagonal, the frame follows the curve
    assert ink[int(d) + 3, int(d) + 3] and not ink[int(d) + 14, int(d) + 14]
