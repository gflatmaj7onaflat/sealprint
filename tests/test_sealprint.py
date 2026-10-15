"""Tests for Seal Print.  Run from the repository folder:

    QT_QPA_PLATFORM=offscreen python -m pytest -q
"""
import random
import re
import sys
import zlib
from pathlib import Path

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


def test_lengthening_strokes_keeps_their_thickness():
    p = sp.Placed("g", "十", glyph_cross(), stretch_v=0.5, stretch_v_at=0.2)
    g, bands = sp.shape_glyph(p)
    assert g.shape == (150, 100)                         # 50% taller
    assert g[:, 50].all()                                # the vertical stroke runs the whole new height
    assert g[:, 10].sum() == 12                          # the horizontal stroke is still 12 px thick
    assert bands[0] < bands[1]


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
    first, second, third, fourth = b.chars
    assert first.cx > third.cx and second.cx > fourth.cx        # first column is on the right
    assert first.cy < second.cy and third.cy < fourth.cy        # each column top to bottom
    for p in b.chars:
        assert 1.0 < p.cx - p.w / 2 and p.cx + p.w / 2 < 19.0   # inside the frame


def test_characters_and_builds_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "LIBRARY", tmp_path / "library")
    monkeypatch.setattr(sp, "CHARACTERS", tmp_path / "characters")
    glyph = sp.tidy_glyph(np.pad(glyph_cross(), 30))
    cid = sp.save_character(glyph, "十", "parent-1", "Test seal", (0.1, 0.1, 0.4, 0.4))
    [entry] = sp.character_entries()
    assert entry["name"] == "十" and entry["parent_name"] == "Test seal"
    assert (sp.png_to_ink(entry["dir"] / "glyph.png") == glyph).all()
    b = sp.Build("ellipse", 18, 24, frame_mm=0.8)
    b.chars = [sp.Placed(cid, "十", glyph, stretch_v=0.2, invert=True)]
    sp.auto_arrange(b)
    sid = sp.save_build(b, "十字", None)
    [seal] = sp.library_entries()
    assert seal["name"] == "十字" and seal["square"] is False and seal["size"] == 24
    again = sp.load_build(seal)
    assert (again.shape, again.w_mm, again.h_mm, again.frame_mm) == ("ellipse", 18, 24, 0.8)
    assert again.chars[0].invert and again.chars[0].stretch_v == 0.2
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


def test_resizing_a_characters_canvas_keeps_it_in_place():
    b = sp.Build("rect", 20, 20, 1.0)
    p = sp.Placed("g", "十", glyph_cross(), stretch_v=0.3, stretch_v_at=0.3)
    b.chars = [p]
    sp.auto_arrange(b)
    before = sp.render_build(b, 20)
    sp.resize_canvas(p, 30, 10, 25, 5)
    assert (before != sp.render_build(b, 20)).sum() <= 4
    sp.resize_canvas(p, -30, -10, -25, -5)
    assert (before == sp.render_build(b, 20)).all()
