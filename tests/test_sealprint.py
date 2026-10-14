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
