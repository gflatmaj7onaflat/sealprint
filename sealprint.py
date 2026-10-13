#!/usr/bin/env python3
"""
Seal Print: print square Chinese seal designs at exact sizes on A4.

Add one or more seal images: files, screenshots pasted with Ctrl+V, or dropped files.
For each image the app finds the seal's square frame and turns the seal into pure
black and white. Every seal has its own size and number of copies, and all of them
are packed onto A4 pages in a single PDF.

    python3 sealprint.py [image ...]                          # open the window
    python3 sealprint.py a.png:20:6 b.png:25:2 -o seals.pdf   # no window; image:size_mm:copies
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF, Qt

APP_NAME = "Seal Print"
A4_W, A4_H = 210.0, 297.0           # page size, mm
PAGE_MARGIN = 10.0                  # mm kept clear at the paper edge
EXPORT_DPI = 1200                   # resolution of the seal image inside the PDF
MAX_EXPORT_PX = 6000                # cap for very large seals
PREVIEW_PX = 640                    # working size for the on-screen preview
MIN_INK = {"red": 45.0, "dark": 60.0}

# Themes are two tones, a background and an accent; every other interface colour is
# worked out from them in derive_theme(). Cinnabar stays fixed for the seal frame itself.
THEMES = {
    "Dark": ("#26302b", "#a9c6a1"),            # seal stone and jade
    "Ink": ("#19191b", "#e3dccd"),             # ink stick and bone
    "Lacquer": ("#2a1917", "#d6a85c"),         # red-brown lacquer and gold
    "Paper": ("#ecebe5", "#2b2b2f"),           # rice paper and ink
    "Celadon": ("#dbe5de", "#2f6a58"),         # celadon glaze and deep green
    "Blue and white": ("#edf1f7", "#284a96"),  # porcelain and cobalt
}
DEFAULT_THEME = "Dark"
CINNABAR = "#d8452f"
C: dict = {}                                   # colours of the theme in use


# ───────────────────────────────────────────────────────────────── image loading

def _flatten(arr: np.ndarray) -> np.ndarray:
    """Gray / RGB / RGBA uint8 -> RGB uint8, with transparency laid onto white paper."""
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        a = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a)
        return (rgb + 0.5).astype(np.uint8)
    return np.ascontiguousarray(arr[:, :, :3])


def qimage_to_rgb(img: QtGui.QImage) -> np.ndarray:
    img = img.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), dtype=np.uint8, count=h * bpl).reshape(h, bpl)
    return _flatten(buf[:, : w * 4].reshape(h, w, 4))


def rgb_to_qimage(rgb: np.ndarray) -> QtGui.QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()


def ink_to_qimage(ink: np.ndarray) -> QtGui.QImage:
    g = np.ascontiguousarray(np.where(ink, 0, 255).astype(np.uint8))
    h, w = g.shape
    return QtGui.QImage(g.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()


def load_image(path: str) -> np.ndarray:
    reader = QtGui.QImageReader(path)
    reader.setAutoTransform(True)                 # honour EXIF rotation from phone photos
    img = reader.read()
    if not img.isNull():
        return qimage_to_rgb(img)
    arr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(reader.errorString() or "unsupported image format")
    if arr.dtype == np.uint16:
        arr = (arr // 257).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA if arr.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return _flatten(arr)


# ───────────────────────────────────────────────────────────────── ink and frame

def ink_map(rgb: np.ndarray, mode: str) -> np.ndarray:
    """0 = paper, 255 = solid ink.

    'red'  measures red minus green. Vermilion seal paste scores high while white or
           yellowed paper, black writing and dark screen backgrounds all score ~0.
    'dark' measures how dark the darkest channel is: black, blue or red ink all count.
    """
    if mode == "red":
        diff = rgb[:, :, 0].astype(np.int16) - rgb[:, :, 1].astype(np.int16)
        return np.clip(diff, 0, 255).astype(np.uint8)
    return 255 - rgb.min(axis=2)


def otsu(gray: np.ndarray) -> float:
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(t)


def order_corners(pts) -> np.ndarray:
    """Return 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


def whole_image_corners(w: int, h: int) -> np.ndarray:
    # Pixel centres sit on integer coordinates, so the image's outer edge is at -0.5.
    return np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])


@dataclass
class _Cluster:
    score: float
    squareness: float
    frac: float          # hull area / image area
    touches: bool        # reaches the image edge
    thresh: float
    xs: np.ndarray       # coordinates of the cluster's core ink pixels
    ys: np.ndarray


def _best_cluster(m: np.ndarray, min_t: float, square: bool = True) -> _Cluster | None:
    """Find the group of ink that looks most like a seal: large, square, built from big pieces.

    Strokes are grouped at several merge distances, so a seal whose frame is chipped
    or missing still comes out as one group. Within a group only the significant pieces
    count (frame, whole characters), which keeps captions and nearby text out.
    """
    H, W = m.shape
    t = max(otsu(m), min_t)
    mask = (m > t).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    keep = st[:, cv2.CC_STAT_AREA] >= max(4, H * W * 3e-5)      # drop specks
    keep[0] = False
    mask = keep[lab].astype(np.uint8)
    if not mask.any():
        return None
    piece = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    best = None
    for reach in (0.012, 0.03, 0.06):
        r = max(2, int(round(reach * min(H, W))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, kernel), connectivity=8)
        for i in np.argsort(-st2[1:, cv2.CC_STAT_AREA])[:25] + 1:
            if st2[i, cv2.CC_STAT_AREA] < H * W * 4e-4:
                break
            x, y, w, h = (int(v) for v in st2[i, :4])
            sub_lab = lab[y:y + h, x:x + w]
            sel = (lab2[y:y + h, x:x + w] == i) & (mask[y:y + h, x:x + w] > 0)
            ids = np.unique(sub_lab[sel])
            ys, xs = np.nonzero(sel)
            if xs.size < 16:
                continue
            size = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
            big = np.zeros(n, bool)
            big[ids[piece[ids] >= 0.08 * size]] = True
            ys, xs = np.nonzero(sel & big[sub_lab])
            if xs.size < 16:
                continue
            xs, ys = xs + x, ys + y
            hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.float32))
            area = float(cv2.contourArea(hull))
            _, (rw, rh), _ = cv2.minAreaRect(hull)
            if min(rw, rh) < 4:
                continue
            sq = min(rw, rh) / max(rw, rh)
            touches = bool(xs.min() <= 0 or ys.min() <= 0 or xs.max() >= W - 1 or ys.max() >= H - 1)
            # Square seals: strongly prefer square groups. Other shapes: only mildly.
            score = area * sq ** (2.0 if square else 0.5) * (0.15 if touches else 1.0)
            if best is None or score > best.score * 1.001:
                best = _Cluster(score, sq, area / (H * W), touches, t, xs, ys)
    return best


def detect_frame(rgb: np.ndarray, mode: str = "auto", square: bool = True):
    """Locate the seal's frame: its outer edge for a square seal, its bounding box for a
    rectangle, oval, circle or any other shape.

    Returns (corners, resolved_mode, found, ratio). corners is 4x2 (TL, TR, BR, BL) in
    source pixel coordinates, sitting on the outer edge of the ink. ratio is short side /
    long side of the box as found, before a square seal's box is made square.
    """
    H, W = rgb.shape[:2]
    s = min(1.0, 1000.0 / max(H, W))
    small = rgb if s >= 1.0 else cv2.resize(
        rgb, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)

    modes = ["red", "dark"] if mode == "auto" else [mode]
    found = {}
    for md in modes:
        c = _best_cluster(ink_map(small, md), MIN_INK[md], square)
        if c is not None:
            found[md] = c
    if not found:
        return whole_image_corners(W, H), ("dark" if mode == "auto" else mode), False, min(W, H) / max(W, H)

    if len(found) == 1:
        pick = next(iter(found))
    else:
        red, dark = found["red"], found["dark"]
        need = 0.35 if square else 0.2          # red ink is a strong sign of a seal, whatever its shape
        red_ok = red.squareness >= need and red.frac >= 0.004
        dark_dominates = dark.squareness >= need + 0.05 and not dark.touches and dark.frac >= 4 * red.frac
        if red_ok and not dark_dominates:
            pick = "red"
        else:
            pick = "red" if red.score * 1.15 >= dark.score else "dark"
    c = found[pick]

    # Refine on the full-resolution image, near the ink of the chosen cluster.
    if s < 1.0:
        core = np.zeros(small.shape[:2], np.uint8)
        core[c.ys, c.xs] = 1
        near = cv2.resize(core, (W, H), interpolation=cv2.INTER_NEAREST)
        pad = int(math.ceil(1.0 / s)) + 1
        near = cv2.dilate(near, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        ys, xs = np.nonzero((ink_map(rgb, pick) > c.thresh) & (near > 0))
        if xs.size < 16:
            xs, ys = c.xs / s, c.ys / s
    else:
        xs, ys = c.xs, c.ys
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(cv2.convexHull(pts))
    rw, rh = rw + 1.0, rh + 1.0                         # centres -> outer pixel edges

    # Screenshots are upright: keep the box axis-aligned unless the seal is clearly tilted.
    x0, y0, x1, y1 = xs.min() - 0.5, ys.min() - 0.5, xs.max() + 0.5, ys.max() + 0.5
    if (x1 - x0) * (y1 - y0) <= rw * rh * 1.04:
        cx, cy, rw, rh, ang = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0

    ratio = min(rw, rh) / max(rw, rh)
    # Square seal, nearly square box: the warp stretches it to exactly square. Clearly not
    # square (e.g. a borderless seal): pad the short side so the design isn't distorted.
    if square and ratio < 1 / 1.04:
        rw = rh = max(rw, rh)
    corners = cv2.boxPoints(((float(cx), float(cy)), (float(rw), float(rh)), float(ang)))
    return order_corners(corners), pick, True, ratio


# ───────────────────────────────────────────────────────────────── rendering

def warp_ink(m: np.ndarray, corners, out_w: int, out_h: int | None = None, max_work: int = 8000):
    """Map the framed quad of an ink map onto an out_w x out_h rectangle (square if out_h
    is omitted). Returns (warped uint8, output pixels per source pixel)."""
    out_h = out_w if out_h is None else out_h
    c = order_corners(corners).astype(np.float32)
    src_w = max(float(np.linalg.norm(c[1] - c[0])), float(np.linalg.norm(c[2] - c[3])), 1.0)
    src_h = max(float(np.linalg.norm(c[3] - c[0])), float(np.linalg.norm(c[2] - c[1])), 1.0)
    # Work at no less than the source resolution, then shrink with area averaging.
    k = min(max(1.0, src_w / out_w, src_h / out_h), max(1.0, max_work / max(out_w, out_h)))
    ww, wh = max(out_w, round(out_w * k)), max(out_h, round(out_h * k))
    dst = np.float32([[-0.5, -0.5], [ww - 0.5, -0.5], [ww - 0.5, wh - 0.5], [-0.5, wh - 0.5]])
    M = cv2.getPerspectiveTransform(c, dst)
    out = cv2.warpPerspective(m, M, (ww, wh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if (ww, wh) != (out_w, out_h):
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return out, 0.5 * (out_w / src_w + out_h / src_h)


def to_black_white(warped: np.ndarray, thresh: float, smooth_src_px: float, px_per_src: float,
                   invert: bool = False, mirror: bool = False) -> np.ndarray:
    """Threshold to pure black/white. Returns a bool array, True = black."""
    g = warped
    sigma = smooth_src_px * px_per_src
    if sigma >= 0.25:
        g = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)
    ink = g > thresh
    if invert:
        ink = ~ink
    if mirror:
        ink = ink[:, ::-1]
    return np.ascontiguousarray(ink)


def export_px(size_mm: float) -> int:
    return int(min(MAX_EXPORT_PX, round(size_mm / 25.4 * EXPORT_DPI)))




# ───────────────────────────────────────────────────────────────── one seal

@dataclass(eq=False)
class SealJob:
    """One seal design: its source image, frame and settings."""
    rgb: np.ndarray
    name: str
    size: float = 20.0                 # side of a square seal, or longest length of any other, mm
    copies: int = 1
    smooth: int = 2                    # 0-10; blur of 0.2 source px per step before thresholding
    ink_choice: str = "auto"           # what the user picked: auto, red or dark
    invert: bool = False
    mode: str = "dark"                 # ink model actually used
    found: bool = True
    frame_manual: bool = False
    auto_t: bool = True                # stroke weight follows the automatic threshold
    weight: int = 128                  # 255 - threshold
    corners: np.ndarray | None = None
    square: bool = True                # False keeps the frame's proportions: rectangle, oval, circle…
    ratio: float = 1.0                 # short / long side of the frame as detected
    lib_id: str | None = None          # folder name in the seal library, once saved
    maps: dict = field(default_factory=dict)
    cache: tuple | None = None
    preview: QtGui.QImage | None = None
    preview_key: tuple | None = None

    def ink(self) -> np.ndarray:
        if self.mode not in self.maps:
            self.maps[self.mode] = ink_map(self.rgb, self.mode)
        return self.maps[self.mode]

    def detect(self):
        self.corners, self.mode, self.found, self.ratio = detect_frame(self.rgb, self.ink_choice, self.square)
        self.frame_manual = False
        self.auto_t = True

    def frame_px(self) -> tuple[float, float]:
        """Width and height of the frame in source pixels."""
        c = order_corners(self.corners)
        w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
        h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
        return float(w), float(h)

    def dims(self) -> tuple[float, float]:
        """Printed width and height in mm."""
        if self.square:
            return self.size, self.size
        w, h = self.frame_px()
        k = self.size / max(w, h, 1e-6)
        return w * k, h * k

    def out_px(self, long_px: int) -> tuple[int, int]:
        w, h = self.dims()
        k = long_px / max(w, h)
        return max(1, round(w * k)), max(1, round(h * k))

    def frame_ok(self) -> bool:
        return cv2.contourArea(order_corners(self.corners).astype(np.float32)) >= 16

    def _warped_preview(self):
        key = (self.mode, self.square, np.round(self.corners, 3).tobytes())
        if self.cache is None or self.cache[0] != key:
            warped, pps = warp_ink(self.ink(), self.corners, *self.out_px(PREVIEW_PX), max_work=2 * PREVIEW_PX)
            self.cache = (key, warped, pps, otsu(warped))
        return self.cache

    def update_preview(self, mirror: bool) -> bool:
        """Re-render the on-screen version if anything changed; True if it did."""
        if not self.frame_ok():
            return False
        key, warped, pps, t_auto = self._warped_preview()
        if self.auto_t:
            self.weight = int(round(255 - t_auto))
        pkey = (key, self.weight, self.smooth, self.invert, mirror)
        if pkey == self.preview_key:
            return False
        ink = to_black_white(warped, 255 - self.weight, self.smooth * 0.2, pps, self.invert, mirror)
        self.preview, self.preview_key = ink_to_qimage(ink), pkey
        return True

    def render(self, mirror: bool, scale: float = 1.0) -> np.ndarray:
        """Full-resolution black/white seal for the PDF (True = black)."""
        if not self.frame_ok():
            raise ValueError(f"The frame of {self.name} has collapsed; drag its corners apart")
        if self.auto_t:
            self.weight = int(round(255 - self._warped_preview()[3]))
        warped, pps = warp_ink(self.ink(), self.corners, *self.out_px(export_px(max(self.dims()) * scale)))
        return to_black_white(warped, 255 - self.weight, self.smooth * 0.2, pps, self.invert, mirror)

    def duplicate(self) -> SealJob:
        return replace(self, corners=self.corners.copy(), lib_id=None)   # shares image and ink maps


# ───────────────────────────────────────────────────────────────── seal library

LIBRARY = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "sealprint" / "library"
SAVED_FIELDS = ("size", "copies", "smooth", "ink_choice", "invert", "mode", "found", "frame_manual",
                "auto_t", "weight", "square", "ratio")


def library_entries() -> list[dict]:
    """Saved seals, newest first. Unreadable entries are skipped."""
    entries = []
    if LIBRARY.is_dir():
        for d in LIBRARY.iterdir():
            try:
                meta = json.loads((d / "seal.json").read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if (d / "image.png").is_file():
                meta["dir"] = d
                entries.append(meta)
    entries.sort(key=lambda e: e.get("saved", ""), reverse=True)
    return entries


def save_to_library(job: SealJob, name: str) -> str:
    """Store the seal (its image around the frame, the frame and every setting) in the
    library. A seal that came from the library is updated in place. Returns its id."""
    if not (job.lib_id and (LIBRARY / job.lib_id).is_dir()):
        job.lib_id = time.strftime("%Y%m%d-%H%M%S-") + os.urandom(3).hex()
    d = LIBRARY / job.lib_id
    d.mkdir(parents=True, exist_ok=True)
    # Keep the frame plus generous room around it, so it can still be adjusted later.
    c = np.asarray(job.corners, dtype=float)
    (x0, y0), (x1, y1) = c.min(axis=0), c.max(axis=0)
    H, W = job.rgb.shape[:2]
    mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    X0, Y0 = max(0, int(math.floor(x0 - mx))), max(0, int(math.floor(y0 - my)))
    X1, Y1 = min(W, int(math.ceil(x1 + mx)) + 1), min(H, int(math.ceil(y1 + my)) + 1)
    if X1 - X0 < 2 or Y1 - Y0 < 2:
        X0, Y0, X1, Y1 = 0, 0, W, H
    ok, png = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(job.rgb[Y0:Y1, X0:X1]), cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError("couldn't encode the image")
    (d / "image.png").write_bytes(png.tobytes())
    meta = {"version": 1, "name": name, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "corners": (c - [X0, Y0]).tolist(), **{k: getattr(job, k) for k in SAVED_FIELDS}}
    (d / "seal.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    if job.preview is not None:
        job.preview.scaled(192, 192, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation).save(str(d / "thumb.png"))
    return job.lib_id


def job_from_library(entry: dict) -> SealJob:
    d = Path(entry["dir"])
    job = SealJob(load_image(str(d / "image.png")), entry.get("name") or d.name,
                  corners=np.array(entry["corners"], dtype=float), lib_id=d.name)
    for k in SAVED_FIELDS:
        if k in entry:
            setattr(job, k, entry[k])
    return job


def seal_icon(img: QtGui.QImage, px: int) -> QtGui.QIcon:
    pm = QtGui.QPixmap.fromImage(img.scaled(px, px, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))
    icon = QtGui.QIcon()
    icon.addPixmap(pm, QtGui.QIcon.Mode.Normal)
    icon.addPixmap(pm, QtGui.QIcon.Mode.Selected)          # no selection tint on the seal
    return icon


# ───────────────────────────────────────────────────────────────── layout and PDF

def pack(seals, gap: float):
    """Lay out every copy of every seal on A4 pages.

    seals: (width_mm, height_mm, copies) per seal. Taller seals are placed first in rows
    ("shelves"); shorter ones fill leftover room in earlier rows and pages before a new
    row or page is started. Seals are never rotated. Returns pages, each a list of
    (seal_index, x_mm, y_mm, width_mm, height_mm) measured from the page's top-left corner.
    """
    uw, uh, eps = A4_W - 2 * PAGE_MARGIN, A4_H - 2 * PAGE_MARGIN, 1e-9
    items = sorted(((w, h, i) for i, (w, h, n) in enumerate(seals) for _ in range(n)),
                   key=lambda t: (-t[1], -t[0]))
    pages, used = [], []                      # shelves per page, used height per page
    for w, h, i in items:
        if w > uw + eps or h > uh + eps:
            raise ValueError(f"A {w:.1f} × {h:.1f} mm seal doesn't fit on A4 with {PAGE_MARGIN:g} mm margins")
        placed = False
        for shelves in pages:                 # room at the end of an existing row?
            for sh in shelves:
                x = sh["w"] + gap
                if h <= sh["h"] + eps and x + w <= uw + eps:
                    sh["items"].append((i, x, w, h))
                    sh["w"] = x + w
                    placed = True
                    break
            if placed:
                break
        if not placed:                        # room for a new row on an existing page?
            for p, shelves in enumerate(pages):
                y = used[p] + gap
                if y + h <= uh + eps:
                    shelves.append({"y": y, "h": h, "w": w, "items": [(i, 0.0, w, h)]})
                    used[p] = y + h
                    placed = True
                    break
        if not placed:
            pages.append([{"y": 0.0, "h": h, "w": w, "items": [(i, 0.0, w, h)]}])
            used.append(h)
    out = []
    for shelves in pages:                     # centre each page's block horizontally
        dx = PAGE_MARGIN + (uw - max(sh["w"] for sh in shelves)) / 2
        out.append([(i, dx + x, PAGE_MARGIN + sh["y"], w, h) for sh in shelves for i, x, w, h in sh["items"]])
    return out


def build_pdf(designs, pages, title: str = "Seals") -> bytes:
    """A4 PDF. designs: bool arrays (True = black), each embedded once as a lossless 1-bit
    image. pages: lists of (design_index, x_mm, y_mm, width_mm, height_mm) from pack()."""
    k = 72.0 / 25.4
    objs: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: ("<< /Title <FEFF" + title.encode("utf-16-be").hex().upper() + "> /Producer (Seal Print) >>").encode(),
    }
    nid, names = 4, []
    for j, ink in enumerate(designs):
        h, w = ink.shape
        data = zlib.compress(np.packbits(~ink, axis=1).tobytes(), 9)     # 1 = white in DeviceGray
        objs[nid] = (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
                     f"/BitsPerComponent 1 /Interpolate false /Filter /FlateDecode /Length {len(data)} >>\n"
                     f"stream\n").encode() + data + b"\nendstream"
        names.append(f"/S{j} {nid} 0 R")
        nid += 1
    xobjects = " ".join(names)
    kids = []
    for spots in pages:
        page_id, content_id = nid, nid + 1
        nid += 2
        ops = "".join(f"q {w * k:.4f} 0 0 {h * k:.4f} {x * k:.4f} {(A4_H - y - h) * k:.4f} cm /S{j} Do Q\n"
                      for j, x, y, w, h in spots)
        body = zlib.compress(ops.encode())
        objs[content_id] = f"<< /Length {len(body)} /Filter /FlateDecode >>\nstream\n".encode() + body + b"\nendstream"
        objs[page_id] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {A4_W * k:.4f} {A4_H * k:.4f}] "
                         f"/Resources << /XObject << {xobjects} >> /ProcSet [/PDF /ImageB] >> "
                         f"/Contents {content_id} 0 R >>").encode()
        kids.append(page_id)
    objs[2] = f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in kids)}] /Count {len(kids)} >>".encode()

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for oid in sorted(objs):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + objs[oid] + b"\nendobj\n"
    xref = len(out)
    n = max(objs) + 1
    out += f"xref\n0 {n}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offsets[i]:010d} 00000 n \n".encode() for i in range(1, n))
    out += f"trailer\n<< /Size {n} /Root 1 0 R /Info 3 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def layout_pages(jobs, gap: float, scale: float = 1.0):
    """Pages as they sit in the PDF. scale > 1 enlarges every seal and gap so that a printer
    which always shrinks pages to fit (and can't print at actual size) brings them back."""
    return pack([(w * scale, h * scale, j.copies) for j in jobs for w, h in [j.dims()]], gap * scale)


def make_pdf(jobs, gap: float, mirror: bool, title: str, scale: float = 1.0):
    pages = layout_pages(jobs, gap, scale)
    return build_pdf([j.render(mirror, scale) for j in jobs], pages, title), pages


# ───────────────────────────────────────────────────────────────── widgets

def make_logo(size: int, dpr: float = 2.0) -> QtGui.QPixmap:
    px = int(size * dpr)
    pm = QtGui.QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QColor(CINNABAR))
    p.drawRoundedRect(QRectF(0, 0, px, px), px * 0.1, px * 0.1)
    pen = QtGui.QPen(QtGui.QColor("#fbefe8"), px * 0.075)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    m = px * 0.19
    p.drawRect(QRectF(m, m, px - 2 * m, px - 2 * m))
    p.drawLine(QPointF(px / 2, m), QPointF(px / 2, px - m))
    p.drawLine(QPointF(m, px * 0.42), QPointF(px / 2, px * 0.42))
    p.drawLine(QPointF(px / 2, px * 0.6), QPointF(px - m, px * 0.6))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


class Handle(QtWidgets.QGraphicsEllipseItem):
    R = 7.0

    def __init__(self, on_move):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self._on_move = on_move
        F = QtWidgets.QGraphicsItem.GraphicsItemFlag
        self.setFlags(F.ItemIsMovable | F.ItemSendsGeometryChanges | F.ItemIgnoresTransformations)
        self.setBrush(QtGui.QColor("#ffffff"))
        self.setPen(QtGui.QPen(QtGui.QColor(CINNABAR), 2.5))
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    def hoverEnterEvent(self, e):
        self.setScale(1.3)
        super().hoverEnterEvent(e)

    def hoverLeaveEvent(self, e):
        self.setScale(1.0)
        super().hoverLeaveEvent(e)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._on_move()
        return super().itemChange(change, value)


class FrameEditor(QtWidgets.QGraphicsView):
    """Shows the source image with the detected frame; the four corners can be dragged."""
    edited = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QtGui.QColor(C["canvas"]))
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        self.setMinimumWidth(220)
        self.pix = self.shade = self.poly = None
        self.handles: list[Handle] = []
        self.image_rect = QRectF()
        self.auto_fit = True
        self._silent = False

    def retheme(self):
        self.setBackgroundBrush(QtGui.QColor(C["canvas"]))

    def clear(self):
        self.scene().clear()
        self.pix = self.shade = self.poly = None
        self.handles = []
        self.viewport().update()

    def set_image(self, qimg: QtGui.QImage, corners):
        self.clear()
        sc = self.scene()
        self.pix = sc.addPixmap(QtGui.QPixmap.fromImage(qimg))
        self.image_rect = QRectF(0, 0, qimg.width(), qimg.height())
        pad = 0.06 * max(qimg.width(), qimg.height())
        sc.setSceneRect(self.image_rect.adjusted(-pad, -pad, pad, pad))
        self.shade = sc.addPath(QtGui.QPainterPath(), QtGui.QPen(Qt.PenStyle.NoPen),
                                QtGui.QBrush(QtGui.QColor(10, 14, 12, 150)))
        pen = QtGui.QPen(QtGui.QColor(CINNABAR), 1.6)
        pen.setCosmetic(True)
        self.poly = sc.addPolygon(QtGui.QPolygonF(), pen)
        for _ in range(4):
            h = Handle(self._moved)
            sc.addItem(h)
            self.handles.append(h)
        self.set_corners(corners)
        self.fit()

    def set_corners(self, corners):
        self._silent = True
        for h, (x, y) in zip(self.handles, np.asarray(corners, dtype=float)):
            h.setPos(x + 0.5, y + 0.5)              # scene pixels cover [x, x+1)
        self._silent = False
        self._redraw()

    def corners(self) -> np.ndarray:
        return np.array([[h.pos().x() - 0.5, h.pos().y() - 0.5] for h in self.handles])

    def _moved(self):
        self._redraw()
        if not self._silent:
            self.edited.emit()

    def _redraw(self):
        if self.poly is None or len(self.handles) < 4:
            return
        quad = QtGui.QPolygonF([h.pos() for h in self.handles])
        self.poly.setPolygon(quad)
        path = QtGui.QPainterPath()
        path.setFillRule(Qt.FillRule.OddEvenFill)
        path.addRect(self.image_rect)
        path.addPolygon(quad)
        path.closeSubpath()
        self.shade.setPath(path)

    def fit(self):
        if self.pix is None:
            return
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.auto_fit = True
        self._pixel_mode()

    def _pixel_mode(self):
        if self.pix is not None:
            sharp = self.transform().m11() >= 2.0     # show real pixels when zoomed in
            self.pix.setTransformationMode(Qt.TransformationMode.FastTransformation if sharp
                                           else Qt.TransformationMode.SmoothTransformation)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.auto_fit:
            self.fit()

    def wheelEvent(self, e):
        if self.pix is None:
            return
        f = 1.2 ** (e.angleDelta().y() / 120.0)
        vr, sr = self.viewport().rect(), self.sceneRect()
        fit_scale = min(vr.width() / sr.width(), vr.height() / sr.height())
        if self.transform().m11() * f <= fit_scale:
            self.fit()
            return
        self.scale(f, f)
        self.auto_fit = False
        self._pixel_mode()

    def mouseDoubleClickEvent(self, e):
        if not isinstance(self.itemAt(e.position().toPoint()), Handle):
            self.fit()
        else:
            super().mouseDoubleClickEvent(e)

    def drawForeground(self, painter, rect):
        if self.pix is None:
            painter.save()
            painter.resetTransform()
            painter.setPen(QtGui.QColor(C["muted"]))
            painter.drawText(QRectF(self.viewport().rect()).adjusted(24, 0, -24, 0),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             "Add an image, paste a screenshot with Ctrl+V, or drop files here")
            painter.restore()


class SealPane(QtWidgets.QWidget):
    """The selected seal in black and white, as it will be printed. Click it to inspect."""
    clicked = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.img: QtGui.QImage | None = None
        self.caption = ""
        self.setMinimumWidth(180)

    def set_image(self, img, caption=""):
        self.img, self.caption = img, caption
        self.setCursor(Qt.CursorShape.PointingHandCursor if img is not None else Qt.CursorShape.ArrowCursor)
        self.setToolTip("Click to look at it up close" if img is not None else "")
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.img is not None:
            self.clicked.emit()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(C["canvas"]))
        p.setPen(QtGui.QColor(C["muted"]))
        if self.img is None:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "The black and white seal\nshows up here")
            return
        lines = self.caption.count("\n") + 1
        area = QRectF(self.rect()).adjusted(22, 22, -22, -(lines * self.fontMetrics().height() + 26))
        k = max(1e-3, min(area.width() / self.img.width(), area.height() / self.img.height()))
        bw, bh = self.img.width() * k, self.img.height() * k
        box = QRectF(area.center().x() - bw / 2, area.center().y() - bh / 2, bw, bh)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(box, self.img)
        p.setPen(QtGui.QColor(C["rule"]))              # outline, so white paper shows on light themes
        p.drawRect(box.adjusted(-0.5, -0.5, 0.5, 0.5))
        p.setPen(QtGui.QColor(C["muted"]))
        p.drawText(QRectF(8, box.bottom() + 10, self.width() - 16, self.height() - box.bottom() - 10),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                   self.caption)


class InspectView(QtWidgets.QGraphicsView):
    """The seal at full PDF resolution. Scroll to zoom, drag to move, click to close."""
    closed = QtCore.Signal()
    zoomed = QtCore.Signal(float)

    def __init__(self, parent):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.pix = self.outline = None
        self._press = None
        self.retheme()

    def retheme(self):
        self.setBackgroundBrush(QtGui.QColor(C["well"]))
        if self.outline is not None:
            pen = QtGui.QPen(QtGui.QColor(C["rule"]), 1)
            pen.setCosmetic(True)
            self.outline.setPen(pen)

    def set_image(self, img: QtGui.QImage, fit: bool):
        if self.pix is None:
            self.pix = self.scene().addPixmap(QtGui.QPixmap.fromImage(img))
            self.outline = self.scene().addRect(QRectF())
            self.retheme()
        else:
            self.pix.setPixmap(QtGui.QPixmap.fromImage(img))
        r = QRectF(0, 0, img.width(), img.height())
        self.outline.setRect(r)
        m = 0.08 * max(r.width(), r.height())
        self.scene().setSceneRect(r.adjusted(-m, -m, m, m))
        if fit:
            self.fit()
        self._changed()

    def fit_scale(self) -> float:
        vr, sr = self.viewport().rect(), self.scene().sceneRect()
        return min(vr.width() / max(sr.width(), 1), vr.height() / max(sr.height(), 1))

    def fit(self):
        self.resetTransform()
        s = self.fit_scale()
        self.scale(s, s)
        self.centerOn(self.scene().sceneRect().center())
        self._changed()

    def zoom(self, f: float):
        cur = self.transform().m11()
        f = max(self.fit_scale() * 0.5, min(40.0, cur * f)) / cur     # between half the fit and 40x
        self.scale(f, f)
        self._changed()

    def _changed(self):
        if self.pix is not None:                    # past 100% show the real pixels, crisp
            sharp = self.transform().m11() >= 1.0
            self.pix.setTransformationMode(Qt.TransformationMode.FastTransformation if sharp
                                           else Qt.TransformationMode.SmoothTransformation)
        self.zoomed.emit(self.transform().m11())

    def wheelEvent(self, e):
        self.zoom(1.15 ** (e.angleDelta().y() / 120.0))

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.closed.emit()
        elif k in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom(1.25)
        elif k == Qt.Key.Key_Minus:
            self.zoom(0.8)
        elif k == Qt.Key.Key_0:
            self.fit()
        else:
            super().keyPressEvent(e)

    def mousePressEvent(self, e):
        self._press = e.position()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if (e.button() == Qt.MouseButton.LeftButton and self._press is not None
                and (e.position() - self._press).manhattanLength() < 5):   # a click, not a drag
            self.closed.emit()
        self._press = None


class Inspector(QtWidgets.QWidget):
    """Covers the preview area with one seal, large, plus zoom buttons. The sidebar stays
    usable, so stroke weight and the rest can be tuned while looking closely."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setObjectName("inspector")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.view = InspectView(self)
        v.addWidget(self.view, 1)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(16, 8, 12, 10)
        bar.setSpacing(6)
        self.caption = QtWidgets.QLabel()
        self.caption.setObjectName("muted")
        self.caption.setWordWrap(True)
        bar.addWidget(self.caption, 1)
        self.btn_out = QtWidgets.QToolButton()
        self.btn_out.setText("−")
        self.btn_out.setToolTip("Zoom out (−)")
        self.lbl_zoom = QtWidgets.QLabel()
        self.lbl_zoom.setObjectName("muted")
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_zoom.setMinimumWidth(self.fontMetrics().horizontalAdvance("4000%") + 8)
        self.btn_in = QtWidgets.QToolButton()
        self.btn_in.setText("+")
        self.btn_in.setToolTip("Zoom in (+)")
        self.btn_fit = QtWidgets.QToolButton()
        self.btn_fit.setText("Fit")
        self.btn_fit.setToolTip("Show the whole seal (0)")
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setToolTip("Back to normal (Esc, or click the seal)")
        for w in (self.btn_out, self.lbl_zoom, self.btn_in, self.btn_fit, self.btn_close):
            bar.addWidget(w)
        v.addLayout(bar)
        self.btn_out.clicked.connect(lambda: self.view.zoom(0.8))
        self.btn_in.clicked.connect(lambda: self.view.zoom(1.25))
        self.btn_fit.clicked.connect(self.view.fit)
        self.btn_close.clicked.connect(self.close_view)
        self.view.closed.connect(self.close_view)
        self.view.zoomed.connect(lambda s: self.lbl_zoom.setText(f"{s * 100:.0f}%"))
        self.left = 0
        self.key = None
        parent.installEventFilter(self)
        self.hide()

    def eventFilter(self, obj, e):
        if obj is self.parent() and e.type() == QtCore.QEvent.Type.Resize and self.isVisible():
            self._place()
        return False

    def _place(self):
        r = self.parent().rect()
        self.setGeometry(r.adjusted(self.left, 0, 0, 0))

    def open_over(self, left: int):
        self.left = left
        self._place()
        self.show()
        self.raise_()
        self.view.setFocus()

    def close_view(self):
        self.hide()
        self.key = None

    def set_image(self, img: QtGui.QImage, caption: str, fit: bool):
        self.caption.setText(caption)
        self.view.set_image(img, fit)


class PagePane(QtWidgets.QWidget):
    """A4 preview of one page of the PDF."""

    def __init__(self):
        super().__init__()
        self.images: list = []
        self.spots: list = []

    def set_content(self, images, spots):
        self.images, self.spots = images, spots
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(C["canvas"]))
        area = QRectF(self.rect()).adjusted(22, 22, -22, -10)
        s = min(area.width() / A4_W, area.height() / A4_H)
        page = QRectF(area.center().x() - A4_W * s / 2, area.center().y() - A4_H * s / 2, A4_W * s, A4_H * s)
        p.fillRect(page.translated(0, 3), QtGui.QColor(0, 0, 0, C["shadow_alpha"]))
        p.fillRect(page, QtGui.QColor("#ffffff"))
        p.setPen(QtGui.QColor(C["rule"]))
        p.drawRect(page.adjusted(-0.5, -0.5, 0.5, 0.5))
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        for j, x, y, w, h in self.spots:
            img = self.images[j] if j < len(self.images) else None
            if img is not None:
                p.drawImage(QRectF(page.x() + x * s, page.y() + y * s, w * s, h * s), img)


class SealStrip(QtWidgets.QListWidget):
    """Every seal in the PDF, as thumbnails. Pick one to edit it."""
    removeRequested = QtCore.Signal(int)
    duplicateRequested = QtCore.Signal(int)
    THUMB = 64

    def show_seal(self, item: QtWidgets.QListWidgetItem, job: SealJob, new_image: bool):
        item.setSizeHint(self.tile)
        item.setToolTip(job.name)
        item.setText(f"{job.size:g} mm × {job.copies}")
        if (new_image or item.icon().isNull()) and job.preview is not None:
            item.setIcon(seal_icon(job.preview, 2 * self.THUMB))

    def __init__(self):
        super().__init__()
        self.setObjectName("strip")
        self.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setMovement(QtWidgets.QListView.Movement.Static)
        self.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.setIconSize(QtCore.QSize(self.THUMB, self.THUMB))
        # Every tile gets a fixed size worked out from the font, so a tile never ends up
        # measured while its thumbnail is missing, and big system fonts still fit.
        fm = self.fontMetrics()
        self.tile = QtCore.QSize(max(self.THUMB + 28, fm.horizontalAdvance("22.5 mm × 99") + 18),
                                 self.THUMB + fm.height() + 22)
        self.setGridSize(self.tile + QtCore.QSize(6, 0))
        self.setFixedHeight(self.tile.height() + 15 + self.horizontalScrollBar().sizeHint().height())
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.setToolTip("Right-click a seal to duplicate or remove it")

    def _menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        row = self.row(item)
        menu = QtWidgets.QMenu(self)
        dup = menu.addAction("Duplicate")
        rem = menu.addAction("Remove")
        act = menu.exec(self.viewport().mapToGlobal(pos))
        if act is dup:
            self.duplicateRequested.emit(row)
        elif act is rem:
            self.removeRequested.emit(row)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.currentRow() >= 0:
            self.removeRequested.emit(self.currentRow())
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.count() == 0:
            p = QtGui.QPainter(self.viewport())
            p.setPen(QtGui.QColor(C["muted"]))
            p.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter,
                       "Each seal you add shows up here. They all go into the same PDF.")


class LibraryDialog(QtWidgets.QDialog):
    """Pick saved seals to put back on the sheet."""
    THUMB = 96

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Saved seals")
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(10)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search by name")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        v.addWidget(self.search)
        self.grid = QtWidgets.QListWidget()
        self.grid.setObjectName("library")
        self.grid.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.grid.setMovement(QtWidgets.QListView.Movement.Static)
        self.grid.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.grid.setWrapping(True)
        self.grid.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.grid.setIconSize(QtCore.QSize(self.THUMB, self.THUMB))
        fm = self.fontMetrics()
        self.tile = QtCore.QSize(max(self.THUMB + 34, fm.horizontalAdvance("M") * 9),
                                 self.THUMB + 2 * fm.height() + 24)
        self.grid.setGridSize(self.tile + QtCore.QSize(8, 8))
        self.grid.itemDoubleClicked.connect(self.accept)
        self.grid.itemSelectionChanged.connect(self._selection)
        v.addWidget(self.grid, 1)
        self.empty = QtWidgets.QLabel("Nothing saved yet. Select a seal on the sheet and press Save to keep it here.")
        self.empty.setObjectName("muted")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.empty)
        row = QtWidgets.QHBoxLayout()
        self.btn_folder = QtWidgets.QPushButton("Open folder")
        self.btn_folder.setToolTip(f"Show the library in your file manager, e.g. to back it up\n{LIBRARY}")
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_delete.setToolTip("Delete the selected seals from the library")
        self.btn_delete.clicked.connect(self._delete)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.btn_add = QtWidgets.QPushButton("Add to sheet")
        self.btn_add.setObjectName("primary")
        self.btn_add.setDefault(True)
        self.btn_add.clicked.connect(self.accept)
        for b in (self.btn_folder, self.btn_delete):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(self.btn_add)
        v.addLayout(row)
        self.resize(self.tile.width() * 5 + 90, self.tile.height() * 3 + 170)
        self._populate()

    def _populate(self):
        self.grid.clear()
        entries = library_entries()
        fm = self.grid.fontMetrics()
        for e in entries:
            name = e.get("name") or e["dir"].name
            size = fmt_mm(float(e.get("size", 0)))
            item = QtWidgets.QListWidgetItem(
                fm.elidedText(name, Qt.TextElideMode.ElideRight, self.tile.width() - 12) + "\n"
                + (f"{size} mm" if e.get("square", True) else f"longest {size} mm"))
            item.setToolTip(f"{name}\nSaved {e.get('saved', '')}")
            item.setSizeHint(self.tile)
            item.setData(Qt.ItemDataRole.UserRole, str(e["dir"]))
            item.setData(Qt.ItemDataRole.UserRole + 1, name.lower())
            thumb = QtGui.QImage(str(e["dir"] / "thumb.png"))
            if not thumb.isNull():
                item.setIcon(seal_icon(thumb, 2 * self.THUMB))
            self.grid.addItem(item)
        self.entries = {str(e["dir"]): e for e in entries}
        self.grid.setVisible(bool(entries))
        self.search.setEnabled(bool(entries))
        self.empty.setVisible(not entries)
        self._filter(self.search.text())
        self._selection()

    def _open_folder(self):
        LIBRARY.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(LIBRARY)))

    def _filter(self, text):
        text = text.strip().lower()
        for i in range(self.grid.count()):
            item = self.grid.item(i)
            item.setHidden(bool(text) and text not in item.data(Qt.ItemDataRole.UserRole + 1))

    def _selection(self):
        n = len(self.grid.selectedItems())
        self.btn_add.setEnabled(n > 0)
        self.btn_delete.setEnabled(n > 0)
        self.btn_add.setText("Add to sheet" if n < 2 else f"Add {n} to sheet")

    def chosen(self) -> list[dict]:
        return [self.entries[it.data(Qt.ItemDataRole.UserRole)] for it in self.grid.selectedItems()]

    def _delete(self):
        picked = self.chosen()
        if not picked:
            return
        what = f"“{picked[0].get('name')}”" if len(picked) == 1 else f"{len(picked)} saved seals"
        sure = QtWidgets.QMessageBox.question(self, "Delete saved seals",
                                              f"Delete {what} from the library? This can't be undone.")
        if sure != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for e in picked:
            shutil.rmtree(e["dir"], ignore_errors=True)
        self._populate()


class ThemeDialog(QtWidgets.QDialog):
    """Pick the two tones of a custom theme. Changes show straight away."""

    def __init__(self, parent, background: str, accent: str, preview):
        super().__init__(parent)
        self.setWindowTitle("Custom theme")
        self.tones = {"background": QtGui.QColor(background), "accent": QtGui.QColor(accent)}
        self.preview = preview
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(10)
        intro = QtWidgets.QLabel("A theme is two tones: a background and an accent. "
                                 "Every other colour is worked out from those two.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        form = QtWidgets.QFormLayout()
        self.buttons = {}
        for key in ("background", "accent"):
            b = QtWidgets.QPushButton()
            b.setMinimumWidth(self.fontMetrics().horizontalAdvance("#000000") + 60)
            b.clicked.connect(lambda _=False, k=key: self._pick(k))
            self.buttons[key] = b
            form.addRow(key.capitalize(), b)
        v.addLayout(form)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                         | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        v.addWidget(box)
        self.setMinimumWidth(self.fontMetrics().horizontalAdvance("M") * 28)
        self._show()

    def _pick(self, key):
        c = QtWidgets.QColorDialog.getColor(self.tones[key], self, f"Theme {key}")
        if c.isValid():
            self.tones[key] = c
            self._show()
            self.preview(*self.values())

    def _show(self):
        for key, b in self.buttons.items():
            hexa = self.tones[key].name()
            b.setIcon(swatch_icon(hexa, hexa))
            b.setText(hexa)

    def values(self):
        return self.tones["background"].name(), self.tones["accent"].name()


class CalibrateDialog(QtWidgets.QDialog):
    """Works out the enlargement from a test print: should be X mm, came out as Y mm."""

    def __init__(self, parent, scale_pct: float, size_mm: float, row_mm: float | None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate for a printer")
        self.scale_pct = scale_pct
        v = QtWidgets.QVBoxLayout(self)
        v.setSpacing(10)
        intro = QtWidgets.QLabel(f"Export a PDF at the current setting ({scale_pct:.2f}%), print it on the printer "
                                 f"you want to use, and measure a seal with a ruler.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        if row_mm:
            tip = QtWidgets.QLabel(f"For a more precise number, measure the top row of page 1 instead, from the "
                                   f"left edge of its first seal to the right edge of its last. That should be "
                                   f"{row_mm:.1f} mm.")
            tip.setObjectName("muted")
            tip.setWordWrap(True)
            v.addWidget(tip)
        form = QtWidgets.QFormLayout()
        self.should = QtWidgets.QDoubleSpinBox()
        self.measured = QtWidgets.QDoubleSpinBox()
        for box in (self.should, self.measured):
            box.setRange(1.0, 400.0)
            box.setDecimals(2)
            box.setSingleStep(0.1)
            box.setSuffix(" mm")
            box.setValue(size_mm)
            box.valueChanged.connect(self._update)
        form.addRow("Should be", self.should)
        form.addRow("Measured", self.measured)
        v.addLayout(form)
        self.result = QtWidgets.QLabel()
        self.result.setObjectName("muted")
        v.addWidget(self.result)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        v.addWidget(buttons)
        self.setMinimumWidth(self.fontMetrics().horizontalAdvance("M") * 32)
        self.measured.setFocus()
        self.measured.selectAll()
        self._update()

    def value(self) -> float:
        return min(200.0, max(50.0, self.scale_pct * self.should.value() / self.measured.value()))

    def _update(self, *_):
        printer = 100.0 * self.measured.value() / self.should.value() / (self.scale_pct / 100.0)
        text = f"The printer prints at {printer:.1f}%, so the new setting is {self.value():.2f}%."
        if not 80.0 <= printer <= 120.0:      # shrink-to-fit is usually a few percent
            text += "\nThat's a big difference. Check that both numbers are for the same seal."
        self.result.setText(text)


def _mix(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    return QtGui.QColor(round(a.red() + (b.red() - a.red()) * t), round(a.green() + (b.green() - a.green()) * t),
                        round(a.blue() + (b.blue() - a.blue()) * t))


def _lum(c: QtGui.QColor) -> float:
    def ch(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c.red()) + 0.7152 * ch(c.green()) + 0.0722 * ch(c.blue())


def _contrast(a: QtGui.QColor, b: QtGui.QColor) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def derive_theme(background: str, accent: str) -> dict:
    """Every interface colour from two tones."""
    base, acc = QtGui.QColor(background), QtGui.QColor(accent)
    W, K = QtGui.QColor("#ffffff"), QtGui.QColor("#000000")
    dark = _lum(base) < 0.2
    if dark:
        text, side = _mix(base, W, 0.885), _mix(base, W, 0.045)
        well, field = _mix(base, K, 0.10), _mix(base, K, 0.17)
        button, border, hover, pressed = (_mix(side, text, t) for t in (0.065, 0.16, 0.12, 0.03))
        handle_hover = _mix(text, W, 0.6)
    else:
        text, side = _mix(base, K, 0.86), _mix(base, K, 0.035)
        well, field = _mix(base, K, 0.055), _mix(base, W, 0.8)
        button, hover = _mix(base, W, 0.55), _mix(base, W, 0.85)
        border, pressed = _mix(side, text, 0.2), _mix(side, text, 0.08)
        handle_hover = _mix(text, acc, 0.5)
    label = acc                       # accent used as text must stay readable on the sidebar
    for i in range(21):
        label = _mix(acc, text, i / 20)
        if _contrast(label, side) >= 3.0:
            break
    c = dict(canvas=base, side=side, well=well, field=field, text=text, muted=_mix(text, base, 0.39 if dark else 0.34),
             rule=_mix(base, text, 0.16), button=button, border=border, hover=hover, pressed=pressed,
             dis_bg=_mix(side, text, 0.025), dis_border=_mix(side, text, 0.08), dis_text=_mix(side, text, 0.35),
             groove=_mix(side, text, 0.155), handle_hover=handle_hover, handle_dis=_mix(side, text, 0.3),
             accent=acc, accent_label=label,
             on_accent=_mix(acc, K, 0.85) if _lum(acc) > 0.3 else _mix(acc, W, 0.92),
             accent_hover=_mix(acc, W, 0.15) if _lum(acc) < 0.75 else _mix(acc, K, 0.1),
             accent_dis=_mix(side, acc, 0.18), accent_dis_text=_mix(side, text, 0.42),
             item_hover=_mix(base, text, 0.04), item_sel=_mix(base, acc, 0.13),
             light=_mix(side, text, 0.2), midlight=_mix(side, text, 0.16), mid=_mix(side, text, 0.065),
             darker=_mix(base, K, 0.25), shadow=_mix(base, K, 0.5))
    out = {k: v.name() for k, v in c.items()}
    out["shadow_alpha"] = 60 if dark else 40
    return out


def build_style(c: dict) -> str:
    return f"""
QWidget#sidebar {{ background: {c['side']}; border-right: 1px solid {c['rule']}; }}
QScrollArea#sidescroll, QWidget#sideinner {{ background: transparent; border: none; }}
QWidget#pagecol {{ background: {c['canvas']}; border-left: 1px solid {c['rule']}; }}
QWidget#inspector {{ background: {c['side']}; border-left: 1px solid {c['rule']}; }}
QLabel#title {{ font-size: 15pt; font-weight: 600; color: {c['text']}; }}
QLabel#section {{ color: {c['accent_label']}; font-weight: 600; padding-top: 10px; }}
QLabel#muted {{ color: {c['muted']}; }}
QPushButton, QToolButton {{ background: {c['button']}; color: {c['text']}; border: 1px solid {c['border']};
    border-radius: 5px; padding: 5px 10px; }}
QPushButton:hover, QToolButton:hover {{ background: {c['hover']}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {c['pressed']}; }}
QPushButton:focus, QToolButton:focus {{ border-color: {c['accent_label']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {c['dis_text']}; background: {c['dis_bg']};
    border-color: {c['dis_border']}; }}
QToolButton#themebtn {{ padding: 4px; }}
QToolButton#themebtn::menu-indicator {{ image: none; width: 0; }}
QPushButton#primary {{ background: {c['accent']}; color: {c['on_accent']}; border: 1px solid {c['accent']};
    font-weight: 600; padding: 9px 12px; }}
QPushButton#primary:hover {{ background: {c['accent_hover']}; }}
QPushButton#primary:focus {{ border-color: {c['text']}; }}
QPushButton#primary:disabled {{ background: {c['accent_dis']}; color: {c['accent_dis_text']};
    border-color: {c['accent_dis']}; }}
QSlider::groove:horizontal {{ height: 4px; background: {c['groove']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {c['text']}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::handle:horizontal:hover {{ background: {c['handle_hover']}; }}
QSlider::handle:horizontal:disabled {{ background: {c['handle_dis']}; }}
QSlider::sub-page:horizontal:disabled {{ background: {c['groove']}; }}
QListWidget#strip {{ background: {c['well']}; border: none; border-top: 1px solid {c['rule']}; padding: 6px 8px; }}
QListWidget#library {{ background: {c['canvas']}; border: 1px solid {c['rule']}; border-radius: 6px; padding: 6px; }}
QListWidget#strip::item, QListWidget#library::item {{ color: {c['muted']}; border: 1px solid transparent;
    border-radius: 6px; }}
QListWidget#strip::item:hover, QListWidget#library::item:hover {{ background: {c['item_hover']}; }}
QListWidget#strip::item:selected, QListWidget#library::item:selected {{ background: {c['item_sel']};
    color: {c['text']}; border: 1px solid {c['accent']}; }}
QLineEdit {{ background: {c['field']}; color: {c['text']}; border: 1px solid {c['border']}; border-radius: 5px;
    padding: 5px 8px; }}
QLineEdit:focus {{ border-color: {c['accent_label']}; }}
QStatusBar {{ background: {c['side']}; color: {c['muted']}; border-top: 1px solid {c['rule']}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: {c['rule']}; }}
QToolTip {{ background: {c['field']}; color: {c['text']}; border: 1px solid {c['rule']}; padding: 4px; }}
QMenu {{ background: {c['field']}; color: {c['text']}; border: 1px solid {c['rule']}; padding: 4px; }}
QMenu::item {{ padding: 5px 16px 5px 6px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c['item_sel']}; }}
QMenu::separator {{ height: 1px; background: {c['rule']}; margin: 4px 6px; }}
"""


def apply_theme(app: QtWidgets.QApplication, background: str | None = None, accent: str | None = None):
    """Switch the whole interface to a two-tone theme (the default one if none is given)."""
    if background is None or accent is None:
        background, accent = THEMES[DEFAULT_THEME]
    C.clear()
    C.update(derive_theme(background, accent))
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    R, G = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    for role, key in [(R.Window, "side"), (R.WindowText, "text"), (R.Base, "field"), (R.AlternateBase, "side"),
                      (R.Text, "text"), (R.Button, "button"), (R.ButtonText, "text"), (R.BrightText, "text"),
                      (R.Highlight, "accent"), (R.HighlightedText, "on_accent"), (R.ToolTipBase, "field"),
                      (R.ToolTipText, "text"), (R.PlaceholderText, "muted"), (R.Light, "light"),
                      (R.Midlight, "midlight"), (R.Mid, "mid"), (R.Dark, "darker"), (R.Shadow, "shadow"),
                      (R.Link, "accent_label")]:
        pal.setColor(role, QtGui.QColor(C[key]))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(G.Disabled, role, QtGui.QColor(C["dis_text"]))
    app.setPalette(pal)
    app.setStyleSheet(build_style(C))


def saved_theme(st: QtCore.QSettings):
    """(name, background, accent) of the theme the user picked last time."""
    name = st.value("theme", DEFAULT_THEME, type=str)
    if name in THEMES:
        return (name, *THEMES[name])
    tones = st.value("theme_custom", "", type=str).split(",")
    if name == "custom" and len(tones) == 2 and all(QtGui.QColor(t).isValid() for t in tones):
        return ("custom", *tones)
    return (DEFAULT_THEME, *THEMES[DEFAULT_THEME])


def swatch_icon(background: str, accent: str, px: int = 18) -> QtGui.QIcon:
    """A circle split into a theme's two tones."""
    dpr = 2
    pm = QtGui.QPixmap(px * dpr, px * dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    r = QRectF(dpr, dpr, (px - 2) * dpr, (px - 2) * dpr)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QtGui.QColor(background))
    p.drawEllipse(r)
    p.setBrush(QtGui.QColor(accent))
    p.drawPie(r, 45 * 16, -180 * 16)
    p.setPen(QtGui.QPen(QtGui.QColor(128, 128, 128, 150), dpr))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(r)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return QtGui.QIcon(pm)


def fmt_mm(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


# ───────────────────────────────────────────────────────────────── main window

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, specs=()):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = QtCore.QSettings("sealprint", "sealprint")
        self.jobs: list[SealJob] = []
        self.cur: SealJob | None = None
        self.page_idx = 0
        self._loading = False                     # True while controls are filled from a seal
        self.last_dir = self.settings.value("last_dir", str(Path.home()))
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self.refresh)
        self._build()
        self._restore()
        self.setAcceptDrops(True)
        self.editor.setFocus()
        for spec in specs:
            try:
                path, size, copies, square = parse_spec(spec, self.default_size, self.default_copies)
            except ValueError as ex:
                self.statusBar().showMessage(str(ex), 6000)
                continue
            self.open_path(path, size, copies, square)
        self._fill_controls()
        self.refresh()

    # ---- construction
    def _section(self, text):
        lab = QtWidgets.QLabel(text)
        lab.setObjectName("section")
        return lab

    def _muted(self, text="", wrap=True):
        lab = QtWidgets.QLabel(text)
        lab.setObjectName("muted")
        lab.setWordWrap(wrap)
        return lab

    def _form(self):
        f = QtWidgets.QFormLayout()
        f.setContentsMargins(0, 0, 0, 0)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        f.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return f

    def _build(self):
        inner = QtWidgets.QWidget()
        inner.setObjectName("sideinner")
        col = QtWidgets.QVBoxLayout(inner)
        col.setContentsMargins(18, 18, 18, 8)
        col.setSpacing(7)

        brand = QtWidgets.QHBoxLayout()
        brand.setSpacing(10)
        logo = QtWidgets.QLabel()
        logo.setPixmap(make_logo(26))
        title = QtWidgets.QLabel(APP_NAME)
        title.setObjectName("title")
        brand.addWidget(logo)
        brand.addWidget(title)
        brand.addStretch(1)
        self.btn_theme = QtWidgets.QToolButton()
        self.btn_theme.setObjectName("themebtn")
        self.btn_theme.setToolTip("Theme")
        self.btn_theme.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_theme.setIconSize(QtCore.QSize(18, 18))
        brand.addWidget(self.btn_theme)
        col.addLayout(brand)
        self._theme_menu()

        # Seals in this PDF
        col.addWidget(self._section("Seals"))
        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add…")
        self.btn_add.setToolTip("Add seal images (Ctrl+O). You can pick several at once.")
        self.btn_paste = QtWidgets.QPushButton("Paste")
        self.btn_paste.setToolTip("Add a screenshot from the clipboard (Ctrl+V)")
        self.btn_load = QtWidgets.QPushButton("Load…")
        self.btn_load.setToolTip("Add seals you saved before (Ctrl+L)")
        for b in (self.btn_add, self.btn_paste, self.btn_load):
            row.addWidget(b)
        col.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.setToolTip("Keep the selected seal, with its frame and settings, so Load can bring it\n"
                                 "back later (Ctrl+S)")
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.setToolTip("Take the selected seal off this sheet (Delete). Saved seals stay saved.")
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_remove)
        col.addLayout(row)
        self.lbl_count = self._muted()
        col.addWidget(self.lbl_count)

        # Frame of the selected seal
        col.addWidget(self._section("Frame"))
        self.lbl_file = self._muted()
        col.addWidget(self.lbl_file)
        f = self._form()
        self.cmb_ink = QtWidgets.QComboBox()
        self.cmb_ink.addItem("Auto", "auto")
        self.cmb_ink.addItem("Red seal paste", "red")
        self.cmb_ink.addItem("Black or dark", "dark")
        self.cmb_ink.setToolTip("Which marks count as the seal. Red ignores black writing and dark\n"
                                "screen backgrounds; black or dark takes any dark ink.")
        f.addRow("Ink", self.cmb_ink)
        col.addLayout(f)
        self.lbl_mode = self._muted()
        self.lbl_mode.hide()
        col.addWidget(self.lbl_mode)
        row = QtWidgets.QHBoxLayout()
        self.btn_detect = QtWidgets.QPushButton("Detect frame")
        self.btn_detect.setToolTip("Find the seal's frame again (Ctrl+R)")
        self.btn_whole = QtWidgets.QPushButton("Use whole image")
        row.addWidget(self.btn_detect)
        row.addWidget(self.btn_whole)
        col.addLayout(row)
        col.addWidget(self._muted("Drag the red corners to adjust. Scroll to zoom, double-click to fit."))

        # Settings of the selected seal
        col.addWidget(self._section("This seal"))
        f = self._form()
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItem("Square", "square")
        self.cmb_shape.addItem("Other shape", "other")
        self.cmb_shape.setToolTip("Square seals print exactly square. Other shapes (rectangle, oval, circle,\n"
                                  "free-form) keep the proportions of their frame.")
        f.addRow("Shape", self.cmb_shape)
        self.spn_size = QtWidgets.QDoubleSpinBox()
        self.spn_size.setRange(3.0, A4_W - 2 * PAGE_MARGIN)
        self.spn_size.setDecimals(1)
        self.spn_size.setSingleStep(0.5)
        self.spn_size.setSuffix(" mm")
        self.lbl_size = QtWidgets.QLabel("Size")
        f.addRow(self.lbl_size, self.spn_size)
        self.spn_copies = QtWidgets.QSpinBox()
        self.spn_copies.setRange(1, 999)
        self.spn_copies.setToolTip("How many copies of this seal go into the PDF")
        f.addRow("Copies", self.spn_copies)
        col.addLayout(f)
        self.lbl_dims = self._muted()
        self.lbl_dims.hide()
        col.addWidget(self.lbl_dims)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Stroke weight"))
        row.addStretch(1)
        self.btn_auto_t = QtWidgets.QPushButton("Auto")
        self.btn_auto_t.setToolTip("Let the app choose the weight from the image")
        row.addWidget(self.btn_auto_t)
        col.addLayout(row)
        self.sld_weight = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_weight.setRange(0, 255)
        self.sld_weight.setValue(128)
        self.sld_weight.setToolTip("Left gives thinner strokes, right gives bolder strokes")
        col.addWidget(self.sld_weight)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Smooth edges"))
        row.addStretch(1)
        self.lbl_smooth = self._muted(wrap=False)
        row.addWidget(self.lbl_smooth)
        col.addLayout(row)
        self.sld_smooth = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_smooth.setRange(0, 10)
        self.sld_smooth.setToolTip("Rounds off jagged pixel steps from small screenshots before\n"
                                   "the seal is turned into black and white")
        col.addWidget(self.sld_smooth)
        self.chk_invert = QtWidgets.QCheckBox("Invert")
        self.chk_invert.setToolTip("Swap black and white for this seal")
        col.addWidget(self.chk_invert)

        # Whole print
        col.addWidget(self._section("Print on A4"))
        f = self._form()
        self.spn_gap = QtWidgets.QDoubleSpinBox()
        self.spn_gap.setRange(0.0, 30.0)
        self.spn_gap.setDecimals(1)
        self.spn_gap.setSingleStep(0.5)
        self.spn_gap.setSuffix(" mm")
        self.spn_gap.setToolTip("Space between seals, for cutting them apart")
        f.addRow("Gap", self.spn_gap)
        self.spn_scale = QtWidgets.QDoubleSpinBox()
        self.spn_scale.setRange(50.0, 200.0)
        self.spn_scale.setDecimals(2)
        self.spn_scale.setSingleStep(0.1)
        self.spn_scale.setSuffix(" %")
        self.spn_scale.setToolTip("For printers that always shrink pages to fit and can't print at actual size,\n"
                                  "like many self-service printers. Everything in the PDF is enlarged by this\n"
                                  "much so the seals come out at their real size. Leave it at 100% otherwise.")
        self.btn_calib = QtWidgets.QPushButton("Calibrate…")
        self.btn_calib.setToolTip("Work out the enlargement from a test print")
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.setSpacing(6)
        scale_row.addWidget(self.spn_scale, 1)
        scale_row.addWidget(self.btn_calib)
        f.addRow("Enlarge", scale_row)
        col.addLayout(f)
        self.chk_mirror = QtWidgets.QCheckBox("Mirror every seal")
        self.chk_mirror.setToolTip("Flip left to right. Leave off for face-down toner transfer, which\n"
                                   "flips the design onto the stone by itself. Turn on if the print\n"
                                   "goes onto the stone face up.")
        col.addWidget(self.chk_mirror)
        self.lbl_fit = self._muted()
        col.addWidget(self.lbl_fit)
        col.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("sidescroll")
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)

        self.side_inner, self.side_scroll = inner, scroll
        self.side = side = QtWidgets.QWidget()
        side.setObjectName("sidebar")
        side.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        side.setFixedWidth(276)
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 16)
        sv.setSpacing(8)
        sv.addWidget(scroll, 1)
        self.btn_export = QtWidgets.QPushButton("Export PDF…")
        self.btn_export.setObjectName("primary")
        self.btn_export.setToolTip("Save every seal on A4 pages as one PDF (Ctrl+E)")
        wrap = QtWidgets.QHBoxLayout()
        wrap.setContentsMargins(18, 0, 18, 0)
        wrap.addWidget(self.btn_export)
        sv.addLayout(wrap)

        # Centre: source with frame and the result, above the strip of all seals
        self.editor = FrameEditor()
        self.seal_pane = SealPane()
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.editor)
        split.addWidget(self.seal_pane)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setHandleWidth(1)
        split.setChildrenCollapsible(False)
        split.setSizes([600, 400])
        self.strip = SealStrip()
        center = QtWidgets.QWidget()
        cv_ = QtWidgets.QVBoxLayout(center)
        cv_.setContentsMargins(0, 0, 0, 0)
        cv_.setSpacing(0)
        cv_.addWidget(split, 1)
        cv_.addWidget(self.strip)

        # Right: page preview
        right = QtWidgets.QWidget()
        right.setObjectName("pagecol")
        right.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        right.setFixedWidth(310)
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(1, 0, 0, 12)
        rv.setSpacing(4)
        self.page_pane = PagePane()
        rv.addWidget(self.page_pane, 1)
        nav = QtWidgets.QHBoxLayout()
        nav.setContentsMargins(22, 0, 22, 0)
        self.btn_prev = QtWidgets.QToolButton()
        self.btn_prev.setText("‹")
        self.btn_prev.setToolTip("Previous page")
        self.btn_next = QtWidgets.QToolButton()
        self.btn_next.setText("›")
        self.btn_next.setToolTip("Next page")
        self.lbl_page = self._muted(wrap=False)
        self.lbl_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_page, 1)
        nav.addWidget(self.btn_next)
        rv.addLayout(nav)

        central = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(side)
        h.addWidget(center, 1)
        h.addWidget(right)
        self.setCentralWidget(central)
        self.inspector = Inspector(central)
        self.lbl_frame = QtWidgets.QLabel()
        self.lbl_frame.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.lbl_frame)
        self.statusBar().setSizeGripEnabled(False)

        # Widgets that edit the selected seal
        self.per_seal = (self.btn_save, self.btn_remove, self.cmb_ink, self.btn_detect, self.btn_whole, self.cmb_shape, self.spn_size,
                         self.spn_copies, self.btn_auto_t, self.sld_weight, self.sld_smooth, self.chk_invert)

        def action(keys, slot):
            a = QtGui.QAction(self)
            a.setShortcuts(keys)
            a.triggered.connect(slot)
            self.addAction(a)
        K = QtGui.QKeySequence
        action([K(K.StandardKey.Open)], self.open_dialog)
        action([K(K.StandardKey.Paste)], self.paste)
        action([K("Ctrl+E")], self.export_pdf)
        action([K(K.StandardKey.Save)], self.save_seal)
        action([K("Ctrl+L")], self.load_saved)
        action([K("Ctrl+R")], self.redetect)
        action([K("Ctrl+D")], lambda: self.duplicate_seal(self.strip.currentRow()))

        self.btn_add.clicked.connect(self.open_dialog)
        self.btn_paste.clicked.connect(self.paste)
        self.btn_load.clicked.connect(self.load_saved)
        self.btn_save.clicked.connect(self.save_seal)
        self.btn_remove.clicked.connect(lambda: self.remove_seal(self.strip.currentRow()))
        self.strip.currentRowChanged.connect(self.select_seal)
        self.strip.removeRequested.connect(self.remove_seal)
        self.strip.duplicateRequested.connect(self.duplicate_seal)
        self.cmb_ink.currentIndexChanged.connect(self.on_ink_changed)
        self.cmb_shape.currentIndexChanged.connect(self.on_shape_changed)
        self.btn_detect.clicked.connect(self.redetect)
        self.btn_whole.clicked.connect(self.use_whole)
        self.editor.edited.connect(self.on_frame_edited)
        self.spn_size.valueChanged.connect(lambda v: self._set("size", v))
        self.spn_copies.valueChanged.connect(lambda v: self._set("copies", v))
        self.chk_invert.toggled.connect(lambda v: self._set("invert", v))
        self.sld_weight.valueChanged.connect(self.on_weight_moved)
        self.btn_auto_t.clicked.connect(self.on_auto_weight)
        self.sld_smooth.valueChanged.connect(self.on_smooth)
        self.spn_gap.valueChanged.connect(self.schedule)
        self.spn_scale.valueChanged.connect(self.schedule)
        self.btn_calib.clicked.connect(self.calibrate)
        self.chk_mirror.toggled.connect(self.schedule)
        self.btn_export.clicked.connect(self.export_pdf)
        self.seal_pane.clicked.connect(self.inspect)
        self.btn_prev.clicked.connect(lambda: self.turn_page(-1))
        self.btn_next.clicked.connect(lambda: self.turn_page(1))

    def showEvent(self, e):
        super().showEvent(e)
        # Size the sidebar to its contents in the current font (big system fonts need more room).
        need = (self.side_inner.minimumSizeHint().width()
                + self.side_scroll.verticalScrollBar().sizeHint().width() + 2)
        if need > self.side.width():
            self.side.setFixedWidth(need)
            self._fill_controls()

    # ---- themes
    def _theme_menu(self):
        self.theme_name, *tones = saved_theme(self.settings)
        self.custom_tones = tuple(tones) if self.theme_name == "custom" else tuple(
            self.settings.value("theme_custom", ",".join(THEMES[DEFAULT_THEME]), type=str).split(","))[:2]
        if len(self.custom_tones) != 2:
            self.custom_tones = THEMES[DEFAULT_THEME]
        menu = QtWidgets.QMenu(self)
        group = QtGui.QActionGroup(self)
        self.theme_actions = {}
        for name, (bg, acc) in THEMES.items():
            a = menu.addAction(swatch_icon(bg, acc), name)
            a.triggered.connect(lambda _=False, n=name: self.set_theme(n, *THEMES[n]))
            self.theme_actions[name] = a
        menu.addSeparator()
        a = menu.addAction(swatch_icon(*self.custom_tones), "Custom…")
        a.triggered.connect(self.custom_theme)
        self.theme_actions["custom"] = a
        for a in self.theme_actions.values():
            a.setCheckable(True)
            group.addAction(a)
        self.btn_theme.setMenu(menu)
        self._mark_theme(*tones)

    def _mark_theme(self, background, accent):
        for name, a in self.theme_actions.items():        # the theme in use shows in bold
            f = a.font()
            f.setBold(name == self.theme_name)
            a.setFont(f)
            a.setChecked(name == self.theme_name)
        self.btn_theme.setIcon(swatch_icon(background, accent))

    def set_theme(self, name: str, background: str, accent: str, remember: bool = True):
        apply_theme(QtWidgets.QApplication.instance(), background, accent)
        self.editor.retheme()
        self.inspector.view.retheme()
        for w in QtWidgets.QApplication.allWidgets():
            w.update()
        self.theme_name = name
        self._mark_theme(background, accent)
        if remember:
            self.settings.setValue("theme", name)
            if name == "custom":
                self.custom_tones = (background, accent)
                self.settings.setValue("theme_custom", f"{background},{accent}")
                self.theme_actions["custom"].setIcon(swatch_icon(background, accent))

    def custom_theme(self):
        before = (self.theme_name, C["canvas"], C["accent"])
        start = self.custom_tones if self.theme_name != "custom" else before[1:]
        dlg = ThemeDialog(self, *start, preview=lambda bg, acc: self.set_theme("custom", bg, acc, remember=False))
        self.set_theme("custom", *start, remember=False)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.set_theme("custom", *dlg.values())
        else:
            self.set_theme(*before, remember=False)

    def _restore(self):
        st = self.settings
        self.default_size = st.value("size", 20.0, type=float)
        self.default_copies = st.value("copies", 1, type=int)
        self.default_smooth = st.value("smooth", 2, type=int)
        self.spn_gap.setValue(st.value("gap", 5.0, type=float))
        self.spn_scale.setValue(st.value("scale", 100.0, type=float))
        self.chk_mirror.setChecked(st.value("mirror", False, type=bool))
        geo = st.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1320, 820)

    def closeEvent(self, e):
        st = self.settings
        if self.cur is not None:
            st.setValue("size", self.cur.size)
            st.setValue("copies", self.cur.copies)
            st.setValue("smooth", self.cur.smooth)
        st.setValue("gap", self.spn_gap.value())
        st.setValue("scale", self.spn_scale.value())
        st.setValue("mirror", self.chk_mirror.isChecked())
        st.setValue("last_dir", self.last_dir)
        st.setValue("geometry", self.saveGeometry())
        super().closeEvent(e)

    # ---- adding and removing seals
    def open_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add seal images", self.last_dir,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff);;All files (*)")
        for path in paths:
            self.open_path(path)

    def open_path(self, path: str, size: float | None = None, copies: int | None = None, square: bool = True):
        try:
            rgb = load_image(path)
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Couldn't open {Path(path).name}.\n{ex}")
            return
        self.last_dir = str(Path(path).parent)
        self.add_seal(rgb, Path(path).name, size, copies, square)

    def paste(self):
        cb = QtWidgets.QApplication.clipboard()
        img = cb.image()
        if not img.isNull():
            self.add_seal(qimage_to_rgb(img), "Pasted image")
            return
        md = cb.mimeData()
        if md is not None and md.hasUrls():
            files = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
            for f in files:
                self.open_path(f)
            if files:
                return
        self.statusBar().showMessage("The clipboard has no image. Take a screenshot, then paste again.", 5000)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasImage():
            e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        files = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
        for f in files:
            self.open_path(f)
        if not files and md.hasImage():
            img = md.imageData()
            if isinstance(img, QtGui.QImage) and not img.isNull():
                self.add_seal(qimage_to_rgb(img), "Dropped image")

    def add_seal(self, rgb: np.ndarray, name: str, size=None, copies=None, square: bool = True):
        base = self.cur          # a new seal starts with the size and copies of the one being edited
        job = SealJob(rgb, name,
                      size=size if size is not None else (base.size if base else self.default_size),
                      copies=copies if copies is not None else (base.copies if base else self.default_copies),
                      smooth=base.smooth if base else self.default_smooth, square=square)
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            job.detect()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._append(job)

    def _append(self, job: SealJob):
        job.update_preview(self.chk_mirror.isChecked())
        self.jobs.append(job)
        item = QtWidgets.QListWidgetItem()
        self.strip.show_seal(item, job, True)
        self.strip.addItem(item)
        self.strip.setCurrentRow(len(self.jobs) - 1)
        self.strip.scrollToItem(item)

    def save_seal(self):
        job = self.cur
        if job is None:
            return
        if job.lib_id and (LIBRARY / job.lib_id).is_dir():
            title, prompt, default = "Update saved seal", "This seal is already saved. Name:", job.name
        else:
            stem = Path(job.name).stem if not job.name.endswith(" image") else "Seal"
            title, prompt, default = "Save seal", "Name it so you can find it again:", stem
        name, ok = QtWidgets.QInputDialog.getText(self, title, prompt, QtWidgets.QLineEdit.EchoMode.Normal, default)
        name = name.strip()
        if not ok or not name:
            return
        try:
            save_to_library(job, name)
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Couldn't save the seal.\n{ex}")
            return
        job.name = name
        self._fill_controls()
        self.schedule()
        self.statusBar().showMessage(f"Saved “{name}”. Load… brings it back any time.", 6000)

    def load_saved(self):
        dlg = LibraryDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        failed = 0
        for entry in dlg.chosen():
            try:
                job = job_from_library(entry)
            except Exception:
                failed += 1
                continue
            self._append(job)
        if failed:
            self.statusBar().showMessage(f"{plural(failed, 'saved seal')} couldn't be read.", 6000)

    def remove_seal(self, row: int):
        if not 0 <= row < len(self.jobs):
            return
        self.jobs.pop(row)
        # Qt reports the new current row mid-removal, numbered as before the removal, so
        # silence the strip and resync once the item is really gone.
        self.strip.blockSignals(True)
        self.strip.takeItem(row)
        self.strip.blockSignals(False)
        self.select_seal(self.strip.currentRow())

    def duplicate_seal(self, row: int):
        if not 0 <= row < len(self.jobs):
            return
        twin = self.jobs[row].duplicate()
        twin.update_preview(self.chk_mirror.isChecked())
        self.jobs.insert(row + 1, twin)
        item = QtWidgets.QListWidgetItem()
        self.strip.show_seal(item, twin, True)
        self.strip.insertItem(row + 1, item)
        self.strip.setCurrentRow(row + 1)

    def select_seal(self, row: int):
        job = self.jobs[row] if 0 <= row < len(self.jobs) else None
        if job is None:
            self.editor.clear()
            self.seal_pane.set_image(None)
            self.lbl_frame.setText("")
        elif job is not self.cur:
            self.editor.set_image(rgb_to_qimage(job.rgb), job.corners)
        self.cur = job
        self._fill_controls()
        self.schedule()

    def _fill_controls(self):
        job = self.cur
        self._loading = True
        for w in self.per_seal:
            w.setEnabled(job is not None)
        self.btn_export.setEnabled(bool(self.jobs))
        if job is not None:
            self.cmb_ink.setCurrentIndex(max(0, self.cmb_ink.findData(job.ink_choice)))
            self.cmb_shape.setCurrentIndex(0 if job.square else 1)
            self.spn_size.setValue(job.size)
            self.spn_copies.setValue(job.copies)
            self.sld_smooth.setValue(job.smooth)
            self.chk_invert.setChecked(job.invert)
            self.sld_weight.setValue(job.weight)
            self.btn_auto_t.setEnabled(not job.auto_t)
            h, w = job.rgb.shape[:2]
            room = self.side.width() - 40 - self.side_scroll.verticalScrollBar().sizeHint().width()
            name = self.lbl_file.fontMetrics().elidedText(job.name, Qt.TextElideMode.ElideMiddle, max(80, room))
            self.lbl_file.setText(f"{name}\n{w} × {h} px")
        else:
            self.lbl_file.setText("No seal selected")
            self.spn_size.setValue(self.default_size)
            self.spn_copies.setValue(self.default_copies)
            self.sld_smooth.setValue(self.default_smooth)
        self.on_smooth(self.sld_smooth.value())
        self._shape_labels()
        self._mode_note()
        n = len(self.jobs)
        self.lbl_count.setText("Add one or more seal images. Each gets its own size and copies."
                               if n == 0 else f"{plural(n, 'seal')} in this PDF. Click one below to edit it.")
        self._loading = False

    # ---- frame of the selected seal
    def redetect(self):
        if self.cur is None:
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.cur.detect()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.editor.set_corners(self.cur.corners)
        self.btn_auto_t.setEnabled(False)
        self._mode_note()
        self.schedule()

    def use_whole(self):
        if self.cur is None:
            return
        h, w = self.cur.rgb.shape[:2]
        self.cur.corners = whole_image_corners(w, h)
        self.cur.frame_manual = True
        self.cur.auto_t = True
        self.btn_auto_t.setEnabled(False)
        self.editor.set_corners(self.cur.corners)
        self.schedule()

    def on_frame_edited(self):
        if self.cur is None:
            return
        self.cur.corners = self.editor.corners()
        self.cur.frame_manual = True
        self.schedule()

    def on_ink_changed(self, *_):
        if self._loading or self.cur is None:
            return
        job = self.cur
        job.ink_choice = self.cmb_ink.currentData()
        if not job.frame_manual:
            self.redetect()
            return
        # keep the hand-placed frame, change only how ink is read
        job.mode = detect_frame(job.rgb, "auto", job.square)[1] if job.ink_choice == "auto" else job.ink_choice
        job.auto_t = True
        self.btn_auto_t.setEnabled(False)
        self._mode_note()
        self.schedule()

    def on_shape_changed(self, *_):
        if self._loading or self.cur is None:
            return
        self.cur.square = self.cmb_shape.currentData() == "square"
        if self.cur.frame_manual:
            self.schedule()                    # keep the hand-placed frame
        else:
            self.redetect()                    # a square seal's box is squared, other shapes' isn't
        self._shape_labels()

    def _shape_labels(self):
        job = self.cur
        if job is None or job.square:
            self.lbl_size.setText("Size")
            self.spn_size.setToolTip("Side length of this printed seal")
            self.lbl_dims.hide()
            return
        self.lbl_size.setText("Longest")
        self.spn_size.setToolTip("The longest length of this seal: the long side of a rectangle, the long\n"
                                 "axis of an oval, the diameter of a circle")
        w, h = job.dims()
        self.lbl_dims.setText(f"Prints at {fmt_mm(w)} × {fmt_mm(h)} mm, proportions from its frame.")
        self.lbl_dims.show()

    def _mode_note(self):
        job, text = self.cur, ""
        if job is not None:
            if not job.found:
                text = "No frame found, so the whole image is used."
            elif job.ink_choice == "auto":
                text = "Found red seal paste." if job.mode == "red" else "Found black or dark ink."
        self.lbl_mode.setText(text)
        self.lbl_mode.setVisible(bool(text))

    # ---- settings of the selected seal
    def _set(self, attr, value):
        if self._loading or self.cur is None:
            return
        setattr(self.cur, attr, value)
        self.schedule()

    def on_weight_moved(self, v):
        if self._loading or self.cur is None:
            return
        self.cur.weight = v
        self.cur.auto_t = False
        self.btn_auto_t.setEnabled(True)
        self.schedule()

    def on_auto_weight(self):
        if self.cur is None:
            return
        self.cur.auto_t = True
        self.btn_auto_t.setEnabled(False)
        self.schedule()

    def on_smooth(self, v):
        self.lbl_smooth.setText("off" if v == 0 else f"{v * 0.2:.1f} px")
        self._set("smooth", v)

    def turn_page(self, step):
        self.page_idx += step
        self.schedule()

    def schedule(self, *_):
        self.timer.start()

    # ---- preview
    def refresh(self):
        mirror = self.chk_mirror.isChecked()
        for i, job in enumerate(self.jobs):
            self.strip.show_seal(self.strip.item(i), job, job.update_preview(mirror))

        job = self.cur
        if job is not None and job.preview is not None:
            self._loading = True
            self.sld_weight.setValue(job.weight)          # follows the automatic weight
            self._loading = False
            e = self.scale()
            w_mm, h_mm = job.dims()
            paper = f"{fmt_mm(w_mm)} × {fmt_mm(h_mm)} mm"
            if abs(e - 1.0) < 1e-6:
                ow, oh = job.out_px(export_px(max(w_mm, h_mm)))
                caption = f"{paper}\n{ow} × {oh} px in the PDF"
            else:
                caption = f"{paper} on paper\n{w_mm * e:.2f} × {h_mm * e:.2f} mm in the PDF ({e * 100:.2f}%)"
            self.seal_pane.set_image(job.preview, caption)
            self._shape_labels()
            self._frame_info(job)

        try:
            pages = layout_pages(self.jobs, self.spn_gap.value(), self.scale()) if self.jobs else []
        except ValueError as ex:
            pages = []
            self.lbl_fit.setText(str(ex))
        else:
            total = sum(j.copies for j in self.jobs)
            note = "" if abs(self.scale() - 1.0) < 1e-6 else f" Scaled to {self.spn_scale.value():.2f}% for your printer."
            self.lbl_fit.setText(f"{plural(total, 'seal')} on {plural(len(pages), 'page')}.{note}" if pages
                                 else "Nothing to print yet.")
        n_pages = max(1, len(pages))
        self.page_idx = max(0, min(self.page_idx, n_pages - 1))
        self.lbl_page.setText(f"Page {self.page_idx + 1} of {n_pages}")
        self.btn_prev.setEnabled(self.page_idx > 0)
        self.btn_next.setEnabled(self.page_idx < n_pages - 1)
        self.page_pane.set_content([j.preview for j in self.jobs], pages[self.page_idx] if pages else [])
        if self.inspector.isVisible():
            self._inspect_update()

    # ---- close-up inspection of the seal, exactly as it goes into the PDF
    def inspect(self):
        if self.cur is None or not self.cur.frame_ok():
            return
        self.inspector.key = None
        self.inspector.open_over(self.side.width())   # placed first, so "fit" knows the real size
        self._inspect_update()

    def _inspect_update(self):
        job = self.cur
        if job is None or not job.frame_ok():
            self.inspector.close_view()
            return
        key = (id(job), job.preview_key, job.size, self.scale())
        if key == self.inspector.key:
            return
        fit = self.inspector.key is None or self.inspector.key[0] != id(job)    # new seal: show it whole
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            img = ink_to_qimage(job.render(self.chk_mirror.isChecked(), self.scale()))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        w_mm, h_mm = job.dims()
        self.inspector.set_image(img, f"{fmt_mm(w_mm)} × {fmt_mm(h_mm)} mm, {img.width()} × {img.height()} px, "
                                      f"exactly as it goes into the PDF. Scroll or use + and − to zoom, drag to "
                                      f"move, click to go back.", fit)
        self.inspector.key = key

    def scale(self) -> float:
        return self.spn_scale.value() / 100.0

    def calibrate(self):
        e = self.scale()
        row_mm = None
        try:
            pages = layout_pages(self.jobs, self.spn_gap.value(), e) if self.jobs else []
        except ValueError:
            pages = []
        if pages:                                   # paper length of page 1's top row
            top = min(spot[2] for spot in pages[0])
            row = [(x, d) for _, x, y, d, _ in pages[0] if abs(y - top) < 1e-6]
            if len(row) > 1:
                row_mm = (max(x + d for x, d in row) - min(x for x, _ in row)) / e
        size = self.cur.size if self.cur is not None else self.default_size
        dlg = CalibrateDialog(self, self.spn_scale.value(), size, row_mm)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.spn_scale.setValue(dlg.value())
            self.statusBar().showMessage(f"Enlargement set to {dlg.value():.2f}%. Export again to use it.", 8000)

    def _frame_info(self, job):
        w, h = job.frame_px()
        text = f"Frame {w:.0f} × {h:.0f} px, {'red' if job.mode == 'red' else 'dark'} ink"
        if job.square and abs(w - h) / max(w, h, 1) > 0.06:
            text = f"Frame {w:.0f} × {h:.0f} px is not square and will be stretched. Drag the corners if that's wrong."
        elif job.square and job.found and not job.frame_manual and job.ratio < 0.96:
            text += ". It doesn't look square: for a rectangle or oval, set Shape to Other shape."
        self.lbl_frame.setText(text)

    # ---- output
    def export_pdf(self):
        if not self.jobs:
            return
        if len(self.jobs) == 1:
            j = self.jobs[0]
            stem = Path(j.name).stem if not j.name.endswith(" image") else "seal"
            suggestion, title = f"{stem}_{j.size:g}mm_x{j.copies}.pdf", f"{stem} {j.size:g} mm"
        else:
            suggestion, title = "seal_sheet.pdf", "Seal sheet"
        pct = self.spn_scale.value()
        if abs(pct - 100.0) > 1e-6:                  # mark enlarged PDFs in the file name
            suggestion = suggestion[:-4] + f"_{pct:g}pct.pdf"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PDF", str(Path(self.last_dir) / suggestion),
                                                        "PDF (*.pdf)")
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data, pages = make_pdf(self.jobs, self.spn_gap.value(), self.chk_mirror.isChecked(), title,
                                   self.scale())
            Path(path).write_bytes(data)
        except Exception as ex:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"Couldn't save the PDF.\n{ex}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.last_dir = str(Path(path).parent)
        total = sum(j.copies for j in self.jobs)
        where = f"{plural(total, 'seal')} on {plural(len(pages), 'page')}"
        self.statusBar().showMessage(f"Saved {Path(path).name}: {where}", 8000)
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("PDF saved")
        box.setText(f"Saved {Path(path).name} ({where}).")
        if abs(pct - 100.0) < 1e-6:
            box.setInformativeText("Print at 100% or “Actual size”. “Fit to page” shrinks the seals, "
                                   "so measure one to check its size.")
        else:
            box.setInformativeText(f"Everything is scaled to {pct:.2f}% for the printer you calibrated for. "
                                   f"Print it there and measure a seal to check its size.")
        open_btn = box.addButton("Open PDF", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is open_btn:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))


# ───────────────────────────────────────────────────────────────── entry points

def parse_spec(spec: str, size: float, copies: int, square: bool = True):
    """'seal.png', 'seal.png:20', 'seal.png:20:6' or 'seal.png:30:2:other'
    ->  (path, size_mm, copies, square). For other shapes the size is the longest length."""
    if Path(spec).is_file():
        return spec, size, copies, square
    parts = spec.split(":")
    if len(parts) > 1 and parts[-1].lower() in ("square", "other"):
        path, size, copies, _ = parse_spec(":".join(parts[:-1]), size, copies, square)
        return path, size, copies, parts[-1].lower() == "square"
    for cut in (2, 1):
        if len(parts) > cut and Path(":".join(parts[:-cut])).is_file():
            try:
                nums = [float(v) for v in parts[-cut:]]
            except ValueError:
                continue
            return ":".join(parts[:-cut]), nums[0], (int(nums[1]) if cut == 2 else copies), square
    raise ValueError(f"Can't find the image {spec}")


def use_wayland_text_input():
    """Make Chinese (or any input method) typing work on Wayland.

    Desktops like Omarchy set QT_IM_MODULE=fcitx, which asks Qt to load fcitx's own Qt
    plugin. The copy of Qt installed with pip doesn't contain that plugin, so no input
    method would reach this app at all. Wayland's built-in text-input protocol does the
    same job, and fcitx5 supports it on Hyprland and most other compositors.
    """
    wanted = os.environ.get("QT_IM_MODULE", "")
    if not sys.platform.startswith("linux") or not os.environ.get("WAYLAND_DISPLAY") or wanted in ("", "wayland"):
        return
    folder = Path(QtCore.QLibraryInfo.path(QtCore.QLibraryInfo.LibraryPath.PluginsPath)) / "platforminputcontexts"
    if folder.is_dir() and any(wanted.rstrip("5") in f.name for f in folder.iterdir()):
        return                                  # this Qt can load the requested plugin itself
    os.environ["QT_IM_MODULE"] = "wayland"
    os.environ.setdefault("QT_WAYLAND_TEXT_INPUT_PROTOCOL", "zwp_text_input_v3")


def run_cli(a) -> int:
    app = QtCore.QCoreApplication(sys.argv[:1])   # lets Qt load its image plugins
    jobs = []
    for spec in a.images:
        path, size, copies, square = parse_spec(spec, a.size, a.copies, a.shape == "square")
        job = SealJob(load_image(path), Path(path).name, size=size, copies=copies, smooth=a.smooth,
                      ink_choice=a.ink, invert=a.invert, square=square)
        job.detect()
        if a.threshold is not None:
            job.auto_t, job.weight = False, int(round(255 - a.threshold))
        jobs.append(job)
        w, h = job.dims()
        print(f"{path}: {fmt_mm(w)} × {fmt_mm(h)} mm × {copies}, {job.mode} ink"
              + ("" if job.found else ", no frame found so the whole image is used"))
    data, pages = make_pdf(jobs, a.gap, a.mirror, "Seal sheet" if len(jobs) > 1 else Path(jobs[0].name).stem,
                           a.scale / 100.0)
    Path(a.output).write_bytes(data)
    print(f"{a.output}: {plural(sum(j.copies for j in jobs), 'seal')} on {plural(len(pages), 'page')}")
    del app
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Print square seal designs at exact sizes on A4.")
    ap.add_argument("images", nargs="*",
                    help="seal images, optionally as image:size_mm, image:size_mm:copies or "
                         "image:size_mm:copies:other (non-square; size is then the longest length)")
    ap.add_argument("--shape", choices=["square", "other"], default="square",
                    help="default shape; for 'other' the size is the longest length (square)")
    ap.add_argument("-o", "--output", help="export a PDF without opening the window")
    ap.add_argument("--size", type=float, default=20.0, help="default seal side in mm (20)")
    ap.add_argument("--copies", type=int, default=1, help="default copies per seal (1)")
    ap.add_argument("--gap", type=float, default=5.0, help="mm between seals (5)")
    ap.add_argument("--ink", choices=["auto", "red", "dark"], default="auto")
    ap.add_argument("--threshold", type=float, help="0-255, default automatic")
    ap.add_argument("--smooth", type=int, default=2, help="edge smoothing 0-10 (2)")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--scale", type=float, default=100.0,
                    help="enlarge everything by this percent, for printers that shrink pages (100)")
    a = ap.parse_args()
    if a.output:
        if not a.images:
            ap.error("give at least one image with --output")
        try:
            return run_cli(a)
        except ValueError as ex:
            print(ex, file=sys.stderr)
            return 1

    use_wayland_text_input()                    # must happen before the QApplication exists
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("sealprint")
    app.setApplicationDisplayName(APP_NAME)
    app.setDesktopFileName("sealprint")
    icon = QtGui.QIcon()
    for s in (16, 32, 64, 128):
        icon.addPixmap(make_logo(s, 1.0))
    app.setWindowIcon(icon)
    _, background, accent = saved_theme(QtCore.QSettings("sealprint", "sealprint"))
    apply_theme(app, background, accent)
    win = MainWindow(a.images)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
