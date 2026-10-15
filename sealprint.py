#!/usr/bin/env python3
"""
Seal Print: print Chinese seal designs at their exact size on A4.

Add one or more seal images: files, screenshots pasted with Ctrl+V, or dropped files.
For each image the app finds the seal's frame (square, or any other shape) and turns
the seal into pure black and white. Every seal has its own size and number of copies,
and all of them are packed onto A4 pages in a single PDF.

    python3 sealprint.py [image ...]                                # open the window
    python3 sealprint.py a.png:20:6 b.png:30:2:other -o seals.pdf   # no window; image:size_mm:copies[:other]
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
__version__ = "1.4.0"
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

    def render_long(self, long_px: int) -> np.ndarray:
        """Black/white seal with its long side long_px pixels (True = black), never mirrored."""
        if self.auto_t:
            self.weight = int(round(255 - self._warped_preview()[3]))
        warped, pps = warp_ink(self.ink(), self.corners, *self.out_px(long_px))
        return to_black_white(warped, 255 - self.weight, self.smooth * 0.2, pps, self.invert, False)

    def duplicate(self) -> SealJob:
        return replace(self, corners=self.corners.copy(), lib_id=None)   # shares image and ink maps


# ───────────────────────────────────────────────────────────────── seal library

# ~/.local/share/sealprint/library on Linux, %LOCALAPPDATA%\sealprint\library on Windows
LIBRARY = Path(QtCore.QStandardPaths.writableLocation(
    QtCore.QStandardPaths.StandardLocation.GenericDataLocation)) / "sealprint" / "library"
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


# ───────────────────────────────────────────────────────────────── characters and builds

CHARACTERS = LIBRARY.parent / "characters"   # one folder per character picked out of a seal
BUILD_PX_PER_MM = 50                         # resolution a built seal is saved at
FORMATS = [                                  # name, shape, width mm, height mm
    ("Square", "rect", 20.0, 20.0),
    ("Tall rectangle", "rect", 15.0, 25.0),
    ("Wide rectangle", "rect", 25.0, 15.0),
    ("Circle", "ellipse", 20.0, 20.0),
    ("Oval", "ellipse", 18.0, 24.0),
]


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + os.urandom(3).hex()


def ink_to_png(ink: np.ndarray) -> bytes:
    ok, png = cv2.imencode(".png", np.where(ink, 0, 255).astype(np.uint8))
    if not ok:
        raise OSError("couldn't encode the image")
    return png.tobytes()


def png_to_ink(path) -> np.ndarray:
    g = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError(f"can't read {path}")
    return g < 128


def tidy_glyph(ink: np.ndarray, drop_edge_pieces: bool = False, edge: np.ndarray | None = None) -> np.ndarray | None:
    """Clean a cropped character: drop specks (and, if asked, bits cut off at the edge of the
    box, or of a hand-drawn outline given as `edge`), then trim it to its ink with an even
    margin. None if nothing is left."""
    H, W = ink.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    cut = set(np.unique(lab[edge & ink]).tolist()) if (drop_edge_pieces and edge is not None) else set()
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < max(4, H * W * 2e-4):
            continue
        if drop_edge_pieces and (i in cut if edge is not None else (x == 0 or y == 0 or x + w >= W or y + h >= H)):
            continue
        keep[i] = True
    ink = keep[lab]
    if not ink.any():
        return None
    ys, xs = np.nonzero(ink)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = max(2, round(0.03 * max(ink.shape)))
    return np.pad(ink, m)


def character_entries() -> list[dict]:
    """Saved characters, sorted by name. Unreadable entries are skipped."""
    entries = []
    if CHARACTERS.is_dir():
        for d in CHARACTERS.iterdir():
            try:
                meta = json.loads((d / "char.json").read_text("utf-8"))
            except (OSError, ValueError):
                continue
            if (d / "glyph.png").is_file():
                meta["dir"] = d
                entries.append(meta)
    entries.sort(key=lambda e: (e.get("name", ""), e.get("saved", "")))
    return entries


def save_character(glyph: np.ndarray, name: str, parent_id: str, parent_name: str, rect, poly=None) -> str:
    """Save a picked character. rect is where it was on its seal (fractions of the seal's size);
    poly the outline drawn around it, if it was drawn by hand."""
    cid = new_id()
    d = CHARACTERS / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "glyph.png").write_bytes(ink_to_png(glyph))
    meta = {"version": 1, "name": name, "parent_id": parent_id, "parent_name": parent_name,
            "rect": [float(v) for v in rect], "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    if poly is not None:
        meta["poly"] = [[round(float(x), 5), round(float(y), 5)] for x, y in poly]
    (d / "char.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    return cid


@dataclass(eq=False)
class Placed:
    """A character placed in a build. Its picture (src) is never changed: squish, crop and
    weight are settings, and stroke edits are a list of operations replayed on the picture.
    Each of them can be switched off ("Original") and on again without losing anything."""
    glyph_id: str
    name: str
    src: np.ndarray
    cx: float = 10.0                 # mm: where the middle of src sits on the seal
    cy: float = 10.0
    scale: float = 0.02              # mm per src pixel
    sq_x: float = 1.0                # squish: 0.8 is 80% as wide
    sq_y: float = 1.0
    use_squish: bool = True
    crop: list | None = None         # [x0, y0, x1, y1] in src pixels, None if never cropped
    use_crop: bool = True
    weight: int = 0                  # -5 (thinner) .. +5 (bolder)
    use_weight: bool = True
    ops: list = field(default_factory=list)   # stroke edits, in order
    use_strokes: bool = True
    use_lengthen: bool = True
    use_erase: bool = True
    invert: bool = False
    z: float = 0.0                   # stacking: higher is on top where characters overlap
    cache: tuple | None = None


PLACED_FIELDS = ("glyph_id", "name", "cx", "cy", "scale", "sq_x", "sq_y", "use_squish", "crop", "use_crop",
                 "weight", "use_weight", "ops", "use_strokes", "use_lengthen", "use_erase", "invert", "z")
OP_SWITCH = {"drag": "use_strokes", "lengthen": "use_lengthen", "erase": "use_erase"}


@dataclass(eq=False)
class Build:
    shape: str = "rect"              # "rect" or "ellipse"
    w_mm: float = 20.0
    h_mm: float = 20.0
    frame_mm: float = 1.0            # frame line thickness, 0 for none
    invert_all: bool = False         # white text (bai wen): ink and paper swapped inside the shape
    chars: list = field(default_factory=list)
    corner_mm: float = 0.0           # rounded corners for rectangles, 0 for square ones


EDIT_PAD = 0.35                      # room around a character for strokes to be pulled into


def edit_pad(p: Placed) -> int:
    return max(8, round(EDIT_PAD * max(p.src.shape)))


def thicken(g: np.ndarray, weight: int, ref: float) -> np.ndarray:
    """Bolder (weight > 0) or thinner strokes; ref is the character's own size in pixels."""
    if not weight:
        return g
    r = max(1, round(abs(weight) * ref * 0.005))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    g8 = np.pad(g.astype(np.uint8), r)
    g8 = cv2.dilate(g8, kernel) if weight > 0 else cv2.erode(g8, kernel)
    return g8[r:-r, r:-r] > 0


def active_ops(p: Placed) -> list:
    return [op for op in p.ops if getattr(p, OP_SWITCH[op["kind"]])]


def erase_mask(shape, points, radius: float, offset: float = 0.0) -> np.ndarray:
    """Where an eraser path (round brush) has been."""
    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(points, np.float64).reshape(-1, 2) + offset).astype(np.int32)
    r = max(1, int(round(radius)))
    if len(pts) > 1:
        cv2.polylines(m, [pts.reshape(-1, 1, 2)], False, 1, 2 * r + 1)
    for x, y in pts[[0, -1]]:
        cv2.circle(m, (int(x), int(y)), r, 1, -1)
    return m > 0


def apply_op(g: np.ndarray, op: dict, pad: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Replay one stroke edit on a padded canvas: (canvas, the stroke afterwards, its centreline)."""
    if op["kind"] == "erase":
        return g & ~erase_mask(g.shape, op["points"], op["radius"], pad), None, None
    found = select_stroke(g, np.asarray(op["path"], np.float64) + pad)
    if found is None:
        return g, None, None
    sel, centre = found
    if op["kind"] == "drag":
        d = np.asarray(op["d"], np.float64)
        drag = StrokeDrag(g, sel, centre, np.asarray(op["grab"], np.float64) + pad, reach=float(np.hypot(*d)) + 8)
        return drag.apply(d)
    return lengthen_stroke(g, sel, centre, float(op["amount"]), int(op["end"]))


def edit_canvas(p: Placed, ops: list | None = None) -> np.ndarray:
    """The picture on a padded canvas with stroke edits replayed (no weight or crop yet)."""
    pad = edit_pad(p)
    g = np.pad(p.src, pad)
    for op in (active_ops(p) if ops is None else ops):
        g = apply_op(g, op, pad)[0]
    return g


def process_char(p: Placed) -> tuple[np.ndarray, int, int]:
    """The character as it appears, trimmed to its ink: (image, ox, oy), where (ox, oy) is
    where the picture's own pixel (0, 0) falls in that image."""
    ops = active_ops(p)
    crop = tuple(p.crop) if (p.crop is not None and p.use_crop) else None
    key = (id(p.src), json.dumps(ops, sort_keys=True), p.weight if p.use_weight else 0, crop)
    if p.cache is not None and p.cache[0] == key:
        return p.cache[1], p.cache[2], p.cache[3]
    pad = edit_pad(p) if ops else 0
    g = edit_canvas(p, ops) if ops else p.src
    if p.use_weight and p.weight:
        g = thicken(g, p.weight, max(p.src.shape))
    if crop is not None:
        x0, y0, x1, y1 = (int(round(v)) + pad for v in crop)
        keep = np.zeros_like(g)
        keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        g = g & keep
    ys, xs = np.nonzero(g)
    if xs.size:
        img = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        ox, oy = pad - int(xs.min()), pad - int(ys.min())
    else:
        img, ox, oy = np.zeros((1, 1), bool), 0, 0
    p.cache = (key, img, ox, oy)
    return img, ox, oy


def char_scale(p: Placed) -> tuple[float, float]:
    """mm per picture pixel, across and down, with the squish if it is on."""
    return (p.scale * (p.sq_x if p.use_squish else 1.0), p.scale * (p.sq_y if p.use_squish else 1.0))


def char_rect(p: Placed) -> tuple[float, float, float, float]:
    """Where the character's ink sits on the seal: (x, y, width, height) in mm."""
    img, ox, oy = process_char(p)
    H0, W0 = p.src.shape
    sx, sy = char_scale(p)
    return p.cx - (ox + W0 / 2) * sx, p.cy - (oy + H0 / 2) * sy, img.shape[1] * sx, img.shape[0] * sy


def to_src(p: Placed, x_mm: float, y_mm: float) -> tuple[float, float]:
    """A point on the seal in the character's own picture pixels."""
    H0, W0 = p.src.shape
    sx, sy = char_scale(p)
    return (x_mm - p.cx) / sx + W0 / 2, (y_mm - p.cy) / sy + H0 / 2


def _resample(points: np.ndarray, step: float) -> np.ndarray:
    """Points spaced evenly along a polyline."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-9]
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def _nearest_param(c: np.ndarray, s: np.ndarray, P: np.ndarray) -> np.ndarray:
    """For each point of P, how far along the polyline c (arc lengths s) its nearest point is."""
    A, AB = c[:-1], np.diff(c, axis=0)
    L2 = (AB ** 2).sum(1) + 1e-9
    out = np.empty(len(P), np.float32)
    for i in range(0, len(P), 4096):
        q = P[i:i + 4096, None, :]
        t = np.clip(((q - A) * AB).sum(2) / L2, 0.0, 1.0)
        d2 = ((q - (A + t[..., None] * AB)) ** 2).sum(2)
        j = d2.argmin(1)
        out[i:i + 4096] = s[j] + t[np.arange(len(j)), j] * np.sqrt(L2[j])
    return out


def select_stroke(ink: np.ndarray, path) -> tuple[np.ndarray, np.ndarray] | None:
    """Best guess at the stroke (or the part of one) a path was drawn along: the ink within
    about one stroke width of the path, plus a smoothed centreline. None if the path missed."""
    pts = _resample(path, 1.0)
    if len(pts) < 3:
        return None
    H, W = ink.shape
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    k = max(2, round(0.025 * max(H, W)))
    local = cv2.dilate(dt, np.ones((2 * k + 1, 2 * k + 1), np.uint8))   # stroke half-width nearby
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    half = local[yi, xi]
    half = half[half > 0]
    if half.size < 2:
        return None
    hw = float(np.median(half))
    ks = max(1, int(round(hw)))                       # move the path onto the middle of its stroke,
    snapped = []                                      # wherever across the stroke it was drawn
    for x, y in pts:
        xc, yc = int(round(x)), int(round(y))
        x0, x1, y0, y1 = max(0, xc - ks), min(W, xc + ks + 1), max(0, yc - ks), min(H, yc + ks + 1)
        win = dt[y0:y1, x0:x1]
        if win.size and win.max() > 0:
            j = np.unravel_index(int(win.argmax()), win.shape)
            snapped.append((x0 + j[1], y0 + j[0]))
        else:
            snapped.append((x, y))
    pts = np.asarray(snapped, dtype=np.float64)
    r = 1.3 * hw + 2.0
    reach, core = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    for x, y in pts[::2]:
        c = (int(round(x)), int(round(y)))
        cv2.circle(reach, c, int(round(r)), 1, -1)
        cv2.circle(core, c, max(2, int(round(r / 2))), 1, -1)
    sel = ink & (reach > 0)
    n, lab = cv2.connectedComponents(sel.astype(np.uint8), connectivity=8)
    hit = np.unique(lab[(core > 0) & sel])
    hit = hit[hit > 0]
    if not hit.size:
        return None
    sel = np.isin(lab, hit)
    win = max(3, int(r)) | 1                          # smooth the hand-drawn path into a centreline
    kern = np.ones(win) / win
    padded = np.pad(pts, ((win // 2, win // 2), (0, 0)), mode="edge")
    centre = np.column_stack([np.convolve(padded[:, i], kern, mode="valid") for i in (0, 1)])
    return sel, centre


class StrokeDrag:
    """Pull or bend a selected stroke like dough. The grabbed point follows the mouse, the end
    of the selection farthest from it stays where it is, and the part in between follows
    smoothly, so pulling along the stroke lengthens it and pulling sideways bends it."""

    def __init__(self, ink: np.ndarray, sel: np.ndarray, centre: np.ndarray, grab, reach: float | None = None):
        self.ink, self.sel = ink, sel
        c = _resample(centre, max(1.0, float(np.hypot(*np.diff(centre, axis=0).T).sum()) / 80))
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
        self.c, self.s, self.L = c, s, max(float(s[-1]), 1e-6)
        self.tg = float(_nearest_param(c, s, np.asarray(grab, np.float64).reshape(1, 2))[0])
        self.from_start = self.tg >= self.L / 2       # anchored at the end farther from the grab
        H, W = ink.shape
        ys, xs = np.nonzero(sel)
        if reach is None:
            reach = 0.45 * max(H, W)                  # the most a drag can move anything
        x0, x1 = max(0, int(xs.min() - reach)), min(W, int(xs.max() + reach) + 1)
        y0, y1 = max(0, int(ys.min() - reach)), min(H, int(ys.max() + reach) + 1)
        self.box = (x0, y0, x1, y1)
        step = 4                                      # how far along the stroke each pixel is, on a coarse grid
        gx, gy = np.meshgrid(np.arange(x0, x1, step, dtype=np.float64), np.arange(y0, y1, step, dtype=np.float64))
        T = _nearest_param(c, s, np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        self.T = cv2.resize(T, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        self.X, self.Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
        self.sel_f = sel.astype(np.float32)

    def weight(self, t):
        if self.from_start:
            u = np.clip(t / max(self.tg, 1e-6), 0.0, 1.0)
        else:
            u = np.clip((self.L - t) / max(self.L - self.tg, 1e-6), 0.0, 1.0)
        return u * u * (3.0 - 2.0 * u)                # smooth at both ends of the bend

    def apply(self, d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The character with the stroke dragged by d pixels: (ink, selection, centreline)."""
        dx, dy = float(d[0]), float(d[1])
        x0, y0, x1, y1 = self.box
        w = self.weight(self.T)
        xs, ys = self.X - dx * w, self.Y - dy * w
        for _ in range(5):                            # find where each pixel came from
            t = cv2.remap(self.T, xs - x0, ys - y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            w = self.weight(t)
            xs, ys = self.X - dx * w, self.Y - dy * w
        moved = cv2.remap(self.sel_f, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0) >= 0.5
        ink = self.ink & ~self.sel
        ink[y0:y1, x0:x1] |= moved
        sel = np.zeros_like(self.sel)
        sel[y0:y1, x0:x1] = moved
        return ink, sel, self.c + np.outer(self.weight(self.s), (dx, dy))


def fit_glyph(g: np.ndarray, gw: int, gh: int) -> np.ndarray:
    shrink = gw < g.shape[1] or gh < g.shape[0]
    f = cv2.resize(g.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    return f >= 0.5


def _rounded(W: int, H: int, inset: float, radius: float) -> np.ndarray:
    """The inside of a rectangle with rounded corners, set in from the image's edges; corners
    set in further get smaller radii, so a frame between two of them is evenly thick."""
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    x0, y0, x1, y1 = inset, inset, W - inset, H - inset
    inside = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    r = radius - inset
    if r > 0:
        nx, ny = np.clip(xx, x0 + r, x1 - r), np.clip(yy, y0 + r, y1 - r)
        inside &= (xx - nx) ** 2 + (yy - ny) ** 2 <= r * r
    return inside


def _ellipse(W: int, H: int, a: float, b: float) -> np.ndarray:
    if a <= 0 or b <= 0:
        return np.zeros((H, W), bool)
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx + 0.5 - W / 2) / a) ** 2 + ((yy + 0.5 - H / 2) / b) ** 2 <= 1.0


def _poly(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arc length at each point of a polyline, and the unit direction of each segment."""
    seg = np.diff(c, axis=0)
    ln = np.hypot(seg[:, 0], seg[:, 1])
    return np.r_[0.0, np.cumsum(ln)], seg / np.maximum(ln, 1e-9)[:, None]


def _point_at(c: np.ndarray, s_c: np.ndarray, u_c: np.ndarray, s: np.ndarray):
    """Points and directions at arc lengths s along a polyline."""
    j = np.clip(np.searchsorted(s_c, s, side="right") - 1, 0, len(u_c) - 1)
    return c[j] + u_c[j] * (s - s_c[j])[:, None], u_c[j]


def _project(c: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point of P: arc length of its nearest point on polyline c, and its signed
    distance from c (positive to the left of the direction of travel)."""
    s_c, u_c = _poly(c)
    A, AB = c[:-1], np.diff(c, axis=0)
    L2 = (AB ** 2).sum(1) + 1e-9
    normal = np.column_stack([-u_c[:, 1], u_c[:, 0]])
    S, N = np.empty(len(P)), np.empty(len(P))
    for i in range(0, len(P), 4096):
        q = P[i:i + 4096, None, :]
        t = np.clip(((q - A) * AB).sum(2) / L2, 0.0, 1.0)
        proj = A + t[..., None] * AB
        j = ((q - proj) ** 2).sum(2).argmin(1)
        r = np.arange(len(j))
        S[i:i + 4096] = s_c[j] + t[r, j] * np.sqrt(L2[j])
        N[i:i + 4096] = ((q[:, 0, :] - proj[r, j]) * normal[j]).sum(1)
    return S, N


def _half_width(sel: np.ndarray, c: np.ndarray) -> float:
    H, W = sel.shape
    dt = cv2.distanceTransform(sel.astype(np.uint8), cv2.DIST_L2, 5)
    xi = np.clip(np.round(c[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(c[:, 1]).astype(int), 0, H - 1)
    v = dt[yi, xi]
    return max(1.0, float(np.median(v[v > 0])) if (v > 0).any() else 1.0)


def free_end(ink: np.ndarray, sel: np.ndarray, centre: np.ndarray) -> int:
    """Which end of a selected stroke to lengthen (0 = start, 1 = end of its centreline): the one
    that is a free tip, if the other runs on into more ink. Otherwise the end it was drawn towards."""
    c = _resample(centre, 1.0)
    if len(c) < 3:
        return 1
    H, W = ink.shape
    hw = _half_width(sel, c)
    k = min(6, len(c) - 1)

    def attached(pt, out):
        out = out / max(np.hypot(*out), 1e-9)
        for ang in (0.0, 0.6, -0.6, 1.2, -1.2):       # straight on, and turning off to either side
            ca, sa = math.cos(ang), math.sin(ang)
            d = np.array([out[0] * ca - out[1] * sa, out[0] * sa + out[1] * ca])
            for dist in np.linspace(hw + 2, 2.5 * hw + 4, 5):
                x, y = int(round(pt[0] + d[0] * dist)), int(round(pt[1] + d[1] * dist))
                if 0 <= x < W and 0 <= y < H and ink[y, x] and not sel[y, x]:
                    return True
        return False

    start_attached = attached(c[0], c[0] - c[k])
    end_attached = attached(c[-1], c[-1] - c[-1 - k])
    if end_attached and not start_attached:
        return 0
    return 1


def _insertion_index(sel: np.ndarray, c: np.ndarray, s_c: np.ndarray, hw: float) -> int:
    """Where to add length to a stroke: a straight spot, clear of strokes crossing it, not too
    close to either end of the selection."""
    n, L = len(c), float(s_c[-1])
    lo, hi = int(np.searchsorted(s_c, 0.15 * L)), int(np.searchsorted(s_c, 0.85 * L))
    if hi - lo < 3:
        return n // 2
    H, W = sel.shape
    k = max(2, int(round(hw)))
    idx = np.arange(n)
    tang = c[np.minimum(n - 1, idx + k)] - c[np.maximum(0, idx - k)]
    tang /= np.maximum(np.hypot(tang[:, 0], tang[:, 1]), 1e-9)[:, None]
    before, after = tang[np.maximum(0, idx - k)], tang[np.minimum(n - 1, idx + k)]
    turn = np.arccos(np.clip((before * after).sum(1), -1.0, 1.0))           # how much it bends here
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    offs = np.arange(-3.0 * hw, 3.0 * hw + 1.0)
    px = np.clip(np.round(c[:, None, 0] + normal[:, None, 0] * offs).astype(int), 0, W - 1)
    py = np.clip(np.round(c[:, None, 1] + normal[:, None, 1] * offs).astype(int), 0, H - 1)
    width = sel[py, px].sum(1).astype(np.float64)                            # how wide the stroke is here:
    typical = float(np.median(width[lo:hi]))                                 # wider where another stroke
    odd = np.abs(width - typical) / hw                                       # crosses, narrower at a notch
    score = turn + 0.6 * odd + 0.4 * np.abs(s_c - 0.5 * L) / max(L, 1e-9)
    return lo + int(np.argmin(score[lo:hi]))


def lengthen_stroke(ink: np.ndarray, sel: np.ndarray, centre: np.ndarray, amount: float, end: int = 1):
    """Make a selected stroke longer by `amount` (a fraction of its length) towards one end. Length
    is added at the straightest spot of the stroke, and everything beyond it (a hook, a turn, the
    tip) moves out as one piece, so the stroke keeps its thickness and the shape of its end.
    Returns (ink, stroke, centreline)."""
    c = _resample(centre, 1.0)
    if amount <= 0 or len(c) < 6:
        return ink, sel, centre
    if end == 0:
        c = c[::-1]
    H, W = ink.shape
    hw = _half_width(sel, c)
    s_c, _ = _poly(c)
    dL = amount * float(s_c[-1])
    i = _insertion_index(sel, c, s_c, hw)
    k = max(2, int(round(hw)))
    t = c[min(len(c) - 1, i + k)] - c[max(0, i - k)]
    t /= max(np.hypot(*t), 1e-9)                      # the stroke's direction where length is added
    k = min(6, len(c) - 1)
    t0 = (c[0] - c[k]) / max(np.hypot(*(c[0] - c[k])), 1e-9)
    t1 = (c[-1] - c[-1 - k]) / max(np.hypot(*(c[-1] - c[-1 - k])), 1e-9)
    ext = 2 * hw + 3                                  # so the stroke's rounded ends project onto it
    step = max(1.0, float(s_c[-1]) / 60)              # a coarser outline is plenty to measure along
    head, tail = _resample(c[:i + 1], step), _resample(c[i:], step)
    old = np.vstack([c[0] + t0 * ext, head, tail[1:], c[-1] + t1 * ext])
    new = np.vstack([c[0] + t0 * ext, head, tail + t * dL, c[-1] + t * dL + t1 * ext])
    si = ext + float(_poly(head)[0][-1])              # where the new piece starts, along the stroke
    r = 3.0 * hw + ext + 2
    x0, y0 = (int(v) for v in np.floor(new.min(0) - r))
    x1, y1 = (int(v) for v in np.ceil(new.max(0) + r))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return ink, sel, centre
    near = np.zeros((y1 - y0, x1 - x0), np.uint8)     # only pixels near the new stroke can be ink
    reach = int(math.ceil(2.2 * hw + 4))
    cv2.polylines(near, [np.round(new - (x0, y0)).astype(np.int32).reshape(-1, 1, 2)], False, 1, 2 * reach + 1)
    ys, xs = np.nonzero(near)
    S_new, N = _project(new, np.column_stack([xs + x0, ys + y0]).astype(np.float64))
    S_old = np.where(S_new <= si, S_new, np.where(S_new <= si + dL, si, S_new - dL))
    s_old, u_old = _poly(old)
    pts, tang = _point_at(old, s_old, u_old, S_old)
    src = (pts + np.column_stack([-tang[:, 1], tang[:, 0]]) * N[:, None]).astype(np.float32)
    n = len(src)                                      # remap takes at most 32767 columns: fold into rows
    rows = max(1, -(-n // 4096))
    fold = lambda v: np.pad(v, (0, rows * 4096 - n), constant_values=-10).reshape(rows, 4096)
    vals = cv2.remap(sel.astype(np.float32), fold(src[:, 0]), fold(src[:, 1]), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).ravel()[:n]
    moved = np.zeros((y1 - y0, x1 - x0), bool)
    moved[ys, xs] = vals >= 0.5
    out = ink & ~sel
    out[y0:y1, x0:x1] |= moved
    nsel = np.zeros_like(sel)
    nsel[y0:y1, x0:x1] = moved
    ncentre = np.vstack([c[:i + 1], c[i:] + t * dL])
    return out, nsel, (ncentre[::-1] if end == 0 else ncentre)


def shape_inside(b: Build, W: int, H: int, px_per_mm: float) -> np.ndarray:
    """Which pixels of the build's image are inside its outline."""
    if b.shape == "ellipse":
        return _ellipse(W, H, W / 2, H / 2)
    if b.corner_mm > 0:
        return _rounded(W, H, 0, b.corner_mm * px_per_mm)
    return np.ones((H, W), bool)


def render_build(b: Build, px_per_mm: float) -> np.ndarray:
    """The built seal as a bool image (True = black)."""
    W, H = max(8, round(b.w_mm * px_per_mm)), max(8, round(b.h_mm * px_per_mm))
    ink = np.zeros((H, W), bool)
    for p in sorted(b.chars, key=lambda q: q.z):     # later ones on top; paper is see-through
        img = process_char(p)[0]
        x, y, w, h = char_rect(p)
        gw, gh = max(1, round(w * px_per_mm)), max(1, round(h * px_per_mm))
        g = fit_glyph(img, gw, gh)
        if p.invert:
            g = ~g
        x0, y0 = round(x * px_per_mm), round(y * px_per_mm)
        dx0, dy0, dx1, dy1 = max(0, x0), max(0, y0), min(W, x0 + gw), min(H, y0 + gh)
        if dx1 <= dx0 or dy1 <= dy0:
            continue
        part = g[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0]
        if p.invert:
            ink[dy0:dy1, dx0:dx1] = part
        else:
            ink[dy0:dy1, dx0:dx1] |= part
    t = b.frame_mm * px_per_mm
    o = t if b.invert_all else 0.0       # white text: the frame becomes a white line set in from a solid rim
    if b.shape == "ellipse":
        inside = _ellipse(W, H, W / 2, H / 2)
        if t > 0:
            ink |= _ellipse(W, H, W / 2 - o, H / 2 - o) & ~_ellipse(W, H, W / 2 - o - t, H / 2 - o - t)
    elif b.corner_mm > 0:
        R = b.corner_mm * px_per_mm
        inside = _rounded(W, H, 0, R)
        if t > 0:
            ink |= _rounded(W, H, o, R) & ~_rounded(W, H, o + t, R)
    else:
        inside = np.ones((H, W), bool)
        if t > 0:
            a, k = round(o), max(1, round(t))
            band = np.zeros((H, W), bool)
            band[a:H - a, a:W - a] = True
            band[a + k:H - a - k, a + k:W - a - k] = False
            ink |= band
    return (~ink & inside) if b.invert_all else (ink & inside)


def fit_into(p: Placed, x: float, y: float, w: float, h: float):
    """Scale a character (keeping its proportions) to fit a box, centred in it."""
    cx0, cy0, cw, ch = char_rect(p)
    p.scale *= min(w / max(cw, 1e-6), h / max(ch, 1e-6))
    cx0, cy0, cw, ch = char_rect(p)
    p.cx += (x + w / 2) - (cx0 + cw / 2)
    p.cy += (y + h / 2) - (cy0 + ch / 2)


def auto_arrange(b: Build):
    """Fit the characters into the seal in reading order: columns from right to left, each
    column top to bottom, like a traditional seal. Characters keep their proportions."""
    n = len(b.chars)
    if not n:
        return
    edge = b.frame_mm * (2 if b.invert_all else 1) + 0.06 * min(b.w_mm, b.h_mm)
    aw, ah = max(1.0, b.w_mm - 2 * edge), max(1.0, b.h_mm - 2 * edge)
    if b.shape == "ellipse":                         # the largest rectangle inside the oval
        aw, ah = aw / math.sqrt(2), ah / math.sqrt(2)
    x0, y0 = (b.w_mm - aw) / 2, (b.h_mm - ah) / 2
    if n == 1:
        columns = [1]
    elif aw / ah > 1.4:                              # wide: one row
        columns = [1] * n
    elif aw / ah < 0.7:                              # tall: one column
        columns = [n]
    elif n == 3:                                     # right column one character, left column two
        columns = [1, 2]
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        columns = [rows] * (n // rows) + ([n % rows] if n % rows else [])
    gap = 0.04 * min(aw, ah)
    colw = (aw - gap * (len(columns) - 1)) / len(columns)
    chars = iter(b.chars)
    for c, k in enumerate(columns):
        x = x0 + aw - (c + 1) * colw - c * gap
        cellh = (ah - gap * (k - 1)) / k
        for i in range(k):
            fit_into(next(chars), x, y0 + i * (cellh + gap), colw, cellh)


def scale_build(b: Build, w_mm: float, h_mm: float):
    """Change the format's size, keeping the characters where they are relative to it."""
    fx, fy = w_mm / b.w_mm, h_mm / b.h_mm
    for p in b.chars:
        p.cx, p.cy, p.scale = p.cx * fx, p.cy * fy, p.scale * min(fx, fy)
    b.w_mm, b.h_mm = w_mm, h_mm


def save_build(b: Build, name: str, lib_id: str | None = None) -> str:
    """Save a build as an ordinary seal in the library (so it prints and loads like any
    other), plus its layout so it can be edited again. Returns the seal's id."""
    ink = render_build(b, BUILD_PX_PER_MM)
    H, W = ink.shape
    rgb = np.repeat(np.where(ink, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    job = SealJob(rgb, name, size=max(b.w_mm, b.h_mm), smooth=1, ink_choice="dark", mode="dark",
                  frame_manual=True, corners=whole_image_corners(W, H), square=abs(b.w_mm - b.h_mm) < 1e-6,
                  ratio=min(W, H) / max(W, H), lib_id=lib_id)
    job.update_preview(False)
    save_to_library(job, name)
    d = LIBRARY / job.lib_id
    layout = {"version": 2, "shape": b.shape, "w_mm": b.w_mm, "h_mm": b.h_mm, "frame_mm": b.frame_mm,
              "invert_all": b.invert_all, "corner_mm": b.corner_mm,
              "chars": [{k: getattr(p, k) for k in PLACED_FIELDS} for p in b.chars]}
    (d / "build.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), "utf-8")
    gd = d / "glyphs"                                # its own copy of each character's picture
    gd.mkdir(exist_ok=True)
    used = {p.glyph_id for p in b.chars}
    for p in b.chars:
        (gd / f"{p.glyph_id}.png").write_bytes(ink_to_png(p.src))
    for f in gd.glob("*.png"):
        if f.stem not in used:
            f.unlink()
    return job.lib_id


def _from_v1(c: dict, g: np.ndarray) -> Placed:
    """A character saved by version 1.1 or 1.2 (a box size and band lengthening) in today's terms."""
    for axis, amount, at in ((0, c.get("stretch_v", 0), c.get("stretch_v_at", 0.5)),
                             (1, c.get("stretch_h", 0), c.get("stretch_h_at", 0.5))):
        if amount > 0:                                # bake the old lengthening into the picture
            n_rows = g.shape[axis]
            r, n = min(n_rows - 1, max(0, round(at * (n_rows - 1)))), max(1, round(amount * n_rows))
            g = np.concatenate([g.take(range(r), axis), np.repeat(g.take([r], axis), n, axis), g.take(range(r, n_rows), axis)], axis)
    H, W = g.shape
    scale = c["w"] / W
    return Placed(c["glyph_id"], c["name"], g, cx=c["cx"], cy=c["cy"], scale=scale, sq_x=1.0,
                  sq_y=(c["h"] / H) / scale, weight=int(c.get("weight", 0)), invert=bool(c.get("invert", False)))


def load_build(entry: dict) -> Build:
    d = Path(entry["dir"])
    data = json.loads((d / "build.json").read_text("utf-8"))
    b = Build(data["shape"], data["w_mm"], data["h_mm"], data["frame_mm"], data["invert_all"],
              corner_mm=float(data.get("corner_mm", 0.0)))
    for c in data["chars"]:
        g = png_to_ink(d / "glyphs" / f"{c['glyph_id']}.png")
        if data.get("version", 1) < 2:
            b.chars.append(_from_v1(c, g))
            continue
        p = Placed(c["glyph_id"], c["name"], g)
        for k in PLACED_FIELDS[2:]:
            if k in c:
                setattr(p, k, c[k])
        b.chars.append(p)
    return b


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


class PickView(QtWidgets.QGraphicsView):
    """A seal in black and white. Drag a box around a character, or in draw mode draw an
    outline around it by hand; scroll to zoom."""
    boxDrawn = QtCore.Signal(QRectF)          # in image pixels
    lassoDrawn = QtCore.Signal(object)        # the outline, an N x 2 array of image pixels
    pickClicked = QtCore.Signal(str)          # id of a saved character that was clicked

    def __init__(self, img: QtGui.QImage):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QtGui.QColor(C["well"]))
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        pix = self.scene().addPixmap(QtGui.QPixmap.fromImage(img))
        pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.img_size = QtCore.QSizeF(img.width(), img.height())
        m = 0.04 * max(img.width(), img.height())
        self.scene().setSceneRect(QRectF(-m, -m, img.width() + 2 * m, img.height() + 2 * m))
        pen = QtGui.QPen(QtGui.QColor(CINNABAR), 2)
        pen.setCosmetic(True)
        self.box = self.scene().addRect(QRectF(), pen, QtGui.QBrush(QtGui.QColor(216, 69, 47, 45)))
        self.box.setZValue(5)
        self.box.hide()
        self.lasso = self.scene().addPath(QtGui.QPainterPath(), pen, QtGui.QBrush(QtGui.QColor(216, 69, 47, 45)))
        self.lasso.setZValue(5)
        self.lasso.hide()
        self.mode = "box"
        self.points: list = []
        self.picks = []
        self.start = None
        self.auto_fit = True

    def show_picks(self, picks):
        """Outline the characters already saved from this seal: (id, name, normalised rect,
        normalised hand-drawn outline or None)."""
        for item in self.picks:
            self.scene().removeItem(item)
        self.picks = []
        W, H = self.img_size.width(), self.img_size.height()
        for cid, name, (x, y, w, h), poly in picks:
            pen = QtGui.QPen(QtGui.QColor(CINNABAR), 1.5, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            if poly:
                path = QtGui.QPainterPath(QPointF(poly[0][0] * W, poly[0][1] * H))
                for px, py in poly[1:]:
                    path.lineTo(px * W, py * H)
                path.closeSubpath()
                r = self.scene().addPath(path, pen)
            else:
                r = self.scene().addRect(QRectF(x * W, y * H, w * W, h * H), pen)
            r.setData(0, cid)
            r.setZValue(3)
            label = QtWidgets.QGraphicsSimpleTextItem(name)
            label.setBrush(QtGui.QColor(CINNABAR))
            label.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            label.setPos(x * W, y * H)
            label.setZValue(4)
            self.scene().addItem(label)
            self.picks += [r, label]

    def clear_box(self):
        self.box.hide()
        self.lasso.hide()

    def _clicked(self, end: QPointF):
        """A click rather than a drag: select the saved character under it, if any."""
        self.box.hide()
        self.lasso.hide()
        for item in self.picks:
            if isinstance(item, QtWidgets.QAbstractGraphicsShapeItem) and item.data(0) and item.contains(end):
                self.pickClicked.emit(item.data(0))
                break

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.auto_fit = True

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.auto_fit:
            self.fit()

    def wheelEvent(self, e):
        f = 1.15 ** (e.angleDelta().y() / 120.0)
        vr, sr = self.viewport().rect(), self.sceneRect()
        if self.transform().m11() * f <= min(vr.width() / sr.width(), vr.height() / sr.height()):
            self.fit()
            return
        self.scale(f, f)
        self.auto_fit = False

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        self.start = self.mapToScene(e.position().toPoint())
        if self.mode == "draw":
            self.points = [self.start]
            self.lasso.setPath(QtGui.QPainterPath(self.start))
            self.lasso.show()
            self.box.hide()
        else:
            self.box.setRect(QRectF(self.start, self.start))
            self.box.show()
            self.lasso.hide()

    def mouseMoveEvent(self, e):
        if self.start is None:
            return super().mouseMoveEvent(e)
        pt = self.mapToScene(e.position().toPoint())
        if self.mode == "draw":
            self.points.append(pt)
            path = self.lasso.path()
            path.lineTo(pt)
            self.lasso.setPath(path)
        else:
            self.box.setRect(QRectF(self.start, pt).normalized())

    def mouseReleaseEvent(self, e):
        if self.start is None or e.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(e)
        end = self.mapToScene(e.position().toPoint())
        self.start = None
        if self.mode == "draw":
            pts = np.array([[p.x(), p.y()] for p in self.points + [end]])
            self.points = []
            if len(pts) < 4 or np.ptp(pts[:, 0]) < 4 or np.ptp(pts[:, 1]) < 4:
                return self._clicked(end)
            path = QtGui.QPainterPath(QPointF(*pts[0]))
            for x, y in pts[1:]:
                path.lineTo(x, y)
            path.closeSubpath()                         # the outline closes itself
            self.lasso.setPath(path)
            pts[:, 0] = np.clip(pts[:, 0], 0, self.img_size.width())
            pts[:, 1] = np.clip(pts[:, 1], 0, self.img_size.height())
            self.lassoDrawn.emit(pts)
            return
        r = QRectF(self.box.rect()).intersected(QRectF(QPointF(0, 0), self.img_size))
        if r.width() < 4 or r.height() < 4:
            return self._clicked(end)
        self.box.setRect(r)
        self.boxDrawn.emit(r)


def glyph_icon(path, px: int) -> QtGui.QIcon:
    img = QtGui.QImage(str(path))
    return seal_icon(img, px) if not img.isNull() else QtGui.QIcon()


class CharacterPicker(QtWidgets.QDialog):
    """Cut single characters out of a saved seal and name them."""

    def __init__(self, parent, entry: dict):
        super().__init__(parent)
        self.parent_id = Path(entry["dir"]).name
        self.parent_name = entry.get("name") or self.parent_id
        self.setWindowTitle(f"Pick characters from “{self.parent_name}”")
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.ink = job_from_library(entry).render_long(1600)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.region = self.mask = self.edge = self.rect = self.poly = None
        h = QtWidgets.QHBoxLayout(self)
        h.setSpacing(12)
        self.view = PickView(ink_to_qimage(self.ink))
        h.addWidget(self.view, 1)
        side = QtWidgets.QVBoxLayout()
        side.setSpacing(8)
        h.addLayout(side)
        fm = self.fontMetrics()
        width = max(240, fm.horizontalAdvance("M") * 19)
        modes = QtWidgets.QHBoxLayout()
        modes.setSpacing(4)
        modes.addWidget(QtWidgets.QLabel("Select with"))
        self.mode_buttons = {}
        for key, label, tip in (("box", "Box", "Drag a box around the character"),
                                ("draw", "Draw", "Draw an outline around the character by hand, "
                                                 "for characters that interlock with their neighbours")):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            modes.addWidget(b)
            self.mode_buttons[key] = b
        self.mode_buttons["box"].setChecked(True)
        modes.addStretch(1)
        side.addLayout(modes)
        self.hint = muted_label("Drag a box around one character. Scroll to zoom.")
        self.hint.setFixedWidth(width)
        side.addWidget(self.hint)
        self.preview = QtWidgets.QLabel()
        self.preview.setFixedSize(width, min(width, 170))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setObjectName("glyphpreview")
        side.addWidget(self.preview)
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("Which character is it?")
        self.name.returnPressed.connect(self.save)
        self.name.textChanged.connect(lambda _: self._buttons())
        side.addWidget(self.name)
        self.chk_flip = QtWidgets.QCheckBox("White strokes: flip them to black")
        self.chk_flip.setToolTip("For white-text seals, so the strokes become the ink of the character")
        self.chk_edge = QtWidgets.QCheckBox("Drop bits cut off at the edge")
        self.chk_edge.setToolTip("Removes pieces of the frame or of neighbouring characters that the box "
                                 "or outline clipped")
        for c in (self.chk_flip, self.chk_edge):
            c.toggled.connect(self._show_preview)
            side.addWidget(c)
        self.btn_save = QtWidgets.QPushButton("Save character")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self.save)
        side.addWidget(self.btn_save)
        side.addWidget(section_label("From this seal"))
        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("library")
        self.list.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list.setMovement(QtWidgets.QListView.Movement.Static)
        self.list.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list.setIconSize(QtCore.QSize(40, 40))
        self.list.setGridSize(QtCore.QSize(64, 40 + fm.height() + 16))
        self.list.setFixedWidth(width)
        self.list.itemSelectionChanged.connect(self._buttons)
        side.addWidget(self.list, 1)
        row = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_delete.clicked.connect(self.delete)
        done = QtWidgets.QPushButton("Done")
        done.clicked.connect(self.accept)
        row.addWidget(self.btn_delete)
        row.addStretch(1)
        row.addWidget(done)
        side.addLayout(row)
        self.view.boxDrawn.connect(self.on_box)
        self.view.lassoDrawn.connect(self.on_lasso)
        self.view.pickClicked.connect(self.select_pick)
        fit_to_screen(self, 980, 660)
        self.refresh()
        self._show_preview()

    def set_mode(self, mode: str):
        self.view.mode = mode
        self.mode_buttons[mode].setChecked(True)
        self.hint.setText("Drag a box around one character. Scroll to zoom." if mode == "box" else
                          "Draw an outline around one character; it closes by itself. Scroll to zoom.")

    def _picked(self, x0, y0, x1, y1, mask, edge, poly):
        H, W = self.ink.shape
        self.region, self.mask, self.edge, self.poly = self.ink[y0:y1, x0:x1], mask, edge, poly
        self.rect = (x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H)
        inside = self.region[mask] if mask is not None else self.region
        self.chk_flip.blockSignals(True)
        self.chk_flip.setChecked(bool(inside.size and inside.mean() > 0.55))   # mostly black: white strokes
        self.chk_flip.blockSignals(False)
        self._show_preview()
        self.name.setFocus()

    def on_box(self, r: QRectF):
        H, W = self.ink.shape
        x0, y0 = max(0, int(r.left())), max(0, int(r.top()))
        x1, y1 = min(W, int(math.ceil(r.right()))), min(H, int(math.ceil(r.bottom())))
        self._picked(x0, y0, x1, y1, None, None, None)

    def on_lasso(self, pts: np.ndarray):
        H, W = self.ink.shape
        x0, y0 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
        x1, y1 = min(W, int(math.ceil(pts[:, 0].max())) + 1), min(H, int(math.ceil(pts[:, 1].max())) + 1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        local = np.round(pts - (x0, y0)).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [local], 1)
        edge = np.zeros_like(mask)
        cv2.polylines(edge, [local], True, 1, 3)
        self._picked(x0, y0, x1, y1, mask > 0, edge > 0, pts / (W, H))

    def glyph(self):
        if self.region is None:
            return None
        g = ~self.region if self.chk_flip.isChecked() else self.region
        if self.mask is not None:
            g = g & self.mask                          # only what is inside the outline
        return tidy_glyph(g, self.chk_edge.isChecked(), self.edge)

    def _show_preview(self, *_):
        g = self.glyph()
        if g is None:
            self.preview.setPixmap(QtGui.QPixmap())
            self.preview.setText("" if self.region is None else "Nothing left in there")
        else:
            img = ink_to_qimage(g).scaled(self.preview.size() * self.devicePixelRatioF(),
                                          Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            pm = QtGui.QPixmap.fromImage(img)
            pm.setDevicePixelRatio(self.devicePixelRatioF())
            self.preview.setPixmap(pm)
        self._buttons()

    def _buttons(self):
        self.btn_save.setEnabled(self.glyph() is not None and bool(self.name.text().strip()))
        self.btn_delete.setEnabled(bool(self.list.selectedItems()))

    def save(self):
        g, name = self.glyph(), self.name.text().strip()
        if g is None:
            return
        if not name:
            self.name.setFocus()
            return
        save_character(g, name, self.parent_id, self.parent_name, self.rect, self.poly)
        self.region = self.mask = self.edge = self.rect = self.poly = None
        self.view.clear_box()
        self.name.clear()
        self.hint.setText(f"Saved “{name}”. Drag a box around the next one.")
        self.refresh()
        self._show_preview()

    def refresh(self):
        self.list.clear()
        mine = [e for e in character_entries() if e.get("parent_id") == self.parent_id]
        for e in mine:
            item = QtWidgets.QListWidgetItem(glyph_icon(e["dir"] / "glyph.png", 80), e.get("name", ""))
            item.setData(Qt.ItemDataRole.UserRole, str(e["dir"]))
            self.list.addItem(item)
        self.view.show_picks([(e["dir"].name, e.get("name", ""), e.get("rect", (0, 0, 0, 0)), e.get("poly"))
                              for e in mine])
        self._buttons()

    def select_pick(self, cid: str):
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setSelected(Path(item.data(Qt.ItemDataRole.UserRole)).name == cid)

    def delete(self):
        items = self.list.selectedItems()
        if not items:
            return
        what = f"“{items[0].text()}”" if len(items) == 1 else f"{len(items)} characters"
        if QtWidgets.QMessageBox.question(self, "Delete characters", f"Delete {what}? This can't be undone.") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for item in items:
            shutil.rmtree(item.data(Qt.ItemDataRole.UserRole), ignore_errors=True)
        self.refresh()


class ResizeHandle(QtWidgets.QGraphicsRectItem):
    """A handle on a character's box: 0-3 are corners (top-left, top-right, bottom-right,
    bottom-left), 4-7 the middles of the sides (top, right, bottom, left). What dragging one
    does depends on the build page's mode: scale, squish or crop."""
    SHAPES = [(-5, -5, 10, 10)] * 4 + [(-7, -3.5, 14, 7), (-3.5, -7, 7, 14)] * 2
    CURSORS = [Qt.CursorShape.SizeFDiagCursor, Qt.CursorShape.SizeBDiagCursor] * 2 + \
              [Qt.CursorShape.SizeVerCursor, Qt.CursorShape.SizeHorCursor] * 2

    def __init__(self, box, which: int):
        super().__init__(*self.SHAPES[which], box)
        self.box, self.which = box, which
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setBrush(QtGui.QColor("#ffffff"))
        self.setPen(QtGui.QPen(QtGui.QColor(CINNABAR), 1.5))
        self.setCursor(self.CURSORS[which])
        self.setZValue(10)
        self.start = None

    def mousePressEvent(self, e):
        self.start = self.box.begin_handle()
        e.accept()

    def mouseMoveEvent(self, e):
        if self.start is not None:
            self.box.drag_handle(self.which, e.scenePos(), self.start, e.modifiers())

    def mouseReleaseEvent(self, e):
        self.start = None


class CharBox(QtWidgets.QGraphicsRectItem):
    """The box around one placed character's ink. Drag it to move the character; its handles
    scale, squish or crop it, depending on the build page's mode."""
    EDGES = {0: "lt", 1: "rt", 2: "rb", 3: "lb", 4: "t", 5: "r", 6: "b", 7: "l"}

    def __init__(self, composer, p: Placed):
        super().__init__()
        self.composer, self.p = composer, p
        F = QtWidgets.QGraphicsItem.GraphicsItemFlag
        self.setFlags(F.ItemIsMovable | F.ItemIsSelectable | F.ItemSendsGeometryChanges)
        self.setBrush(QtGui.QBrush(Qt.GlobalColor.transparent))
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setZValue(2)
        self.handles = [ResizeHandle(self, i) for i in range(8)]
        self.base_z = 2.0
        self._silent = False
        self._pos = QPointF()
        self.sync()

    def sync(self):
        """Move and size the box to match its character."""
        x, y, w, h = char_rect(self.p)
        self._silent = True
        self.setPos(x, y)
        self.setRect(0, 0, w, h)
        self._silent = False
        self._pos = QPointF(x, y)
        self.decorate()

    def decorate(self):
        sel, mode = self.isSelected(), self.composer.mode
        w, h = self.rect().width(), self.rect().height()
        spots = [(0, 0), (w, 0), (w, h), (0, h), (w / 2, 0), (w, h / 2), (w / 2, h), (0, h / 2)]
        for i, (handle, (x, y)) in enumerate(zip(self.handles, spots)):
            handle.setPos(x, y)
            handle.setVisible(sel and (i < 4 or mode != "scale"))
        pen = QtGui.QPen(QtGui.QColor(CINNABAR if sel else C["muted"]), 1.5 if sel else 1,
                         Qt.PenStyle.DashLine if sel else Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(self.base_z + (1 if sel else 0))

    def begin_handle(self) -> dict:
        p = self.p
        return {"rect": QRectF(self.pos(), self.rect().size()), "scale": p.scale, "sq": (p.sq_x, p.sq_y),
                "use_sq": p.use_squish, "cx": p.cx, "cy": p.cy,
                "crop": None if p.crop is None else list(p.crop), "use_crop": p.use_crop}

    def drag_handle(self, which: int, pos: QPointF, st: dict, mods):
        p, r0, mode = self.p, st["rect"], self.composer.mode
        if which < 4:                                 # a corner: the opposite corner stays put
            anchor = [r0.bottomRight(), r0.bottomLeft(), r0.topLeft(), r0.topRight()][which]
            dx, dy = pos.x() - anchor.x(), pos.y() - anchor.y()
            sx, sy = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
            if mode == "scale" or (mode == "squish" and mods & Qt.KeyboardModifier.ShiftModifier):
                f = max(abs(dx) / max(r0.width(), 1e-6), abs(dy) / max(r0.height(), 1e-6), 0.05)
                w, h = r0.width() * f, r0.height() * f
            else:
                w, h = max(0.5, abs(dx)), max(0.5, abs(dy))
            r = QRectF(anchor.x() if sx > 0 else anchor.x() - w, anchor.y() if sy > 0 else anchor.y() - h, w, h)
        else:                                         # a side: only that side moves
            r = QRectF(r0)
            if which == 4:
                r.setTop(min(pos.y(), r0.bottom() - 0.5))
            elif which == 5:
                r.setRight(max(pos.x(), r0.left() + 0.5))
            elif which == 6:
                r.setBottom(max(pos.y(), r0.top() + 0.5))
            else:
                r.setLeft(min(pos.x(), r0.right() - 0.5))
        p.scale, (p.sq_x, p.sq_y), p.use_squish, p.cx, p.cy = st["scale"], st["sq"], st["use_sq"], st["cx"], st["cy"]
        p.crop, p.use_crop = (None if st["crop"] is None else list(st["crop"])), st["use_crop"]
        if mode == "crop":                            # moved edges become a crop of the picture
            if p.crop is None or not p.use_crop:
                p.crop = [*to_src(p, r0.left(), r0.top()), *to_src(p, r0.right(), r0.bottom())]
            x0, y0 = to_src(p, r.left(), r.top())
            x1, y1 = to_src(p, r.right(), r.bottom())
            edges, c = self.EDGES[which], p.crop
            if "l" in edges:
                c[0] = min(x0, c[2] - 2)
            if "r" in edges:
                c[2] = max(x1, c[0] + 2)
            if "t" in edges:
                c[1] = min(y0, c[3] - 2)
            if "b" in edges:
                c[3] = max(y1, c[1] + 2)
            p.use_crop = True
            p.cache = None
        else:
            if mode == "scale":
                p.scale = st["scale"] * r.width() / max(r0.width(), 1e-6)
            else:
                sx0, sy0 = st["sq"] if st["use_sq"] else (1.0, 1.0)
                p.sq_x = sx0 * r.width() / max(r0.width(), 1e-6)
                p.sq_y = sy0 * r.height() / max(r0.height(), 1e-6)
                p.use_squish = True
            nx, ny, _, _ = char_rect(p)               # keep the fixed corner or side where it was
            p.cx += r.left() - nx
            p.cy += r.top() - ny
        self.sync()
        self.composer.box_changed(self)

    def ink_near(self, pos: QPointF) -> bool:
        """Whether this character has ink under a point (within a few screen pixels). Its paper
        is see-through, like a PNG, so clicks there reach the character underneath."""
        x, y, w, h = char_rect(self.p)
        if not (x <= pos.x() <= x + w and y <= pos.y() <= y + h):
            return False
        if self.p.invert:                             # an inverted character is solid
            return True
        img = process_char(self.p)[0]
        Hi, Wi = img.shape
        views = self.scene().views() if self.scene() else []
        tol = 4 / views[0].transform().m11() if views else 0.3
        u, v = (pos.x() - x) / max(w, 1e-6) * Wi, (pos.y() - y) / max(h, 1e-6) * Hi
        ru, rv = max(1.0, tol / max(w, 1e-6) * Wi), max(1.0, tol / max(h, 1e-6) * Hi)
        return bool(img[int(max(0, v - rv)):int(min(Hi, v + rv + 1)), int(max(0, u - ru)):int(min(Wi, u + ru + 1))].any())

    def _beneath_has_ink(self, pos: QPointF) -> bool:
        return any(isinstance(it, CharBox) and it is not self and it.ink_near(pos)
                   for it in self.scene().items(pos))

    def mousePressEvent(self, e):
        if not self.ink_near(e.scenePos()) and self._beneath_has_ink(e.scenePos()):
            e.ignore()                                # let it through to the character below
            return
        super().mousePressEvent(e)

    def contextMenuEvent(self, e):
        if not self.ink_near(e.scenePos()) and self._beneath_has_ink(e.scenePos()):
            e.ignore()
            return
        self.setSelected(True)
        menu = QtWidgets.QMenu()
        front = menu.addAction("Bring to front")
        back = menu.addAction("Send to back")
        menu.addSeparator()
        edit = menu.addAction("Edit character…")
        remove = menu.addAction("Remove")
        chosen = menu.exec(e.screenPos())
        if chosen is front or chosen is back:
            self.composer.restack(self, chosen is front)
        elif chosen is edit:
            self.composer.edit_selected()
        elif chosen is remove:
            self.composer.remove_selected()

    def mouseDoubleClickEvent(self, e):
        self.setSelected(True)
        self.composer.edit_selected()
        e.accept()

    def itemChange(self, change, value):
        GC = QtWidgets.QGraphicsItem.GraphicsItemChange
        if change == GC.ItemPositionHasChanged and not self._silent:
            delta = self.pos() - self._pos
            self._pos = QPointF(self.pos())
            self.p.cx += delta.x()
            self.p.cy += delta.y()
            self.composer.box_changed(self)
        elif change == GC.ItemSelectedHasChanged:
            if self.isSelected():                     # one character at a time
                for other in self.composer.boxes:
                    if other is not self and other.isSelected():
                        other.setSelected(False)
            self.decorate()
        return super().itemChange(change, value)


class BuildView(QtWidgets.QGraphicsView):
    """The seal being built, in millimetres. Scroll to zoom, drag with the middle button to pan."""

    def __init__(self, composer):
        super().__init__()
        self.composer = composer
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(260, 260)
        self.pix = self.scene().addPixmap(QtGui.QPixmap())
        self.pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.outline = self.scene().addPath(QtGui.QPainterPath())
        self.outline.setZValue(1)
        self.guides = [self.scene().addLine(QtCore.QLineF()) for _ in range(2)]
        for g in self.guides:
            g.setZValue(1.5)
        self.auto_fit = True
        self.pan = None
        self.retheme()

    def retheme(self):
        self.setBackgroundBrush(QtGui.QColor(C["well"]))
        pen = QtGui.QPen(QtGui.QColor(C["rule"]), 1, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.outline.setPen(pen)
        faint = QtGui.QColor(C["muted"])
        faint.setAlpha(120)
        pen = QtGui.QPen(faint, 1, Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        for g in self.guides:
            g.setPen(pen)

    def set_format(self, b: Build):
        path = QtGui.QPainterPath()
        r = QRectF(0, 0, b.w_mm, b.h_mm)
        if b.shape == "ellipse":
            path.addEllipse(r)
        elif b.corner_mm > 0:
            path.addRoundedRect(r, b.corner_mm, b.corner_mm)
        else:
            path.addRect(r)
        self.outline.setPath(path)
        self.guides[0].setLine(b.w_mm / 2, 0, b.w_mm / 2, b.h_mm)      # a faint cross through the middle
        self.guides[1].setLine(0, b.h_mm / 2, b.w_mm, b.h_mm / 2)
        m = 0.08 * max(b.w_mm, b.h_mm)
        self.scene().setSceneRect(QRectF(-m, -m, b.w_mm + 2 * m, b.h_mm + 2 * m))
        if self.auto_fit:
            self.fit()

    def show_guides(self, on: bool):
        for g in self.guides:
            g.setVisible(on)

    def set_image(self, img: QtGui.QImage, px_per_mm: float):
        self.pix.setPixmap(QtGui.QPixmap.fromImage(img))
        self.pix.setScale(1.0 / px_per_mm)

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.auto_fit = True
        if hasattr(self.composer, "timer"):
            self.composer.schedule()

    def zoom(self, f: float):
        vr, sr = self.viewport().rect(), self.sceneRect()
        if self.transform().m11() * f <= min(vr.width() / sr.width(), vr.height() / sr.height()):
            self.fit()
            return
        if self.transform().m11() * f > 400:
            return
        self.scale(f, f)
        self.auto_fit = False
        self.composer.schedule()                      # sharper picture at the new zoom

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.auto_fit:
            self.fit()

    def wheelEvent(self, e):
        self.zoom(1.15 ** (e.angleDelta().y() / 120.0))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton:
            self.pan = e.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.pan is not None:
            d = e.position() - self.pan
            self.pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.MiddleButton and self.pan is not None:
            self.pan = None
            self.viewport().unsetCursor()
            return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        step = 1.0 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.1
        moves = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0),
                 Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.composer.remove_selected()
        elif e.key() in moves:
            self.composer.nudge(*moves[e.key()])
        elif e.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom(1.25)
        elif e.key() == Qt.Key.Key_Minus:
            self.zoom(0.8)
        elif e.key() == Qt.Key.Key_0:
            self.fit()
        else:
            super().keyPressEvent(e)


def section_label(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("section")
    return lab


def muted_label(text: str = "") -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("muted")
    lab.setWordWrap(True)
    return lab


def fit_to_screen(w: QtWidgets.QWidget, width: int, height: int):
    screen = w.screen() or QtWidgets.QApplication.primaryScreen()
    geo = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1280, 800)
    w.resize(min(width, int(geo.width() * 0.94)), min(height, int(geo.height() * 0.9)))


def side_panel(width: int) -> tuple[QtWidgets.QScrollArea, QtWidgets.QVBoxLayout]:
    inner = QtWidgets.QWidget()
    inner.setObjectName("sideinner")
    v = QtWidgets.QVBoxLayout(inner)
    v.setContentsMargins(2, 0, 10, 0)
    v.setSpacing(7)
    scroll = QtWidgets.QScrollArea()
    scroll.setObjectName("sidescroll")
    scroll.setWidget(inner)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.viewport().setAutoFillBackground(False)
    scroll.setFixedWidth(width)
    return scroll, v


def slider(lo: int, hi: int, tip: str = "") -> QtWidgets.QSlider:
    s = QtWidgets.QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setToolTip(tip)
    return s


def mm_box(lo: float, hi: float, step: float = 0.5) -> QtWidgets.QDoubleSpinBox:
    b = QtWidgets.QDoubleSpinBox()
    b.setRange(lo, hi)
    b.setDecimals(1)
    b.setSingleStep(step)
    b.setSuffix(" mm")
    return b


class EditorView(QtWidgets.QGraphicsView):
    """One character, large, in millimetres. Draw along a stroke to select it, drag the
    selection to pull or bend it, right-click to clear it, scroll to zoom."""

    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QtGui.QColor(C["well"]))
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(300, 300)
        self.pix = self.scene().addPixmap(QtGui.QPixmap())
        self.pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.overlay = self.scene().addPixmap(QtGui.QPixmap())
        self.overlay.setZValue(1)
        self.bands = []
        for _ in range(2):
            band = self.scene().addRect(QRectF(), QtGui.QPen(Qt.PenStyle.NoPen),
                                        QtGui.QBrush(QtGui.QColor(216, 69, 47, 40)))
            band.setZValue(1)
            self.bands.append(band)
        pen = QtGui.QPen(QtGui.QColor(CINNABAR), 2)
        pen.setCosmetic(True)
        self.trace = self.scene().addPath(QtGui.QPainterPath(), pen)
        self.trace.setZValue(3)
        pen = QtGui.QPen(QtGui.QColor(CINNABAR), 1, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        self.centre = self.scene().addPath(QtGui.QPainterPath(), pen)
        self.centre.setZValue(2)
        pen = QtGui.QPen(QtGui.QColor(CINNABAR), 1.5)
        pen.setCosmetic(True)
        self.brush = self.scene().addEllipse(QRectF(), pen)
        self.brush.setZValue(4)
        self.brush.hide()
        self.setMouseTracking(True)
        self.auto_fit = True
        self.pressed = False

    def fit(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.auto_fit = True

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.auto_fit:
            self.fit()

    def wheelEvent(self, e):
        f = 1.15 ** (e.angleDelta().y() / 120.0)
        vr, sr = self.viewport().rect(), self.sceneRect()
        if self.transform().m11() * f <= min(vr.width() / sr.width(), vr.height() / sr.height()):
            self.fit()
            return
        self.scale(f, f)
        self.auto_fit = False

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.editor.clear_selection()
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        self.pressed = True
        self.editor.press(self.mapToScene(e.position().toPoint()))

    def mouseMoveEvent(self, e):
        pt = self.mapToScene(e.position().toPoint())
        self.editor.hover(pt)
        if self.pressed:
            self.editor.move(pt)
        else:
            super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        self.brush.hide()
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.pressed and e.button() == Qt.MouseButton.LeftButton:
            self.pressed = False
            self.editor.release(self.mapToScene(e.position().toPoint()))
        else:
            super().mouseReleaseEvent(e)


def _slider_row(s: QtWidgets.QSlider, label: QtWidgets.QLabel) -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    row.addWidget(s, 1)
    label.setObjectName("muted")
    label.setMinimumWidth(label.fontMetrics().horizontalAdvance("200%") + 4)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(label)
    return row


def section_with_original(title: str, tip: str) -> tuple[QtWidgets.QHBoxLayout, QtWidgets.QCheckBox]:
    """A section heading with an "Original" switch on the right."""
    row = QtWidgets.QHBoxLayout()
    row.addWidget(section_label(title))
    row.addStretch(1)
    box = QtWidgets.QCheckBox("Original")
    box.setToolTip(tip)
    row.addWidget(box, 0, Qt.AlignmentFlag.AlignBottom)
    return row, box


class CharacterEditor(QtWidgets.QDialog):
    """Edit one character of a build: squish it, reshape and lengthen single strokes, change its
    weight. Every function has an "Original" switch that shows the character without it.
    Nothing in the build changes until Done."""

    def __init__(self, parent, p: Placed):
        super().__init__(parent)
        self.setWindowTitle(f"Edit character “{p.name}”")
        self.orig = p
        self.work = replace(p, ops=json.loads(json.dumps(p.ops)), crop=None if p.crop is None else list(p.crop),
                            cache=None)
        self.pad = edit_pad(self.work)
        self.canvas = edit_canvas(self.work)
        self.sel = self.centre = self.sel_path = None
        self.len_op = self.len_base = None
        self.history: list[tuple] = []
        self.drag = self.grab_at = self.pending = None
        self.path: list = []
        self._loading = False
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._preview)
        self.len_timer = QtCore.QTimer(self)
        self.len_timer.setSingleShot(True)
        self.len_timer.setInterval(40)
        self.len_timer.timeout.connect(self._apply_lengthen)

        self.tool = "strokes"
        self.erase_pts: list = []
        self.erase_live = None
        outer = QtWidgets.QHBoxLayout(self)
        outer.setSpacing(12)
        left = QtWidgets.QVBoxLayout()
        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(4)
        tools.addWidget(QtWidgets.QLabel("Tool"))
        self.tool_buttons = {}
        for key, label, tip in (("strokes", "Strokes", "Draw along a stroke to select it, then drag it"),
                                ("erase", "Eraser", "Paint over ink to erase it")):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.set_tool(k))
            tools.addWidget(b)
            self.tool_buttons[key] = b
        self.tool_buttons["strokes"].setChecked(True)
        tools.addSpacing(12)
        tools.addWidget(QtWidgets.QLabel("Eraser size"))
        self.sld_brush = slider(1, 20, "How big the eraser is")
        self.sld_brush.setValue(4)
        self.sld_brush.setMaximumWidth(160)
        tools.addWidget(self.sld_brush)
        tools.addStretch(1)
        left.addLayout(tools)
        self.view = EditorView(self)
        left.addWidget(self.view, 1)
        self.hint = muted_label()
        left.addWidget(self.hint)
        outer.addLayout(left, 1)

        M = self.fontMetrics().horizontalAdvance("M")
        side, sv = side_panel(max(260, M * 20))
        row, self.orig_squish = section_with_original("Squish", "Show the character without the squish")
        sv.addLayout(row)
        f = QtWidgets.QFormLayout()
        f.setHorizontalSpacing(10)
        self.sld_sq_h = slider(30, 200, "Left makes the character narrower, right makes it wider")
        self.sld_sq_v = slider(30, 200, "Left makes the character shorter, right makes it taller")
        self.lbl_sq_h, self.lbl_sq_v = QtWidgets.QLabel(), QtWidgets.QLabel()
        f.addRow("Horizontal", _slider_row(self.sld_sq_h, self.lbl_sq_h))
        f.addRow("Vertical", _slider_row(self.sld_sq_v, self.lbl_sq_v))
        sv.addLayout(f)

        row, self.orig_strokes = section_with_original("Strokes", "Show the strokes without the ones you pulled or bent")
        sv.addLayout(row)
        sv.addWidget(muted_label("1. Draw along a stroke, or along just the part you want to change.\n"
                                 "2. Drag any point of the red selection: along the stroke to stretch it, sideways "
                                 "to bend it. The far end of the selection stays put.\n"
                                 "Right-click clears the selection."))
        row = QtWidgets.QHBoxLayout()
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_undo.setToolTip("Undo the last change to the strokes (Ctrl+Z)")
        self.btn_restart = QtWidgets.QPushButton("Start over")
        self.btn_restart.setToolTip("Put every stroke back the way it was")
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_restart)
        sv.addLayout(row)

        row, self.orig_lengthen = section_with_original("Lengthen", "Show the strokes without the lengthening")
        sv.addLayout(row)
        sv.addWidget(muted_label("Makes the selected stroke longer at its free end. Its curve, thickness and tip "
                                 "stay the same."))
        self.sld_len = slider(0, 100, "How much longer the selected stroke gets")
        self.lbl_len = QtWidgets.QLabel()
        sv.addLayout(_slider_row(self.sld_len, self.lbl_len))

        row, self.orig_erase = section_with_original("Eraser", "Show the character without what you erased")
        sv.addLayout(row)
        sv.addWidget(muted_label("Pick Eraser above the character, then paint over ink to erase it."))

        row, self.orig_weight = section_with_original("Weight", "Show the strokes at their original weight")
        sv.addLayout(row)
        self.sld_weight = slider(-5, 5, "Left for thinner strokes, right for bolder")
        sv.addWidget(self.sld_weight)
        self.chk_invert = QtWidgets.QCheckBox("Invert this character")
        sv.addWidget(self.chk_invert)
        sv.addStretch(1)
        buttons = QtWidgets.QHBoxLayout()
        cancel = QtWidgets.QPushButton("Cancel")
        done = QtWidgets.QPushButton("Done")
        done.setObjectName("primary")
        cancel.clicked.connect(self.reject)
        done.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(done)
        col = QtWidgets.QVBoxLayout()
        col.addWidget(side, 1)
        col.addLayout(buttons)
        outer.addLayout(col)

        w = self.work
        self._loading = True
        self.sld_sq_h.setValue(round(min(200, max(30, w.sq_x * 100))))
        self.sld_sq_v.setValue(round(min(200, max(30, w.sq_y * 100))))
        self.sld_weight.setValue(w.weight)
        self.chk_invert.setChecked(w.invert)
        self.orig_squish.setChecked(not w.use_squish)
        self.orig_strokes.setChecked(not w.use_strokes)
        self.orig_lengthen.setChecked(not w.use_lengthen)
        self.orig_erase.setChecked(not w.use_erase)
        self.orig_weight.setChecked(not w.use_weight)
        self._loading = False
        for s in (self.sld_sq_h, self.sld_sq_v):
            s.valueChanged.connect(self.on_squish)
        self.sld_weight.valueChanged.connect(self.on_weight)
        self.chk_invert.toggled.connect(self.on_weight)
        self.sld_len.valueChanged.connect(self.on_lengthen)
        self.orig_squish.toggled.connect(self.on_originals)
        self.orig_weight.toggled.connect(self.on_originals)
        self.orig_strokes.toggled.connect(self.on_stroke_originals)
        self.orig_lengthen.toggled.connect(self.on_stroke_originals)
        self.orig_erase.toggled.connect(self.on_stroke_originals)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_restart.clicked.connect(self.start_over)
        undo = QtGui.QShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Undo), self)
        undo.activated.connect(self.undo)
        fit_to_screen(self, 1040, 720)
        self.update_controls()
        self.on_squish()

    # ---- whole-character settings and the Original switches
    def editable(self) -> bool:
        return self.work.use_strokes and self.work.use_lengthen and self.work.use_erase

    def set_tool(self, tool: str):
        self.finish_lengthen()
        self.tool = tool
        self.tool_buttons[tool].setChecked(True)
        if tool == "erase":
            self.sel = self.centre = self.sel_path = None
        self.view.brush.setVisible(False)
        self.update_controls()
        self.refresh()

    def brush_radius(self) -> float:
        """The eraser's radius in the character's pixels."""
        return self.sld_brush.value() / 100 * max(self.work.src.shape)

    def hover(self, pt):
        if self.tool != "erase" or not self.editable():
            self.view.brush.hide()
            return
        sx, sy = char_scale(self.work)
        r = self.brush_radius()
        self.view.brush.setRect(QRectF(pt.x() - r * sx, pt.y() - r * sy, 2 * r * sx, 2 * r * sy))
        self.view.brush.show()

    def update_controls(self):
        w = self.work
        for s in (self.sld_sq_h, self.sld_sq_v):
            s.setEnabled(w.use_squish)
        self.sld_weight.setEnabled(w.use_weight)
        self.btn_undo.setEnabled(self.editable() and bool(self.history))
        self.btn_restart.setEnabled(self.editable() and bool(w.ops))
        self.sld_len.setEnabled(self.editable() and self.sel is not None and self.tool == "strokes")
        self.lbl_len.setText(f"{self.sld_len.value()}%")
        self.sld_brush.setEnabled(self.tool == "erase" and self.editable())
        if not self.editable():
            self.hint.setText("Turn off Original under Strokes, Lengthen and Eraser to change strokes.")
        elif self.tool == "erase":
            self.hint.setText("Paint over ink to erase it. Ctrl+Z undoes.")
        elif self.sel is None and self.drag is None:
            self.hint.setText("Draw along a stroke, or along just the part of it you want to change, to select it.")

    def on_squish(self, *_):
        self.lbl_sq_h.setText(f"{self.sld_sq_h.value()}%")
        self.lbl_sq_v.setText(f"{self.sld_sq_v.value()}%")
        if not self._loading:
            self.work.sq_x, self.work.sq_y = self.sld_sq_h.value() / 100, self.sld_sq_v.value() / 100
        self.refresh(fit=True)

    def on_weight(self, *_):
        self.work.weight = self.sld_weight.value()
        self.work.invert = self.chk_invert.isChecked()
        self.refresh()

    def on_originals(self, *_):
        self.work.use_squish = not self.orig_squish.isChecked()
        self.work.use_weight = not self.orig_weight.isChecked()
        self.update_controls()
        self.refresh(fit=True)

    def on_stroke_originals(self, *_):
        self.finish_lengthen()
        self.work.use_strokes = not self.orig_strokes.isChecked()
        self.work.use_lengthen = not self.orig_lengthen.isChecked()
        self.work.use_erase = not self.orig_erase.isChecked()
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.canvas = edit_canvas(self.work)       # replay only the edits that are switched on
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.clear_selection()

    # ---- drawing
    def to_canvas(self, pt) -> np.ndarray:
        sx, sy = char_scale(self.work)
        return np.array([pt.x() / sx, pt.y() / sy], dtype=np.float64)

    def refresh(self, canvas=None, sel=None, centre=None, fit=False):
        w = self.work
        g = self.canvas if canvas is None else canvas
        if canvas is None:
            sel, centre = (self.sel, self.centre) if self.tool == "strokes" else (None, None)
        shown = thicken(g, w.weight, max(w.src.shape)) if w.use_weight else g
        H, W = shown.shape
        sx, sy = char_scale(w)
        tf = QtGui.QTransform.fromScale(sx, sy)
        v = self.view
        v.pix.setPixmap(QtGui.QPixmap.fromImage(ink_to_qimage(~shown if w.invert else shown)))
        v.pix.setTransform(tf)
        if sel is not None:
            rgba = np.zeros((H, W, 4), np.uint8)
            rgba[sel] = (216, 69, 47, 130)
            img = QtGui.QImage(rgba.data, W, H, 4 * W, QtGui.QImage.Format.Format_RGBA8888).copy()
            v.overlay.setPixmap(QtGui.QPixmap.fromImage(img))
            v.overlay.setTransform(tf)
        else:
            v.overlay.setPixmap(QtGui.QPixmap())
        path = QtGui.QPainterPath()
        if centre is not None and len(centre) > 1:
            path.moveTo(centre[0][0] * sx, centre[0][1] * sy)
            for x, y in centre[1:]:
                path.lineTo(x * sx, y * sy)
        v.centre.setPath(path)
        for band in v.bands:
            band.hide()
        m = 0.04 * max(W * sx, H * sy)
        v.scene().setSceneRect(QRectF(-m, -m, W * sx + 2 * m, H * sy + 2 * m))
        if fit and v.auto_fit:
            v.fit()

    def select_along(self, path) -> bool:
        """Select the stroke along a path, the same way replaying the edits will."""
        found = select_stroke(self.canvas, np.asarray(path, np.float64))
        if found is None:
            return False
        self.sel, self.centre = found
        self.sel_path = np.asarray(path, np.float64)
        return True

    def near_selection(self, b) -> bool:
        H, W = self.sel.shape
        r = max(6, round(0.012 * max(H, W)))
        x, y = int(round(b[0])), int(round(b[1]))
        x0, x1, y0, y1 = max(0, x - r), min(W, x + r + 1), max(0, y - r), min(H, y + r + 1)
        return x1 > x0 and y1 > y0 and bool(self.sel[y0:y1, x0:x1].any())

    def press(self, pt):
        if not self.editable():
            self.update_controls()
            return
        self.finish_lengthen()
        b = self.to_canvas(pt)
        if self.tool == "erase":
            self.erase_pts = [b]
            self.erase_live = erase_mask(self.canvas.shape, [b], self.brush_radius())
            self.refresh(self.canvas & ~self.erase_live, None, None)
            return
        if self.sel is not None and self.near_selection(b):
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.drag = StrokeDrag(self.canvas, self.sel, self.centre, b)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self.grab_at = b
            self.hint.setText("Pull along the stroke to stretch it, or sideways to bend it.")
        else:
            self.path = [b]
            p = QtGui.QPainterPath()
            p.moveTo(pt)
            self.view.trace.setPath(p)

    def move(self, pt):
        if self.erase_pts:
            b = self.to_canvas(pt)
            r = max(1, int(round(self.brush_radius())))
            m = self.erase_live.view(np.uint8)
            a = np.round(self.erase_pts[-1]).astype(int)
            cv2.line(m, (int(a[0]), int(a[1])), (int(round(b[0])), int(round(b[1]))), 1, 2 * r + 1)
            cv2.circle(m, (int(round(b[0])), int(round(b[1]))), r, 1, -1)
            self.erase_pts.append(b)
            self.timer.start()
            return
        if self.drag is not None:
            self.pending = self.to_canvas(pt) - self.grab_at
            self.timer.start()
        elif self.path:
            self.path.append(self.to_canvas(pt))
            p = self.view.trace.path()
            p.lineTo(pt)
            self.view.trace.setPath(p)

    def release(self, pt):
        if self.erase_pts:
            self.timer.stop()
            op = {"kind": "erase", "points": (np.asarray(self.erase_pts) - self.pad).round(2).tolist(),
                  "radius": round(self.brush_radius(), 2)}
            self.erase_pts, self.erase_live = [], None
            self.remember()
            self.canvas = apply_op(self.canvas, op, self.pad)[0]
            self.work.ops.append(op)
            self.sel = self.centre = self.sel_path = None
            self.update_controls()
            self.refresh()
            return
        if self.drag is not None:
            d = self.to_canvas(pt) - self.grab_at
            self.timer.stop()
            if np.hypot(*d) >= 0.5:
                op = {"kind": "drag", "path": (self.sel_path - self.pad).tolist(),
                      "grab": (self.grab_at - self.pad).tolist(), "d": [float(d[0]), float(d[1])]}
                self.remember()
                self.canvas, _, centre = apply_op(self.canvas, op, self.pad)
                self.work.ops.append(op)
                if centre is None or not self.select_along(centre):
                    self.sel = self.centre = self.sel_path = None
            self.drag = self.grab_at = self.pending = None
            self.hint.setText("Drag it again, lengthen it, draw along another stroke, or right-click to clear it.")
            self.update_controls()
            self.refresh()
            return
        path, self.path = self.path, []
        self.view.trace.setPath(QtGui.QPainterPath())
        if len(path) > 2 and self.select_along(path):
            self.hint.setText("Drag any point of the red stroke, or use Lengthen. Right-click clears the selection.")
        else:
            self.hint.setText("No stroke found there. Draw along the middle of a stroke.")
        self._loading = True
        self.sld_len.setValue(0)
        self._loading = False
        self.update_controls()
        self.refresh()

    def _preview(self):
        if self.erase_live is not None:
            self.refresh(self.canvas & ~self.erase_live, None, None)
        elif self.drag is not None and self.pending is not None:
            self.refresh(*self.drag.apply(self.pending))

    def clear_selection(self):
        self.finish_lengthen()
        self.sel = self.centre = self.sel_path = None
        self.update_controls()
        self.refresh()

    # ---- lengthening the selected stroke
    def on_lengthen(self, v):
        self.lbl_len.setText(f"{v}%")
        if self._loading or self.sel is None or not self.editable():
            return
        if self.len_op is None:
            if v <= 0:
                return
            self.remember()
            self.len_base = (self.canvas.copy(), self.sel_path.copy())
            self.len_op = {"kind": "lengthen", "path": (self.sel_path - self.pad).tolist(), "amount": 0.0,
                           "end": free_end(self.canvas, self.sel, self.centre)}
            self.work.ops.append(self.len_op)
        self.len_op["amount"] = v / 100
        self.len_timer.start()

    def _apply_lengthen(self):
        if self.len_op is None:
            return
        base, path = self.len_base
        if self.len_op["amount"] <= 0:              # slid back to nothing
            self.canvas = base
            self.select_along(path)
        else:
            self.canvas, sel, centre = apply_op(base, self.len_op, self.pad)
            if sel is not None:
                self.sel, self.centre = sel, centre
        self.update_controls()
        self.refresh()

    def finish_lengthen(self):
        """Keep the lengthening as it is; the longer stroke becomes the selection."""
        if self.len_op is None:
            return
        self.len_timer.stop()
        self._apply_lengthen()
        if self.len_op["amount"] <= 0:
            self.work.ops = [op for op in self.work.ops if op is not self.len_op]
            if self.history:
                self.history.pop()
        elif self.centre is not None and not self.select_along(self.centre):
            self.sel = self.centre = self.sel_path = None
        self.len_op = self.len_base = None
        self._loading = True
        self.sld_len.setValue(0)
        self._loading = False

    # ---- undo
    def remember(self):
        self.history.append((json.loads(json.dumps(self.work.ops)), self.canvas.copy(),
                             None if self.sel is None else self.sel.copy(),
                             None if self.centre is None else self.centre.copy(),
                             None if self.sel_path is None else self.sel_path.copy()))
        self.update_controls()

    def undo(self):
        if not self.history or not self.editable():
            return
        self.len_timer.stop()
        self.len_op = self.len_base = None
        self.work.ops, self.canvas, self.sel, self.centre, self.sel_path = self.history.pop()
        self._loading = True
        self.sld_len.setValue(0)
        self._loading = False
        self.update_controls()
        self.refresh()

    def start_over(self):
        if not self.editable():
            return
        self.finish_lengthen()
        self.remember()
        self.work.ops = []
        self.canvas = edit_canvas(self.work)
        self.clear_selection()

    def accept(self):
        self.finish_lengthen()
        p, w = self.orig, self.work
        for k in ("sq_x", "sq_y", "use_squish", "weight", "use_weight", "ops", "use_strokes", "use_lengthen",
                  "invert"):
            setattr(p, k, getattr(w, k))
        p.cache = None
        super().accept()


class Composer(QtWidgets.QWidget):
    """Build a seal out of saved characters."""
    saved = QtCore.Signal(dict, bool)          # library entry, and whether to add it to the sheet
    cancelled = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.build = Build()
        self.lib_id: str | None = None         # set once saved, so saving again updates it
        self.auto_layout = True                # arrange automatically until a box is moved by hand
        self.name_typed = False
        self.boxes: list[CharBox] = []
        self.mode = "scale"                    # what the handles do: scale, squish or crop
        self._loading = False
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self.render)
        fm = self.fontMetrics()
        M = fm.horizontalAdvance("M")

        # left: format, then the characters to choose from
        left, lv = side_panel(max(250, M * 20))
        lv.addWidget(section_label("Format"))
        f = QtWidgets.QFormLayout()
        f.setHorizontalSpacing(10)
        self.cmb_format = QtWidgets.QComboBox()
        for name, *_ in FORMATS:
            self.cmb_format.addItem(name)
        self.cmb_format.addItem("Custom")
        self.cmb_shape = QtWidgets.QComboBox()
        self.cmb_shape.addItem("Rectangle", "rect")
        self.cmb_shape.addItem("Ellipse", "ellipse")
        self.spn_w, self.spn_h = mm_box(5, 190), mm_box(5, 190)
        self.spn_frame = mm_box(0, 6, 0.1)
        self.spn_frame.setToolTip("Thickness of the frame line. 0 for no frame.")
        corners = QtWidgets.QHBoxLayout()
        self.chk_round = QtWidgets.QCheckBox("Round")
        self.chk_round.setToolTip("Round corners for rectangles, with this radius")
        self.spn_corner = mm_box(0.2, 40, 0.2)
        self.spn_corner.setValue(2.0)
        self.spn_corner.setEnabled(False)
        corners.addWidget(self.chk_round)
        corners.addWidget(self.spn_corner, 1)
        for label, w in (("Format", self.cmb_format), ("Shape", self.cmb_shape), ("Width", self.spn_w),
                         ("Height", self.spn_h), ("Frame", self.spn_frame), ("Corners", corners)):
            f.addRow(label, w)
        lv.addLayout(f)
        self.chk_white = QtWidgets.QCheckBox("White text (invert the whole seal)")
        self.chk_white.setToolTip("Bai wen: the ground becomes ink and the characters are cut out of it")
        lv.addWidget(self.chk_white)
        lv.addWidget(section_label("Characters"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search by character or seal")
        self.search.setClearButtonEnabled(True)
        lv.addWidget(self.search)
        self.chars = QtWidgets.QListWidget()
        self.chars.setObjectName("library")
        self.chars.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.chars.setMovement(QtWidgets.QListView.Movement.Static)
        self.chars.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.chars.setIconSize(QtCore.QSize(48, 48))
        self.char_tile = QtCore.QSize(max(76, M * 6), 48 + 2 * fm.height() + 18)
        self.chars.setGridSize(self.char_tile + QtCore.QSize(4, 4))
        self.chars.setMinimumHeight(self.char_tile.height() * 2 + 20)
        self.chars.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        lv.addWidget(self.chars, 1)
        self.chars_empty = muted_label("No characters yet. In the Saved seals tab, select a seal and press "
                                       "Pick characters… to cut some out.")
        lv.addWidget(self.chars_empty)
        self.btn_add_char = QtWidgets.QPushButton("Add to seal")
        self.btn_add_char.setToolTip("Or double-click a character")
        lv.addWidget(self.btn_add_char)

        # middle: the seal itself
        mid = QtWidgets.QVBoxLayout()
        self.view = BuildView(self)
        mid.addWidget(self.view, 1)
        modes = QtWidgets.QHBoxLayout()
        modes.setSpacing(4)
        modes.addWidget(QtWidgets.QLabel("Handles"))
        self.mode_buttons = {}
        for key, label, tip in (("scale", "Scale", "Corners make the character bigger or smaller, keeping its proportions"),
                                ("squish", "Squish", "Corners and sides change its width and height separately"),
                                ("crop", "Crop", "Corners and sides trim the character's edges")):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            modes.addWidget(b)
            self.mode_buttons[key] = b
        self.mode_buttons["scale"].setChecked(True)
        modes.addStretch(1)
        self.chk_guides = QtWidgets.QCheckBox("Guides")
        self.chk_guides.setToolTip("A faint cross through the middle of the seal")
        self.chk_guides.setChecked(True)
        modes.addWidget(self.chk_guides)
        fit = QtWidgets.QPushButton("Fit")
        fit.setToolTip("Show the whole seal again (scroll to zoom, middle-drag to move around)")
        fit.clicked.connect(self.view.fit)
        modes.addWidget(fit)
        mid.addLayout(modes)
        row = QtWidgets.QHBoxLayout()
        self.btn_arrange = QtWidgets.QPushButton("Auto-arrange")
        self.btn_arrange.setToolTip("Lay the characters out in reading order: columns from right to left")
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_clear = QtWidgets.QPushButton("Start over")
        for b in (self.btn_arrange, self.btn_remove, self.btn_clear):
            row.addWidget(b)
        row.addStretch(1)
        mid.addLayout(row)
        mid.addWidget(muted_label("Drag a character to move it and its corners to resize it; characters may "
                                  "overlap. Arrow keys nudge, Delete removes, double-click edits, right-click "
                                  "stacks. Scroll to zoom."))

        # right: the selected character
        right, rv = side_panel(max(240, M * 18))
        rv.addWidget(section_label("Selected character"))
        self.lbl_char = QtWidgets.QLabel("Click a character to adjust it")
        self.lbl_char.setObjectName("muted")
        self.lbl_char.setWordWrap(True)
        rv.addWidget(self.lbl_char)
        self.big_font = QtGui.QFont(self.font())
        self.big_font.setPointSizeF(self.font().pointSizeF() * 2.2)
        f = QtWidgets.QFormLayout()
        f.setHorizontalSpacing(10)
        self.spn_cw, self.spn_ch = mm_box(1, 190, 0.1), mm_box(1, 190, 0.1)
        f.addRow("Width", self.spn_cw)
        f.addRow("Height", self.spn_ch)
        rv.addLayout(f)
        row, self.orig_squish = section_with_original("Squish", "Show this character with its own proportions")
        rv.addLayout(row)
        self.lbl_squish = muted_label()
        rv.addWidget(self.lbl_squish)
        row, self.orig_crop = section_with_original("Crop", "Show this character without the crop")
        rv.addLayout(row)
        self.lbl_crop = muted_label()
        rv.addWidget(self.lbl_crop)
        self.btn_edit_char = QtWidgets.QPushButton("Edit character…")
        self.btn_edit_char.setToolTip("Squish it, reshape or lengthen single strokes, change its weight")
        rv.addWidget(self.btn_edit_char)
        rv.addWidget(muted_label("Squish and crop here with the Handles switch under the seal. The character "
                                 "editor (double-click) does squish, strokes, lengthen, eraser and weight."))
        rv.addStretch(1)
        self.char_controls = (self.spn_cw, self.spn_ch, self.orig_squish, self.orig_crop, self.btn_edit_char,
                              self.btn_remove)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(left)
        body.addLayout(mid, 1)
        body.addWidget(right)
        bottom = QtWidgets.QHBoxLayout()
        self.name = QtWidgets.QLineEdit()
        self.name.setPlaceholderText("Name of this seal")
        self.name.textEdited.connect(lambda _: setattr(self, "name_typed", True))
        self.btn_save = QtWidgets.QPushButton("Save")
        self.btn_save.setToolTip("Save it to the library; it shows up under Saved seals")
        self.btn_save_add = QtWidgets.QPushButton("Save and add to sheet")
        self.btn_save_add.setObjectName("primary")
        cancel = QtWidgets.QPushButton("Cancel")
        bottom.addWidget(QtWidgets.QLabel("Name"))
        bottom.addWidget(self.name, 1)
        bottom.addWidget(cancel)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_save_add)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 12, 10, 10)
        outer.addLayout(body, 1)
        outer.addLayout(bottom)

        self.cmb_format.currentIndexChanged.connect(self.on_format)
        for w in (self.cmb_shape,):
            w.currentIndexChanged.connect(self.on_custom)
        for w in (self.spn_w, self.spn_h):
            w.valueChanged.connect(self.on_custom)
        self.spn_frame.valueChanged.connect(self.on_style)
        self.chk_white.toggled.connect(self.on_style)
        self.chk_round.toggled.connect(self.on_corners)
        self.spn_corner.valueChanged.connect(self.on_corners)
        self.chk_guides.toggled.connect(self.view.show_guides)
        self.search.textChanged.connect(self.filter_chars)
        self.chars.itemDoubleClicked.connect(lambda _: self.add_selected())
        self.chars.itemSelectionChanged.connect(
            lambda: self.btn_add_char.setEnabled(bool(self.chars.selectedItems())))
        self.chars.customContextMenuRequested.connect(self.char_menu)
        self.btn_add_char.clicked.connect(self.add_selected)
        self.btn_arrange.clicked.connect(self.arrange)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear)
        self.view.scene().selectionChanged.connect(self.on_select)
        for w in (self.spn_cw, self.spn_ch):
            w.valueChanged.connect(self.on_char_edit)
        self.orig_squish.toggled.connect(self.on_originals)
        self.orig_crop.toggled.connect(self.on_originals)
        self.btn_edit_char.clicked.connect(self.edit_selected)
        self.btn_save.clicked.connect(lambda: self.save(False))
        self.btn_save_add.clicked.connect(lambda: self.save(True))
        cancel.clicked.connect(self.cancelled.emit)

        self.on_format(0)
        self.spn_frame.setValue(self.build.frame_mm)
        self.refresh_characters()
        self.on_select()

    # ---- the characters to choose from
    def refresh_characters(self):
        self.chars.clear()
        entries = character_entries()
        fm = self.chars.fontMetrics()
        for e in entries:
            name, parent = e.get("name", ""), e.get("parent_name", "")
            item = QtWidgets.QListWidgetItem(glyph_icon(e["dir"] / "glyph.png", 96), name + "\n" +
                                             fm.elidedText(parent, Qt.TextElideMode.ElideRight, self.char_tile.width() - 8))
            item.setToolTip(f"{name}\nfrom {parent}")
            item.setSizeHint(self.char_tile)
            item.setData(Qt.ItemDataRole.UserRole, str(e["dir"]))
            item.setData(Qt.ItemDataRole.UserRole + 1, f"{name}\n{parent}".lower())
            self.chars.addItem(item)
        self.chars.setVisible(bool(entries))
        self.search.setEnabled(bool(entries))
        self.chars_empty.setVisible(not entries)
        self.btn_add_char.setEnabled(False)
        self.filter_chars(self.search.text())

    def filter_chars(self, text):
        text = text.strip().lower()
        for i in range(self.chars.count()):
            item = self.chars.item(i)
            item.setHidden(bool(text) and text not in item.data(Qt.ItemDataRole.UserRole + 1))

    def char_menu(self, pos):
        item = self.chars.itemAt(pos)
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        delete = menu.addAction("Delete character")
        if menu.exec(self.chars.viewport().mapToGlobal(pos)) is delete:
            name = item.text().split("\n")[0]
            if QtWidgets.QMessageBox.question(self, "Delete character", f"Delete “{name}”? Seals already "
                                              f"built with it keep their own copy.") \
                    == QtWidgets.QMessageBox.StandardButton.Yes:
                shutil.rmtree(item.data(Qt.ItemDataRole.UserRole), ignore_errors=True)
                self.refresh_characters()

    def add_selected(self):
        for item in self.chars.selectedItems():
            d = Path(item.data(Qt.ItemDataRole.UserRole))
            try:
                meta = json.loads((d / "char.json").read_text("utf-8"))
                src = png_to_ink(d / "glyph.png")
            except (OSError, ValueError):
                continue
            p = Placed(d.name, meta.get("name", ""), src, cx=self.build.w_mm / 2, cy=self.build.h_mm / 2,
                       scale=0.45 * min(self.build.w_mm, self.build.h_mm) / max(src.shape))
            self.build.chars.append(p)
        if self.auto_layout:
            auto_arrange(self.build)
        self.sync_boxes(select=len(self.build.chars) - 1)
        self.update_name()
        self.schedule()

    # ---- the seal's format
    def on_format(self, index):
        if self._loading:
            return
        if index < len(FORMATS):
            _, shape, w, h = FORMATS[index]
            self._set_format_widgets(shape, w, h)
            self.apply_format(shape, w, h)

    def on_custom(self, *_):
        if self._loading:
            return
        self._loading = True
        self.cmb_format.setCurrentIndex(len(FORMATS))
        self._loading = False
        self.apply_format(self.cmb_shape.currentData(), self.spn_w.value(), self.spn_h.value())

    def _set_format_widgets(self, shape, w, h):
        self._loading = True
        self.cmb_shape.setCurrentIndex(0 if shape == "rect" else 1)
        self.spn_w.setValue(w)
        self.spn_h.setValue(h)
        self._loading = False

    def on_corners(self, *_):
        if self._loading:
            return
        rect = self.build.shape == "rect"
        self.chk_round.setEnabled(rect)
        self.spn_corner.setEnabled(rect and self.chk_round.isChecked())
        self.build.corner_mm = self.spn_corner.value() if (rect and self.chk_round.isChecked()) else 0.0
        self.view.set_format(self.build)
        self.schedule()

    def apply_format(self, shape, w, h):
        self.build.shape = shape
        rect = shape == "rect"
        self.chk_round.setEnabled(rect)
        self.spn_corner.setEnabled(rect and self.chk_round.isChecked())
        self.build.corner_mm = self.spn_corner.value() if (rect and self.chk_round.isChecked()) else 0.0
        scale_build(self.build, w, h)
        if self.auto_layout:
            auto_arrange(self.build)
        self.view.set_format(self.build)
        self.sync_boxes()
        self.schedule()

    def on_style(self, *_):
        if self._loading:
            return
        self.build.frame_mm = self.spn_frame.value()
        self.build.invert_all = self.chk_white.isChecked()
        if self.auto_layout:
            auto_arrange(self.build)
            self.sync_boxes()
        self.schedule()

    def arrange(self):
        self.auto_layout = True
        auto_arrange(self.build)
        self.sync_boxes()
        self.schedule()

    # ---- placed characters
    def sync_boxes(self, select: int | None = None):
        if select is None:
            sel = self.selected()
            select = self.build.chars.index(sel.p) if sel is not None and sel.p in self.build.chars else None
        for b in self.boxes:
            self.view.scene().removeItem(b)
        self.boxes = [CharBox(self, p) for p in self.build.chars]
        order = sorted(range(len(self.boxes)), key=lambda i: self.build.chars[i].z)
        for rank, i in enumerate(order):              # stacking on the canvas follows the seal
            self.boxes[i].base_z = 2.0 + 0.001 * rank
            self.boxes[i].decorate()
        for b in self.boxes:
            self.view.scene().addItem(b)
        if select is not None and 0 <= select < len(self.boxes):
            self.boxes[select].setSelected(True)
        self.on_select()

    def selected(self) -> CharBox | None:
        for b in self.boxes:
            if b.isSelected():
                return b
        return None

    def box_changed(self, box: CharBox):
        self.auto_layout = False               # placed by hand from now on
        if box is self.selected():
            self._fill_char_controls(box.p)
        self.schedule()

    def on_select(self):
        box = self.selected()
        for w in self.char_controls:
            w.setEnabled(box is not None)
        if box is None:
            self.lbl_char.setFont(self.font())
            self.lbl_char.setText("Click a character on the seal to adjust it" if self.boxes else
                                  "Add characters from the list on the left")
            return
        self.lbl_char.setFont(self.big_font)
        self.lbl_char.setText(box.p.name)
        self._fill_char_controls(box.p)

    def _fill_char_controls(self, p: Placed):
        self._loading = True
        _, _, w, h = char_rect(p)
        self.spn_cw.setValue(w)
        self.spn_ch.setValue(h)
        self.orig_squish.setChecked(not p.use_squish)
        self.orig_crop.setChecked(not p.use_crop)
        self.lbl_squish.setText(f"{p.sq_x * 100:.0f}% wide, {p.sq_y * 100:.0f}% tall"
                                if (p.sq_x, p.sq_y) != (1.0, 1.0) else "Not squished")
        self.lbl_crop.setText("Cropped" if p.crop is not None else "Not cropped")
        self._loading = False

    def on_char_edit(self, *_):
        box = self.selected()
        if self._loading or box is None:
            return
        p = box.p
        x, y, w, h = char_rect(p)
        cx, cy = x + w / 2, y + h / 2
        nw, nh = self.spn_cw.value(), self.spn_ch.value()
        if self.mode == "squish":                     # width and height separately
            if not p.use_squish:
                p.sq_x, p.sq_y, p.use_squish = 1.0, 1.0, True
            p.sq_x *= nw / max(w, 1e-6)
            p.sq_y *= nh / max(h, 1e-6)
        else:                                         # one size: the other follows
            f = nw / max(w, 1e-6) if abs(nw - w) > 1e-6 else nh / max(h, 1e-6)
            p.scale *= f
        x, y, w, h = char_rect(p)                     # around the same middle
        p.cx += cx - (x + w / 2)
        p.cy += cy - (y + h / 2)
        self.auto_layout = False
        box.sync()
        self._fill_char_controls(p)
        self.schedule()

    def on_originals(self, *_):
        box = self.selected()
        if self._loading or box is None:
            return
        box.p.use_squish = not self.orig_squish.isChecked()
        box.p.use_crop = not self.orig_crop.isChecked()
        box.p.cache = None
        box.sync()
        self._fill_char_controls(box.p)
        self.schedule()

    def set_mode(self, mode: str):
        self.mode = mode
        self.mode_buttons[mode].setChecked(True)
        for b in self.boxes:
            b.decorate()

    def edit_selected(self):
        box = self.selected()
        if box is None:
            return
        if CharacterEditor(self, box.p).exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.auto_layout = False
            box.sync()
            self._fill_char_controls(box.p)
            self.schedule()

    def restack(self, box: CharBox, front: bool):
        zs = [q.z for q in self.build.chars]
        box.p.z = (max(zs) + 1) if front else (min(zs) - 1)
        self.sync_boxes(select=self.build.chars.index(box.p))
        self.schedule()

    def remove_selected(self):
        box = self.selected()
        if box is None:
            return
        i = self.build.chars.index(box.p)
        self.build.chars.pop(i)
        if self.auto_layout:
            auto_arrange(self.build)
        self.sync_boxes(select=min(i, len(self.build.chars) - 1) if self.build.chars else None)
        self.update_name()
        self.schedule()

    def nudge(self, dx, dy):
        box = self.selected()
        if box is not None:
            box.p.cx += dx
            box.p.cy += dy
            box.sync()
            self.box_changed(box)

    def update_name(self):
        if not self.name_typed:
            self.name.setText("".join(p.name for p in self.build.chars))

    def schedule(self):
        self.timer.start()

    def render(self):
        """Draw the seal at the resolution the canvas is zoomed to, see-through outside its shape."""
        b = self.build
        zoom = self.view.transform().m11() * self.view.devicePixelRatioF()
        k = min(60.0, max(640.0 / max(b.w_mm, b.h_mm), zoom))
        ink = render_build(b, k)
        H, W = ink.shape
        rgba = np.empty((H, W, 4), np.uint8)
        rgba[..., :3] = np.where(ink, 0, 255)[..., None]
        rgba[..., 3] = np.where(shape_inside(b, W, H, k), 255, 0)
        img = QtGui.QImage(rgba.data, W, H, 4 * W, QtGui.QImage.Format.Format_RGBA8888).copy()
        self.view.set_image(img, k)

    # ---- starting over, editing, saving
    def clear(self):
        keep = (self.build.shape, self.build.w_mm, self.build.h_mm, self.build.frame_mm, self.build.invert_all)
        self.build = Build(*keep, corner_mm=self.build.corner_mm)
        self.lib_id = None
        self.auto_layout = True
        self.name_typed = False
        self.name.clear()
        self.sync_boxes()
        self.schedule()

    def load(self, entry: dict):
        """Open a seal built earlier so it can be changed."""
        self.build = load_build(entry)
        self.lib_id = Path(entry["dir"]).name
        self.auto_layout = False
        self.name_typed = True
        self.name.setText(entry.get("name", ""))
        b = self.build
        preset = next((i for i, (_, s, w, h) in enumerate(FORMATS) if (s, w, h) == (b.shape, b.w_mm, b.h_mm)),
                      len(FORMATS))
        self._loading = True
        self.cmb_format.setCurrentIndex(preset)
        self.cmb_shape.setCurrentIndex(0 if b.shape == "rect" else 1)
        self.spn_w.setValue(b.w_mm)
        self.spn_h.setValue(b.h_mm)
        self.spn_frame.setValue(b.frame_mm)
        self.chk_white.setChecked(b.invert_all)
        self.chk_round.setChecked(b.corner_mm > 0)
        if b.corner_mm > 0:
            self.spn_corner.setValue(b.corner_mm)
        self.chk_round.setEnabled(b.shape == "rect")
        self.spn_corner.setEnabled(b.shape == "rect" and b.corner_mm > 0)
        self._loading = False
        self.view.set_format(b)
        self.sync_boxes()
        self.schedule()

    def save(self, add: bool):
        if not self.build.chars:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Add at least one character first.")
            return
        name = self.name.text().strip() or "".join(p.name for p in self.build.chars) or "Built seal"
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.lib_id = save_build(self.build, name, self.lib_id)
        except Exception as ex:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Couldn't save the seal.\n{ex}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        entry = next((e for e in library_entries() if e["dir"].name == self.lib_id), None)
        if entry is not None:
            self.saved.emit(entry, add)


class LibraryDialog(QtWidgets.QDialog):
    """Saved seals to put back on the sheet, characters picked out of them, and seals built
    from those characters."""
    THUMB = 96

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Seal library")
        self.picked: list[dict] | None = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs)

        saved = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(saved)
        v.setContentsMargins(10, 12, 10, 10)
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
        self.empty = QtWidgets.QLabel("Nothing saved yet. Select a seal on the sheet and press Save to keep it "
                                      "here, or build one in the Build from scratch tab.")
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
        self.btn_pick = QtWidgets.QPushButton("Pick characters…")
        self.btn_pick.setToolTip("Cut single characters out of the selected seal, to build new seals with")
        self.btn_pick.clicked.connect(self._pick_characters)
        self.btn_edit = QtWidgets.QPushButton("Edit build")
        self.btn_edit.setToolTip("Change a seal you built from characters")
        self.btn_edit.clicked.connect(self._edit_build)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.btn_add = QtWidgets.QPushButton("Add to sheet")
        self.btn_add.setObjectName("primary")
        self.btn_add.setDefault(True)
        self.btn_add.clicked.connect(self.accept)
        for b in (self.btn_folder, self.btn_delete, self.btn_pick, self.btn_edit):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(cancel)
        row.addWidget(self.btn_add)
        v.addLayout(row)
        self.tabs.addTab(saved, "Saved seals")

        self.composer = Composer(self)
        self.composer.saved.connect(self._built)
        self.composer.cancelled.connect(self.reject)
        self.tabs.addTab(self.composer, "Build from scratch")
        self.tabs.currentChanged.connect(lambda i: self.composer.refresh_characters() if i == 1 else None)
        fit_to_screen(self, 1180, 760)
        self._populate()

    def _populate(self):
        self.grid.clear()
        entries = library_entries()
        counts: dict[str, int] = {}
        for c in character_entries():
            counts[c.get("parent_id", "")] = counts.get(c.get("parent_id", ""), 0) + 1
        fm = self.grid.fontMetrics()
        for e in entries:
            name = e.get("name") or e["dir"].name
            size = fmt_mm(float(e.get("size", 0)))
            item = QtWidgets.QListWidgetItem(
                fm.elidedText(name, Qt.TextElideMode.ElideRight, self.tile.width() - 12) + "\n"
                + (f"{size} mm" if e.get("square", True) else f"longest {size} mm"))
            n = counts.get(e["dir"].name, 0)
            built = (e["dir"] / "build.json").is_file()
            item.setToolTip(f"{name}\nSaved {e.get('saved', '')}" + ("\nBuilt from characters" if built else "")
                            + (f"\n{plural(n, 'character')} picked from it" if n else ""))
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

    def _selected(self) -> list[dict]:
        return [self.entries[it.data(Qt.ItemDataRole.UserRole)] for it in self.grid.selectedItems()]

    def _selection(self):
        picked = self._selected()
        n = len(picked)
        self.btn_add.setEnabled(n > 0)
        self.btn_delete.setEnabled(n > 0)
        self.btn_pick.setEnabled(n == 1)
        self.btn_edit.setEnabled(n == 1 and (picked[0]["dir"] / "build.json").is_file())
        self.btn_add.setText("Add to sheet" if n < 2 else f"Add {n} to sheet")

    def chosen(self) -> list[dict]:
        return self.picked if self.picked is not None else self._selected()

    def _pick_characters(self):
        picked = self._selected()
        if len(picked) != 1:
            return
        try:
            picker = CharacterPicker(self, picked[0])
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Couldn't open that seal.\n{ex}")
            return
        picker.exec()
        self._populate()
        self.composer.refresh_characters()

    def _edit_build(self):
        picked = self._selected()
        if len(picked) == 1:
            self.composer.load(picked[0])
            self.tabs.setCurrentIndex(1)

    def _built(self, entry: dict, add: bool):
        self._populate()
        if add:
            self.picked = [entry]
            self.accept()
        else:
            self.window().setWindowTitle(f"Seal library: saved “{entry.get('name')}”")

    def _delete(self):
        picked = self._selected()
        if not picked:
            return
        what = f"“{picked[0].get('name')}”" if len(picked) == 1 else f"{len(picked)} saved seals"
        sure = QtWidgets.QMessageBox.question(self, "Delete saved seals",
                                              f"Delete {what} from the library? This can't be undone. "
                                              f"Characters picked from it stay in your character list.")
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
QPushButton:checked {{ background: {c['item_sel']}; border-color: {c['accent']}; color: {c['text']}; }}
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
QTabWidget::pane {{ border: 1px solid {c['rule']}; border-radius: 6px; top: -1px; background: {c['side']}; }}
QTabBar::tab {{ background: {c['button']}; color: {c['muted']}; border: 1px solid {c['border']}; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px; padding: 6px 16px; margin-right: 3px; }}
QTabBar::tab:selected {{ background: {c['side']}; color: {c['text']}; border-color: {c['rule']}; }}
QTabBar::tab:hover {{ color: {c['text']}; }}
QLabel#glyphpreview {{ background: #ffffff; color: #666666; border: 1px solid {c['rule']}; border-radius: 4px; }}
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
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
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
    if sys.platform == "win32":                 # own taskbar button and icon on Windows
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SealPrint")
        except Exception:
            pass
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
