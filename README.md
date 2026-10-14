# Seal Print

Turn a scan, photo or screenshot of a Chinese seal (印章) into a pure black-and-white print at its exact
physical size on A4, ready to transfer onto a seal stone.

![Seal Print with three seals on one sheet](docs/screenshot.png)

## What it does

- **Finds the seal by itself.** Rough screenshots are fine: extra margin, captions, a dark image-viewer
  background, a slight tilt or a chipped border. Drag the corners if you want to adjust the frame.
- **Pure black and white.** The PDF holds a lossless 1-bit image at 1200 dpi. Stroke weight and edge
  smoothing are adjustable, and red seal paste is read separately from black calligraphy around it.
- **Square seals and other shapes.** Rectangles, ovals, circles and free-form seals keep their
  proportions and are sized by their longest length.
- **Many seals on one sheet.** Each seal has its own size and number of copies, and everything is packed
  onto as few A4 pages as possible.
- **Exact size, even on stubborn printers.** For self-service printers that always shrink pages to fit,
  the Enlarge setting compensates, and Calibrate works out the number from a test print.
- **Seal library.** Save a seal with its frame and settings, then load it onto any later sheet.
- **Close-up inspection.** Click the black-and-white seal to zoom into the exact pixels that get printed.
- **Two-tone themes.** Dark, Ink, Lacquer, Paper, Celadon, Blue and white, or your own two colours.

## Install

### Linux

You need Python 3 (on Debian or Ubuntu also `sudo apt install python3-venv`). Clone this repository or
download it as a zip from the green Code button, then in its folder:

```sh
./run.sh --install
```

That puts PySide6, OpenCV and NumPy into `.venv` inside the folder (nothing is installed system-wide)
and adds **Seal Print** to your app launcher. `./run.sh` also starts it.

### Windows

Download `SealPrint-windows.zip` from the [Releases](../../releases) page, unzip it and run
`SealPrint.exe`. The exe isn't signed, so the first time Windows shows "Windows protected your PC":
click **More info**, then **Run anyway**.

## Using it

1. **Add** image files, **Paste** a screenshot (Ctrl+V) or drop files on the window. Each image becomes
   a seal in the strip along the bottom.
2. Click a seal in the strip to edit it. If its red frame is off, drag the corners; scroll to zoom,
   double-click to fit.
3. Under **This seal**, choose Square or Other shape, the size in mm (the longest length for other
   shapes) and the number of copies. Stroke weight, Smooth edges and Invert tune the black-and-white
   result. Right-click a seal in the strip to duplicate or remove it.
4. Click the black-and-white seal to inspect it up close (scroll or + and − to zoom, click to go back).
5. **Export PDF…**

## Printing

- Print at **100%** or **Actual size**. "Fit to page" shrinks everything by a few percent, so measure a
  seal to check.
- If a printer can't print at actual size, export once, print, press **Calibrate…** and enter what a
  seal should measure and what it measured. Export again with the new Enlarge value.
- For face-down toner transfer leave **Mirror every seal** off, since the transfer flips the design
  onto the stone. Turn it on if the print goes onto the stone face up.

## Seal library

**Save** keeps the selected seal with its image (cropped around the frame), the frame itself and all
its settings. **Load…** brings saved seals back. Each seal is a folder with `image.png`, `seal.json`
and `thumb.png`:

- Linux: `~/.local/share/sealprint/library`
- Windows: `%LOCALAPPDATA%\sealprint\library`

Copy that folder to back up your seals or move them to another computer.

## Command line

```sh
python sealprint.py a.png:20:6 b.png:30:2:other -o seals.pdf [--gap 5] [--scale 103.6] [--mirror] [--invert] [--ink red|dark]
```

Each image can carry `:size_mm:copies`, plus `:other` for a non-square seal (the size is then its
longest length). `--size`, `--copies` and `--shape` set the defaults; `--help` lists everything.

## Notes for Linux

On Wayland desktops that set `QT_IM_MODULE=fcitx` (Omarchy, for example), the app switches itself to
Wayland's own text input so fcitx5 can type Chinese into it. The pip-installed copy of Qt doesn't
include fcitx's Qt plugin.

## Development

```sh
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Every push runs the tests and builds the Windows version on GitHub Actions; the zip is attached to the
run under **Artifacts**. Pushing a version tag (`git tag v1.0.0 && git push origin v1.0.0`) also
publishes it on the Releases page.

## License

MIT, see [LICENSE](LICENSE). Seal Print uses PySide6 (Qt for Python, LGPLv3), OpenCV (Apache 2.0)
and NumPy (BSD).
