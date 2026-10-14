#!/usr/bin/env bash
# Seal Print launcher.
#   ./run.sh [image]     start the app (sets up its Python environment on first run)
#   ./run.sh --install   set up the environment and add "Seal Print" to your app launcher
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV="$HERE/.venv"

setup() {
  echo "Setting up the Python environment in $VENV ..."
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install -r "$HERE/requirements.txt"
  touch "$VENV/.ready"
}

if [ "${1:-}" = "--install" ]; then
  setup
  APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  mkdir -p "$APPS"
  cat > "$APPS/sealprint.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Seal Print
Comment=Print seal designs at exact sizes on A4
Exec="$HERE/run.sh" %f
Icon=$HERE/sealprint.svg
Terminal=false
Categories=Graphics;
MimeType=image/png;image/jpeg;image/webp;image/bmp;image/gif;image/tiff;
StartupWMClass=sealprint
DESKTOP
  command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" >/dev/null 2>&1 || true
  echo "Done. Open \"Seal Print\" from your app launcher, or run $HERE/run.sh"
  exit 0
fi

[ -f "$VENV/.ready" ] || setup
exec "$VENV/bin/python" "$HERE/sealprint.py" "$@"
