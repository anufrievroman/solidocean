"""Create a macOS application bundle ("SolidOcean.app") for the GUI.

The bundle contains a launcher script and a symlink to this environment's
python named "SolidOcean". Because the running executable lives inside the
bundle, macOS shows the app's own name and icon in the Dock, menu bar and
Cmd-Tab instead of "Python".
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import __version__

APP_NAME = "SolidOcean"
BUNDLE_ID = "org.solidocean.SolidOcean"
ICON_PNG = Path(__file__).parent / "gui" / "resources" / "icon.png"

LAUNCHER = """#!/bin/bash
# Started by Finder/Dock: run the GUI with the python environment it was installed from.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/{name}" -m solidocean --gui "$@"
"""


def _make_icns(png: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                subprocess.run(["sips", "-z", str(px), str(px), str(png), "--out", str(iconset / name)],
                               check=True, capture_output=True)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True, capture_output=True)


def install_app(dest_dir: str | Path) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("--install-app is only available on macOS")
    dest_dir = Path(dest_dir).expanduser()
    app = dest_dir / f"{APP_NAME}.app"
    if app.exists():
        info = app / "Contents" / "Info.plist"
        try:
            ours = plistlib.loads(info.read_bytes()).get("CFBundleIdentifier") == BUNDLE_ID
        except Exception:
            ours = False
        if not ours:
            raise RuntimeError(f"{app} exists and is not a SolidOcean bundle; not replacing it")
        shutil.rmtree(app)

    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    resources.mkdir()
    launcher = macos / "solidocean-launcher"
    launcher.write_text(LAUNCHER.format(name=APP_NAME), encoding="utf-8")
    launcher.chmod(0o755)
    os.symlink(os.path.realpath(sys.executable), macos / APP_NAME)
    _make_icns(ICON_PNG, resources / "SolidOcean.icns")
    plist = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": launcher.name,
        "CFBundleIconFile": "SolidOcean",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    }
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps(plist))
    # make Finder/Launch Services notice the new bundle and icon
    subprocess.run(["touch", str(app)], check=False)
    return app
