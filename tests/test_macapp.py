import plistlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")


def test_install_app_bundle(tmp_path):
    from solidocean.macapp import BUNDLE_ID, install_app
    app = install_app(tmp_path)
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleName"] == "SolidOcean" and info["CFBundleIdentifier"] == BUNDLE_ID
    assert (app / "Contents" / "Resources" / "SolidOcean.icns").stat().st_size > 10_000
    python = app / "Contents" / "MacOS" / "SolidOcean"
    out = subprocess.run([str(python), "-m", "solidocean", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and "solidocean" in out.stdout
    install_app(tmp_path)  # reinstalling over our own bundle is allowed


def test_install_app_refuses_foreign_bundle(tmp_path):
    from solidocean.macapp import install_app
    (tmp_path / "SolidOcean.app" / "Contents").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="not a SolidOcean bundle"):
        install_app(tmp_path)
