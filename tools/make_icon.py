"""Render the SolidOcean app icon (src/solidocean/gui/resources/icon.png).

An ocean shore seen from above, drawn as a triangulated (finite-element) mesh: the mesh points lie on
wave crests that run along the shore (the bottom-left to top-right diagonal is open sea to beach), the
crests bunch up as they approach the shore, and every triangle is coloured by its distance from the open
sea, with a faint foam just behind each crest.

Run: python tools/make_icon.py [output.png] [palette: shore (default), viridis, tropical, mako, deep, cividis, dusk, lagoon] [foam or nofoam]
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QPolygonF
from scipy.spatial import Delaunay

SIZE = 1024
MARGIN = 100           # macOS icon grid: the body is inset from the canvas
RADIUS = 185           # corner radius of the body
CRESTS = 11            # wave crests from the open sea to the shore
SHOALING = 1.7         # > 1: crests get closer together towards the shore
POINTS = 13            # typical points per crest across the icon (each crest varies)
BETWEEN = 0.45         # extra random points between crests, per crest point
WAVELENGTH = 0.75      # of the crest undulation along the shore (diagonal units)
SEED = 7               # fixed: the same mesh on every run
EDGE = QColor(255, 255, 255, 120)      # mesh edges
EDGE_WIDTH = 5.5
CREST = QColor(255, 255, 255, 215)     # wave crests (foam lines)
CREST_WIDTH = 9
OUT = Path(__file__).resolve().parent.parent / "src" / "solidocean" / "gui" / "resources" / "icon.png"

def _palette(name, colors):
    return LinearSegmentedColormap.from_list(name, colors)


PALETTES = {   # open sea (bottom-left) -> shore (top-right)
    "viridis": colormaps["viridis"],
    "shore": _palette("shore", ["#0b2a5b", "#13608f", "#1b9aaa", "#5cc8c0", "#bfe8dc", "#e9d8a6", "#f2e3bc"]),
    "tropical": _palette("tropical", ["#03045e", "#0077b6", "#00b4d8", "#48cae4", "#90e0ef", "#f3e9c6"]),
    "mako": _palette("mako", ["#0b0405", "#2e1e3b", "#3b4f8a", "#357ba3", "#3ba3ac", "#60ceac", "#def5e5"]),
    "deep": colormaps["YlGnBu"].reversed(),
    "cividis": colormaps["cividis"],
    "dusk": _palette("dusk", ["#1d1145", "#47217a", "#7a2f8f", "#b9407f", "#e8686f", "#f7a35c", "#fde2a4"]),
    "lagoon": _palette("lagoon", ["#05445e", "#0a7c86", "#1fb5a8", "#75e6cf", "#d4f5e9", "#f6efd2"]),
}


def crests() -> np.ndarray:
    """Position s of each crest across the diagonal (0 = open sea corner, 1 = shore corner)."""
    k = np.linspace(0, 1, CRESTS)
    return -0.08 + 1.16 * (1 - (1 - k) ** SHOALING)


def crest_shape(k: int, phases: np.ndarray):
    """s(w) of crest k: a gentle undulation along the shore, growing as the wave approaches the shore."""
    s0 = crests()[k]
    amp = 0.012 + 0.028 * (k / (CRESTS - 1)) ** 1.5
    return lambda w: s0 + amp * np.sin(2 * np.pi * w / WAVELENGTH + phases[k])


def mesh_points(rng, phases) -> tuple[np.ndarray, np.ndarray]:
    """Points on the wavy crests, in diagonal coordinates s (sea -> shore) and w (along the shore);
    returned as (x, y) on the unit square (some lie outside: the icon is clipped) with the crest index."""
    pts, rows = [], []
    s_crest = crests()
    for k in range(CRESTS):
        n = POINTS + int(rng.integers(-3, 4))                 # each crest has its own spacing ...
        dw = 1.3 / (n - 1)
        w = np.linspace(-0.65, 0.65, n) + rng.uniform(0, dw)  # ... and offset
        w = w + rng.uniform(-0.35, 0.35, n) * dw              # irregular along the crest
        s = crest_shape(k, phases)(w)
        pts.append(np.column_stack([s + w, s - w]))           # back to x, y
        rows.append(np.full(n, k))
        if k + 1 < CRESTS:                                    # scattered points in the trough behind it
            m = rng.binomial(n, BETWEEN)
            wb = rng.uniform(-0.65, 0.65, m)
            f = rng.uniform(0.3, 0.7, m)
            sb = (1 - f) * crest_shape(k, phases)(wb) + f * crest_shape(k + 1, phases)(wb)
            pts.append(np.column_stack([sb + wb, sb - wb]))
            rows.append(np.full(m, -1))
    return np.vstack(pts), np.concatenate(rows)


def render(palette: str = "shore", foam: bool = True) -> QImage:
    rng = np.random.default_rng(SEED)
    img = QImage(SIZE, SIZE, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    body = QRectF(MARGIN, MARGIN, SIZE - 2 * MARGIN, SIZE - 2 * MARGIN)
    clip = QPainterPath()
    clip.addRoundedRect(body, RADIUS, RADIUS)
    p.setClipPath(clip)

    phases = np.cumsum(rng.uniform(0.5, 1.1, CRESTS))     # neighbouring crests drift slowly in phase
    pts, row = mesh_points(rng, phases)
    tri = Delaunay(pts).simplices
    span = SIZE - 2 * MARGIN

    def to_px(q):          # unit square -> body (y up in the mesh, down on screen)
        return QPointF(MARGIN + q[0] * span, SIZE - MARGIN - q[1] * span)

    cmap = PALETTES[palette]
    pen = QPen(EDGE, EDGE_WIDTH)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    s_crest = crests()
    for t in tri:
        c = pts[t].mean(axis=0)
        s = (c[0] + c[1]) / 2
        color = np.array(cmap(float(np.clip(s, 0, 1)))[:3])
        if foam:   # whiter just shoreward of a crest, more so near the shore
            behind = s - s_crest
            d = behind[behind >= 0].min() if np.any(behind >= 0) else 1.0
            color = color + (1 - color) * 0.22 * np.exp(-(d / 0.035) ** 2) * np.clip(s, 0, 1)
        p.setBrush(QColor.fromRgbF(*np.clip(color, 0, 1)))
        p.drawPolygon(QPolygonF([to_px(pts[k]) for k in t]))
    # the crests as foam lines over the mesh, fading out on the beach
    p.setBrush(Qt.BrushStyle.NoBrush)
    w = np.linspace(-0.7, 0.7, 300)
    for k in range(CRESTS):
        s = crest_shape(k, phases)(w)
        fade = float(np.clip((1.0 - np.mean(s)) / 0.25, 0, 1))
        if fade <= 0:
            continue
        line = QColor(CREST)
        line.setAlpha(int(CREST.alpha() * fade))
        pen = QPen(line, CREST_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawPolyline(QPolygonF([to_px((si + wi, si - wi)) for si, wi in zip(s, w)]))
    p.end()
    return img


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    palette = sys.argv[2] if len(sys.argv) > 2 else "shore"
    foam = (sys.argv[3] != "nofoam") if len(sys.argv) > 3 else True
    out.parent.mkdir(parents=True, exist_ok=True)
    render(palette, foam).save(str(out))
    print(f"wrote {out}")
