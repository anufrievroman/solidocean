"""Helpers shared by the GUI adapters."""

from __future__ import annotations


class Tint(str):
    """A condition type whose tint varies over its faces: weight(points (n, 3), SI) -> the factor there (1 = the
    full tint). scale (m) and focus (x, y) mark a small feature, a beam spot, that the geometry view refines
    around even where the coarse preview triangles miss it. Equal to the plain type, so legends and lookups hold."""

    def __new__(cls, kind: str, weight, scale: float | None = None, focus=None):
        tint = super().__new__(cls, kind)
        tint.weight, tint.scale, tint.focus = weight, scale, focus
        return tint


def condition_highlights(project, conditions, out: dict | None = None, tint=None) -> dict:
    """Viewer tints of conditions with a selection: (face, object box, selection box) -> condition type, or a
    Tint where tint(condition, evaluator) gives one (a spatial profile)."""
    from ..panels.common import model_box
    out = {} if out is None else out
    try:
        ev = project.evaluator()
    except Exception:          # invalid variables: marked in the variables panel; nothing to place
        return out
    for c in conditions:
        obj = tuple(tuple(map(float, v)) for v in model_box(project, only=c.on)) if c.on else None
        try:
            within = (tuple(ev.vector(c.box["min"])), tuple(ev.vector(c.box["max"]))) if c.box else None
        except Exception:
            continue
        kind = (tint(c, ev) if tint else None) or c.type
        for face in c.faces:
            out[(face, obj, within)] = kind
    return out

