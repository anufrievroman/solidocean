"""Helpers shared by the GUI adapters."""

from __future__ import annotations


def condition_highlights(project, conditions, out: dict | None = None) -> dict:
    """Viewer tints of conditions with a selection: (face, object box, selection box) -> condition type."""
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
        for face in c.faces:
            out[(face, obj, within)] = c.type
    return out

