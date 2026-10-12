"""GUI adapters of the physics (see base.py), keyed like physics.REGISTRY."""

from __future__ import annotations

from ... import physics
from .base import PhysicsUI
from .heat import HeatUI
from .solid import SolidUI

UI: dict[str, PhysicsUI] = {ui.kind: ui for ui in (SolidUI(), HeatUI())}
assert set(UI) == set(physics.REGISTRY), "every physics needs a GUI adapter"


def of(project) -> PhysicsUI:
    return UI[project.physics_kind]
