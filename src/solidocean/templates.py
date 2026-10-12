"""Ready-made, consistent starting models (templates), collected from every physics.

Each template is a complete project: variables, materials, geometry, physics settings, study and plots
that fit together; users then edit the variables. Every physics lists its own in
physics/<name>/templates.py. Available in the GUI (New) and the CLI (--template NAME FILE).
"""

from __future__ import annotations

from typing import Callable, NamedTuple

from .materials import LIBRARY
from .project import Material, Project


class Template(NamedTuple):
    title: str              # also the model name; descriptive enough for the New dialog
    factory: Callable[[], Project]
    physics: str            # physics kind: the New dialog groups templates by physics


def library_material(name: str, **overrides) -> Material:
    """A library material with its properties written out (so the model file shows them), plus overrides."""
    lib = LIBRARY[name]
    return Material(name, {"model": lib["model"], **{k: v for k, v in lib.items() if k != "model"}, **overrides})


def all_templates() -> dict[str, Template]:
    """name -> template, physics by physics, each in display order."""
    from .physics import REGISTRY
    out = {}
    for spec in REGISTRY.values():
        out.update(spec.templates())
    return out


def by_physics(physics: str) -> dict[str, Template]:
    return {k: t for k, t in all_templates().items() if t.physics == physics}


def create(name: str) -> Project:
    templates = all_templates()
    if name not in templates:
        raise ValueError(f"unknown template '{name}' (choose from {', '.join(templates)})")
    return templates[name].factory()
