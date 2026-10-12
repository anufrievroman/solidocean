"""Physics registry. Every physics is a subpackage with a PhysicsSpec (see base.py); shared code looks physics
up here instead of testing for them. Adding a physics: write the subpackage (settings, solver, results,
plots, templates, spec), register its SPEC below, add its settings section to Project, and its GUI adapter
in gui/physics/."""

from __future__ import annotations

from .base import Cancelled, PhysicsSpec  # noqa: F401
from .heat import SPEC as _HEAT
from .solid import SPEC as _SOLID

REGISTRY: dict[str, PhysicsSpec] = {spec.kind: spec for spec in (_SOLID, _HEAT)}   # in menu order
KINDS = tuple(REGISTRY)


def get(kind: str) -> PhysicsSpec:
    if kind not in REGISTRY:
        raise ValueError(f"unknown physics '{kind}' (use {', '.join(REGISTRY)})")
    return REGISTRY[kind]


def of(project) -> PhysicsSpec:
    """The physics a project solves."""
    return get(project.physics_kind)


def active(project) -> list[PhysicsSpec]:
    """Every physics of the project's setup (`project.physics_kind`). One for now; coupled setups (e.g. thermal
    expansion: heat transfer and solid mechanics) will list several, each with its own settings table."""
    return [of(project)]


def titles() -> dict[str, str]:
    return {kind: spec.title for kind, spec in REGISTRY.items()}


def all_field_names() -> tuple[str, ...]:
    return tuple(dict.fromkeys(n for spec in REGISTRY.values() for n in spec.field_names))


def all_scalar_names() -> tuple[str, ...]:
    return tuple(dict.fromkeys(n for spec in REGISTRY.values() for n in spec.scalar_names))
