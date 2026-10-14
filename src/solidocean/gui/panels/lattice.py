"""Lattice helpers shared by the geometry (lattice cell) and the periodicity panels."""

from __future__ import annotations

from ...project import Feature


def _lattice_kind(project) -> str:
    from ... import lattice as lat
    try:
        return lat.classify(project.lattice, project.evaluator())
    except Exception:
        return "custom"


def _wrap(expr: str) -> str:
    """Parenthesise an expression unless it is a plain name or number."""
    import re
    expr = expr.strip()
    return expr if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|[0-9.]+(\[[^\]]*\])?", expr) else f"({expr})"


def hexagonal_vectors(const: str) -> list[list[str]]:
    c = _wrap(const)
    return [[const.strip(), "0", "0"], [f"{c}/2", f"sqrt(3)*{c}/2", "0"]]


def lattice_mode(lt) -> tuple[str, str]:
    """('rect' | 'hex' | 'custom', lattice constant for hex)."""
    if not lt.vectors:
        return "rect", ""
    if len(lt.vectors) == 2 and lt.vectors == hexagonal_vectors(str(lt.vectors[0][0])):
        return "hex", str(lt.vectors[0][0])
    return "custom", ""


def use_lattice_cell(win):
    """Make the unit cell of the lattice the base solid: replace the first block or add a cell."""
    from ...project import Feature
    geo = win.project.objects
    block = next((f for f in geo if f.enabled and f.op == "add" and f.type == "block"), None)
    if block is not None:
        size = block.params.get("size", ["1", "1", "1"])
        cell = Feature("cell", block.name, "add", block.material,
                       {"center": list(block.params.get("center", ["0", "0", "0"])), "height": size[2]})
        geo[geo.index(block)] = cell
        win.log(f"Replaced block '{block.name}' with the lattice unit cell")
    else:
        material = win.project.materials[0].name if win.project.materials else ""
        geo.insert(0, Feature("cell", "cell", "add", material, {"center": ["0", "0", "0"], "height": "1[um]"}))
        win.log("Added a lattice unit cell object")
    win.preview = None
    win.mark_dirty()
    win.rebuild_tree(("periodicity", None))
