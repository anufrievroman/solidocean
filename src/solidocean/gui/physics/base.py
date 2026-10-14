"""GUI side of a physics: what the main window asks of it, so the window itself never tests which physics it
shows. One subclass per physics (solid.py, heat.py), registered in __init__.py next to the core PhysicsSpec."""

from __future__ import annotations

TreeNode = tuple[str, tuple, bool]   # (label, (node kind, index or None), dimmed)


class PhysicsUI:
    kind = ""
    physics_panel: type = None       # settings panel of the physics node
    problem_panel: type = None         # settings panel of the problem node
    field_panel: type = None         # panel of 3D field plots (the field of the selected solution)
    node_panels: dict[str, type] = {}        # node kind -> panel class, panel(win)
    indexed_panels: dict[str, type] = {}     # node kind -> panel class, panel(win, index)
    highlight_nodes: tuple[str, ...] = ()    # node kinds that tint the boundary conditions in the 3D view

    # ---- model tree ------------------------------------------------------------------
    def physics_nodes(self, project) -> list[TreeNode]:
        """Children of the physics node."""
        return []

    def problem_label(self, project) -> str:
        return "Problem"

    def problem_nodes(self, project) -> list[TreeNode]:
        """Children of the problem node."""
        return []

    def context_menu(self, win, kind: str, index, menu) -> None:
        """Add the actions of a physics node (or of the physics section) to its context menu."""

    def delete_node(self, win, kind: str, index) -> bool:
        """Delete a physics node (Delete key); False if it is not one of this physics."""
        return False

    def highlights(self, project) -> dict:
        """Tints of the boundary conditions in the 3D view: (face, object box, selection box) or face ->
        condition type (see viewer._cell_colors)."""
        return {}

    # ---- results ---------------------------------------------------------------------
    def run_label(self, project) -> str:
        return "Solving…"

    def show_field(self, win) -> None:
        """Show the field of the selected solution (win.selection) of field results."""

    def results_summary(self, win, results) -> str:
        """One line about field results."""
        return ""

    def quantity_placeholder(self, project) -> str:
        return ""

    def line_help(self, project) -> str:
        return ""

    def plot_help(self, project, kind: str) -> str:
        return ""
