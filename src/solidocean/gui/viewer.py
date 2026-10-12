"""3D graphics window: geometry, mesh and mode shapes (pyvista)."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QToolBar, QVBoxLayout, QWidget

from .. import units

from ..materials import DEFAULT_COLORS
from ..meshing import Mesh, SurfacePreview

pv.global_theme.font.family = "arial"  # same typeface as the matplotlib plots

TRIAD_GRAY = "#808080"
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "coolwarm", "RdBu_r", "jet", "gray"]

BACKGROUND = "#f4f5f7"
# boundary-condition tints; periodic colours match the axes triad (x red, y green, z blue)
BC_COLORS = {"periodic x": "#ff6347", "periodic y": "#2e8b57", "periodic z": "#4169e1",
             "periodic a1": "#ff6347", "periodic a2": "#2e8b57", "periodic a3": "#4169e1",
             "symmetry": "#9b59d0", "antisymmetry": "#e69a28", "fixed": "#3c3c3c",
             "displacement": "#8c564b", "load": "#bcbd22",
             "temperature": "#d62728", "flux": "#c2189b", "convection": "#17becf",
             "radiation": "#ff7f0e"}
BC_LEGEND = {"periodic x": "Periodic x", "periodic y": "Periodic y", "periodic z": "Periodic z",
             "periodic a1": "Periodic a1", "periodic a2": "Periodic a2", "periodic a3": "Periodic a3",
             "symmetry": "Symmetry", "antisymmetry": "Antisymmetry", "fixed": "Fixed",
             "displacement": "Prescribed displacement", "load": "Boundary load",
             "temperature": "Temperature", "flux": "Heat flux", "convection": "Convection",
             "radiation": "Radiation"}
BC_BLEND = 0.6
# horizontal colour bar at the top centre; its title is drawn separately (Viewer._bar_title)
_BAR = {"title": "", "vertical": False, "position_x": 0.3, "position_y": 0.83, "width": 0.4, "height": 0.06,
        "n_labels": 3, "label_font_size": 11, "font_family": "arial", "color": "black"}


def vtk_text(text: str) -> str:
    """VTK's built-in font has no µ glyph."""
    return text.replace("µ", "u")


def _rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def _cell_colors(poly: pv.PolyData, domain: np.ndarray, conditions: dict[str, str] | None,
                 periodic=(), colors=None) -> np.ndarray:
    """Per-triangle RGB: material colour, tinted where a boundary condition applies.

    conditions: bounding-box face -> condition, or (face, (lo, hi)) -> condition for a face of an object's
    box (e.g. the top of a pad: triangles in that plane within the box), or (face, object box or None, selection
    box or None) -> condition, where face may be "all" (the whole surface, of the object's box if given) and the
    selection box keeps only triangles inside it; periodic: (name, lattice vector) pairs, whose faces are found
    as triangles that a lattice translation maps onto another triangle.
    """
    colors = colors or DEFAULT_COLORS
    base = np.array([_rgb(colors[d % len(colors)]) if d >= 0 else _rgb("#bbbbbb")
                     for d in range(domain.max() + 1)] or [_rgb("#bbbbbb")])
    rgb = base[np.clip(domain, 0, None)]
    if not conditions and not periodic:
        return rgb
    tri = poly.points[poly.regular_faces]           # (n, 3 vertices, 3 coords)
    lo, hi = poly.points.min(axis=0), poly.points.max(axis=0)
    tol = 1e-6 * np.linalg.norm(hi - lo)
    if periodic:
        from scipy.spatial import cKDTree
        centres = tri.mean(axis=1)
        tree = cKDTree(centres)
        for name, vec in periodic:
            on = np.zeros(len(centres), dtype=bool)
            for sign in (1, -1):
                dist, _ = tree.query(centres + sign * np.asarray(vec))
                on |= dist < 1e3 * tol
            rgb[on] = (1 - BC_BLEND) * rgb[on] + BC_BLEND * _rgb(BC_COLORS[f"periodic {name}"])
    conditions = conditions or {}
    for key, cond in conditions.items():
        face, box, within = (tuple(key) + (None, None))[:3] if isinstance(key, tuple) else (key, None, None)
        blo, bhi = (np.asarray(box[0]), np.asarray(box[1])) if box is not None else (lo, hi)
        c = tri.mean(axis=1)
        if face == "all":
            on = np.ones(len(tri), dtype=bool)
        else:
            i = "xyz".index(face[0])
            plane = blo[i] if face.endswith("min") else bhi[i]
            on = np.all(np.abs(tri[:, :, i] - plane) < max(tol, 1e-3 * (bhi[i] - blo[i])), axis=1)
        if box is not None:
            pad = 1e-3 * np.linalg.norm(bhi - blo)
            on &= np.all((c >= blo - pad) & (c <= bhi + pad), axis=1)
        if within is not None:
            wlo, whi = np.minimum(within[0], within[1]), np.maximum(within[0], within[1])
            pad = 1e-3 * np.linalg.norm(hi - lo)
            on &= np.all((tri >= np.asarray(wlo) - pad) & (tri <= np.asarray(whi) + pad), axis=(1, 2))
        rgb[on] = (1 - BC_BLEND) * rgb[on] + BC_BLEND * _rgb(BC_COLORS[cond])
    return rgb


def _linear_grid(mesh: Mesh) -> pv.UnstructuredGrid:
    """Corner-node tetra grid (fast to render even for P2 meshes)."""
    used = np.unique(mesh.cells[:, :4])
    remap = np.full(mesh.n_nodes, -1)
    remap[used] = np.arange(len(used))
    cells = remap[mesh.cells[:, :4]]
    flat = np.hstack([np.full((len(cells), 1), 4), cells]).ravel()
    grid = pv.UnstructuredGrid(flat, np.full(len(cells), pv.CellType.TETRA, dtype=np.uint8),
                               mesh.points[used])
    grid.cell_data["domain"] = mesh.cell_domain
    grid.field_data["node_ids"] = used
    return grid


class Viewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_toolbar())
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self._last_axis_view = None
        self.length_unit = units.DEFAULT_LENGTH_UNIT
        self.plotter.set_background(BACKGROUND)
        self.plotter.enable_depth_peeling()  # correct ordering of semi-transparent surfaces
        self._mode_grid = None
        self.clear()

    def _build_toolbar(self) -> QToolBar:
        tb = QToolBar()
        tb.setObjectName("viewToolbar")
        for text, tip, fn in (
            ("Zoom In", "Zoom in", lambda: self._zoom(1.25)),
            ("Zoom Out", "Zoom out", lambda: self._zoom(0.8)),
            ("Zoom Fit", "Zoom to fit the whole model (keeps the view direction)", self.fit),
            (None, None, None),
            ("X", "View along x (y-z plane). Click again for the opposite side.", lambda: self.axis_view("x")),
            ("Y", "View along y (x-z plane). Click again for the opposite side.", lambda: self.axis_view("y")),
            ("Z", "View along z (x-y plane). Click again for the opposite side.", lambda: self.axis_view("z")),
            ("Default", "Default angled view", self.default_view),
        ):
            if text is None:
                tb.addSeparator()
                continue
            action = tb.addAction(text, fn)
            action.setToolTip(tip)
        tb.addSeparator()
        self.opaque_action = tb.addAction("Opaque")
        self.opaque_action.setCheckable(True)
        self.opaque_action.setToolTip("Draw the structure opaque; by default it is slightly transparent so that "
                                      "hidden faces (and boundary conditions on them) show through")
        self.opaque_action.triggered.connect(self._apply_opacity)
        self.scale_action = tb.addAction("Rulers")
        self.scale_action.setCheckable(True)
        self.scale_action.setChecked(False)
        self.scale_action.setToolTip("Show rulers along the model's bounding box, in the preferred length unit")
        self.scale_action.triggered.connect(self._toggle_scale)
        save = tb.addAction("Save Image…", self.save_image)
        save.setToolTip("Save the current view as an image")
        return tb

    def set_length_unit(self, unit: str):
        self.length_unit = unit
        self._update_scale()

    def _toggle_scale(self, _on: bool = False):
        self._update_scale()

    def _update_scale(self):
        """(Re)draw bounding-box rulers labelled in the preferred length unit."""
        self.plotter.remove_bounds_axes()
        bounds = np.array(self.plotter.bounds, dtype=float)
        if not self.scale_action.isChecked() or not np.all(np.isfinite(bounds)) or np.ptp(bounds) == 0:
            self.plotter.render()
            return
        scale = units.LENGTH_UNITS.get(self.length_unit, 1e-6)
        lu = vtk_text(units.unit_label(self.length_unit))
        ranges = bounds / scale
        # round to ~4 significant digits of the extent so midpoints are exact (0 rather than 6.7e-07)
        span = np.ptp(ranges.reshape(3, 2), axis=1).max()
        ranges = np.round(ranges, max(0, 3 - int(np.floor(np.log10(span)))))
        self.plotter.show_bounds(bounds=list(bounds), axes_ranges=list(ranges),
                                 xtitle=f"x ({lu})", ytitle=f"y ({lu})", ztitle=f"z ({lu})",
                                 n_xlabels=3, n_ylabels=3, n_zlabels=3, fmt="%.3g", bold=False,
                                 font_size=14, font_family="arial", color="#505050", location="outer",
                                 use_3d_text=False,
                                 render=False)
        self.plotter.render()

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save image", "structure.png",
                                              "PNG image (*.png);;JPEG image (*.jpg)")
        if path:
            # exactly what is on screen (already full Retina resolution); scaling would shrink the text
            self.plotter.screenshot(path)

    def _zoom(self, factor: float):
        self.plotter.camera.zoom(factor)
        self.plotter.render()

    def fit(self):
        self.plotter.reset_camera()
        self.plotter.render()

    def axis_view(self, axis: str):
        """Look along an axis; repeating the same axis flips to the opposite side."""
        negative = self._last_axis_view == (axis, False)
        {"x": self.plotter.view_yz, "y": self.plotter.view_xz, "z": self.plotter.view_xy}[axis](negative=negative)
        self._last_axis_view = (axis, negative)
        self.plotter.reset_camera()
        self.plotter.render()

    def default_view(self):
        self._last_axis_view = None
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    show_triad = True  # the small x/y/z orientation axes

    def set_triad(self, on: bool):
        self.show_triad = on
        self._update_triad()
        self.plotter.render()

    def _update_triad(self):
        if not self.show_triad:
            self.plotter.hide_axes()
            return
        actor = self.plotter.add_axes(color=TRIAD_GRAY, x_color=TRIAD_GRAY, y_color=TRIAD_GRAY,
                                      z_color=TRIAD_GRAY, line_width=2)
        for caption in (actor.GetXAxisCaptionActor2D(), actor.GetYAxisCaptionActor2D(),
                        actor.GetZAxisCaptionActor2D()):
            prop = caption.GetCaptionTextProperty()
            prop.SetFontFamilyToArial()
            prop.SetBold(False)
            prop.SetItalic(False)
            prop.SetShadow(False)
            prop.SetColor(*_rgb(TRIAD_GRAY))

    def set_rulers(self, on: bool):
        """Bounding-box rulers (same as the Rulers toolbar toggle)."""
        self.scale_action.setChecked(on)
        self._update_scale()

    def clear(self):
        """Empty the scene for a new model; the next scene gets the default view."""
        self.plotter.clear()
        self._surface = None
        self._camera_placed = False
        self._update_triad()

    def _apply_opacity(self, _on: bool = False):
        """Opacity of the structure in the geometry and mesh views: the requested one, or 1 if Opaque is on."""
        if getattr(self, "_surface", None) is None:
            return
        actor, opacity = self._surface
        actor.GetProperty().SetOpacity(1.0 if self.opaque_action.isChecked() else opacity)
        self.plotter.render()

    def _finish(self, reset_camera: bool):
        """Overlays, then the camera: the default view the first time something is shown after clear() (or
        when asked), otherwise the user's current zoom and angle are kept (e.g. while rebuilding)."""
        self._update_triad()
        self._update_scale()
        if reset_camera or not self._camera_placed:
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self._camera_placed = True
        self.plotter.render()

    def _legend(self, names: list[str], conditions: dict[str, str] | None, periodic=(), colors=None):
        entries = []
        colors = colors or DEFAULT_COLORS
        if len(names) > 1:
            entries += [(n, colors[i % len(colors)]) for i, n in enumerate(names)]
        active = set((conditions or {}).values()) | {f"periodic {n}" for n, _ in periodic}
        for cond in BC_COLORS:  # fixed order
            if cond in active:
                entries.append((BC_LEGEND[cond], BC_COLORS[cond]))
        # plain coloured labels, top-left (VTK's legend box scales its text and overflows)
        for i, (label, color) in enumerate(entries):
            self.plotter.add_text(label, position=(0.012, 0.96 - 0.03 * i), viewport=True,
                                  font_size=8, color=color, shadow=False)

    def set_line(self, start, end):
        """Overlay a probe line (a line plot's segment) with labelled end points on the current scene."""
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        self.plotter.add_mesh(pv.Line(a, b), color="#d62728", line_width=5, render_lines_as_tubes=True,
                              name="probe_line", pickable=False)
        self.plotter.add_points(np.array([a, b]), color="#d62728", point_size=14, render_points_as_spheres=True,
                                name="probe_ends", pickable=False)
        self.plotter.add_point_labels(np.array([a, b]), ["start", "end"], name="probe_labels", font_size=12,
                                      text_color="#d62728", shape_opacity=0.0, show_points=False,
                                      always_visible=True)
        self.plotter.render()

    def set_plane(self, corners):
        """Overlay a cross-section's plane (4 corners) on the current scene."""
        quad = pv.PolyData(np.asarray(corners, dtype=float), np.array([4, 0, 1, 2, 3]))
        self.plotter.add_mesh(quad, color="#d62728", opacity=0.25, name="probe_plane", pickable=False)
        self.plotter.add_mesh(quad.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                         manifold_edges=False),
                              color="#d62728", line_width=3, name="probe_plane_edge", pickable=False)
        self.plotter.render()

    def show_geometry(self, preview: SurfacePreview, conditions: dict[str, str] | None = None,
                      reset_camera=True, opacity: float = 1.0, periodic=(), material_colors=None):
        """The geometry; material_colors: name -> colour (default: the automatic colours by domain)."""
        self.stop_animation()
        self._mode_state = None
        self.plotter.clear()
        faces = np.hstack([np.full((len(preview.triangles), 1), 3), preview.triangles]).ravel()
        poly = pv.PolyData(preview.points, faces)
        colors = [material_colors[n] for n in preview.material_names] if material_colors else None
        poly.cell_data["rgb"] = _cell_colors(poly, preview.domain, conditions, periodic, colors)
        actor = self.plotter.add_mesh(poly, scalars="rgb", rgb=True, smooth_shading=True, specular=0.2)
        self._surface = (actor, opacity)
        self._apply_opacity()
        self.plotter.add_mesh(poly.extract_feature_edges(30), color="#333333", line_width=1)
        self._legend(preview.material_names, conditions, periodic, colors)
        self._finish(reset_camera)

    def show_mesh(self, mesh: Mesh, conditions: dict[str, str] | None = None, reset_camera=True,
                  opacity: float = 1.0, periodic=(), material_colors=None):
        self.stop_animation()
        self._mode_state = None
        self.plotter.clear()
        surf = _linear_grid(mesh).extract_surface(algorithm="dataset_surface")
        colors = [material_colors[n] for n in mesh.material_names] if material_colors else None
        surf.cell_data["rgb"] = _cell_colors(surf, surf.cell_data["domain"], conditions, periodic, colors)
        actor = self.plotter.add_mesh(surf, scalars="rgb", rgb=True, show_edges=True, edge_color="#2a2a2a",
                                      line_width=0.5)
        self._surface = (actor, opacity)
        self._apply_opacity()
        self._legend(mesh.material_names, conditions, periodic, colors)
        self._finish(reset_camera)

    # ---- displacement field ----------------------------------------------
    COMPONENTS = ("|u|", "ux", "uy", "uz")

    def show_mode(self, mesh: Mesh, u: np.ndarray, title: str, scale: float = 0.05,
                  component: str = "|u|", reset_camera=False, style: dict | None = None):
        """Displacement field u (N, 3) complex, drawn deformed and coloured.

        |u| is the phase-independent amplitude; ux/uy/uz are the real parts at
        the displayed phase (they change while animating).
        """
        self.stop_animation()
        self.plotter.clear()
        self._surface = None
        grid = _linear_grid(mesh)
        ids = grid.field_data["node_ids"]
        surf = grid.extract_surface(algorithm="dataset_surface", pass_pointid=True)
        us = u[ids][surf.point_data["vtkOriginalPointIds"]]
        # rotate the global phase so the displayed real part is as large as possible
        us = us * np.exp(-1j * np.angle(np.sum(us * us)) / 2)
        amp = np.sqrt(np.sum(np.abs(us) ** 2, axis=1))
        amax = amp.max() or 1.0
        diag = np.linalg.norm(np.ptp(mesh.points, axis=0))
        self._mode_state = {"surf": surf, "base": surf.points.copy(), "us": us, "amax": amax,
                            "factor": scale * diag / amax, "component": component, "phase": 0.0}

        style = {"cmap": "viridis", "colorbar": True, "outline": True, "title": True, **(style or {})}
        if style["outline"]:  # undeformed shape for reference
            outline = grid.extract_surface(algorithm="dataset_surface").extract_feature_edges(30)
            self.plotter.add_mesh(outline, color="#888888", line_width=1)
        if component == "|u|":
            surf.point_data["field"] = amp / amax
            clim = (0, 1)
        else:
            surf.point_data["field"] = np.zeros(surf.n_points)
            clim = (-1, 1)
        self._apply_phase(0.0, render=False)
        # VTK's font has no "|" glyph, so the amplitude is named in words
        label = "Amplitude (normalized)" if component == "|u|" else f"{component} (normalized)"
        self.plotter.add_mesh(surf, scalars="field", cmap=style["cmap"], clim=clim, smooth_shading=True,
                              show_scalar_bar=style["colorbar"], scalar_bar_args=_BAR | {"fmt": "%.1f"})
        if style["colorbar"]:
            self._bar_title(label)
        if style["title"]:
            self.plotter.add_text(vtk_text(title), position="upper_left", font_size=9, color="black")
        self._finish(reset_camera)

    def show_static(self, mesh: Mesh, u: np.ndarray, values: np.ndarray, title: str, label: str,
                    scale: float = 0.05, style: dict | None = None, fmt: str = "%.3g", reset_camera=False):
        """A stationary displacement u (N, 3; complex for a harmonic response, drawn at phase 0): the structure
        deformed (largest displacement = `scale` x the model size), coloured by nodal `values` with their true
        range; the deformation scale can then be changed with set_mode_scale."""
        self.stop_animation()
        self.plotter.clear()
        self._surface = None
        grid = _linear_grid(mesh)
        ids = grid.field_data["node_ids"]
        surf = grid.extract_surface(algorithm="dataset_surface", pass_pointid=True)
        orig = ids[surf.point_data["vtkOriginalPointIds"]]
        us = np.asarray(u[orig], dtype=complex)
        amax = float(np.sqrt(np.sum(np.abs(us) ** 2, axis=1)).max()) or 1.0
        diag = np.linalg.norm(np.ptp(mesh.points, axis=0))
        self._mode_state = {"surf": surf, "base": surf.points.copy(), "us": us, "amax": amax,
                            "factor": scale * diag / amax, "component": "|u|", "phase": 0.0}
        style = {"cmap": "viridis", "colorbar": True, "outline": True, "title": True, **(style or {})}
        if style["outline"]:
            outline = grid.extract_surface(algorithm="dataset_surface").extract_feature_edges(30)
            self.plotter.add_mesh(outline, color="#888888", line_width=1)
        surf.point_data["field"] = values[orig]
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi - lo <= 1e-9 * max(abs(hi), abs(lo), 1e-300):
            lo, hi = (lo - 1, hi + 1) if lo == 0 else (lo - 0.5 * abs(lo), hi + 0.5 * abs(hi))
        self._apply_phase(0.0, render=False)
        self.plotter.add_mesh(surf, scalars="field", cmap=style["cmap"], clim=(lo, hi), smooth_shading=True,
                              show_scalar_bar=style["colorbar"], scalar_bar_args=_BAR | {"fmt": fmt})
        if style["colorbar"]:
            self._bar_title(vtk_text(label))
        if style["title"]:
            self.plotter.add_text(vtk_text(title), position="upper_left", font_size=9, color="black")
        self._finish(reset_camera)

    def _bar_title(self, label: str):
        """Tick labels above the horizontal colour bar and the title above them, as separate text
        (VTK draws the bar's own title on top of the middle tick label)."""
        self.plotter.scalar_bar.SetTextPositionToSucceedScalarBar()
        t = self.plotter.add_text(label, position=(0.5, 0.9), viewport=True, font_size=8, color="black",
                                  font="arial")
        t.GetTextProperty().SetJustificationToCentered()

    def show_scalar(self, mesh: Mesh, values: np.ndarray, title: str, label: str, style: dict | None = None,
                    reset_camera=False):
        """A nodal scalar field (e.g. temperature) on the outer surface, coloured with a colour bar."""
        self.stop_animation()
        self._mode_state = None
        self.plotter.clear()
        self._surface = None
        style = {"cmap": "magma", "colorbar": True, "title": True, **(style or {})}
        grid = _linear_grid(mesh)
        grid.point_data["field"] = values[grid.field_data["node_ids"]]
        surf = grid.extract_surface(algorithm="dataset_surface")
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi - lo < 1e-9 * max(abs(hi), abs(lo), 1e-300):
            lo, hi = lo - 0.5, hi + 0.5
        self.plotter.add_mesh(surf, scalars="field", cmap=style["cmap"], clim=(lo, hi), smooth_shading=True,
                              show_scalar_bar=style["colorbar"], scalar_bar_args=_BAR | {"fmt": "%.4g"})
        if style["colorbar"]:
            self._bar_title(vtk_text(label))
        if style["title"]:
            self.plotter.add_text(vtk_text(title), position="upper_left", font_size=9, color="black")
        self._finish(reset_camera)

    def _apply_phase(self, phase: float, render=True):
        st = getattr(self, "_mode_state", None)
        if st is None:
            return
        st["phase"] = phase
        ur = np.real(st["us"] * np.exp(1j * phase))
        surf = st["surf"]
        surf.points[:] = st["base"] + ur * st["factor"]
        surf.GetPoints().Modified()
        if st["component"] != "|u|":
            c = "xyz".index(st["component"][1])
            surf.point_data["field"][:] = ur[:, c] / st["amax"]
            surf.GetPointData().GetArray("field").Modified()
        if render:
            self.plotter.render()

    def set_mode_scale(self, scale: float):
        st = getattr(self, "_mode_state", None)
        if st is None:
            return
        diag = np.linalg.norm(np.ptp(st["base"], axis=0))
        st["factor"] = scale * diag / st["amax"]
        self._apply_phase(st["phase"])

    def start_animation(self):
        if getattr(self, "_mode_state", None) is None:
            return
        if not hasattr(self, "_timer"):
            self._timer = QTimer(self)
            self._timer.timeout.connect(lambda: self._apply_phase(self._mode_state["phase"] + 2 * np.pi / 36))
        self._timer.start(40)

    def export_animation(self, path: str, frames_per_cycle: int = 36, fps: int = 24) -> int:
        """Write one cycle of the phase animation of the displayed mode as a looping GIF (Pillow); returns the
        frame count. All frames share one palette, so colours do not flicker."""
        from PIL import Image
        st = getattr(self, "_mode_state", None)
        if st is None:
            raise RuntimeError("no displacement field is displayed")
        running = hasattr(self, "_timer") and self._timer.isActive()
        self.stop_animation()
        start = st["phase"]
        frames = []
        try:
            for f in range(frames_per_cycle):
                self._apply_phase(start + 2 * np.pi * f / frames_per_cycle, render=False)
                self.plotter.ren_win.Render()  # synchronous: pyvistaqt's render() is deferred
                frames.append(Image.fromarray(self.plotter.screenshot(return_img=True)).convert("RGB"))
        finally:
            self._apply_phase(start)
            if running:
                self.start_animation()
        w, h = frames[0].size
        sample = Image.new("RGB", (w, h * 4))                 # palette from frames across the cycle
        for i in range(4):
            sample.paste(frames[i * len(frames) // 4], (0, h * i))
        palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0,
                       optimize=False)
        return len(images)

    def stop_animation(self):
        if hasattr(self, "_timer"):
            self._timer.stop()
