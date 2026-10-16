"""Box zoom in the 3D view: dragging a rectangle zooms into it from whatever view the user is looking at."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, Qt
from PySide6.QtGui import QMouseEvent

from solidocean import meshing, templates


@pytest.fixture
def view():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from solidocean.gui.viewer import Viewer
    v = Viewer()
    v.resize(800, 600)
    v.show()
    p = templates.create("square")
    v.show_geometry(meshing.preview_geometry(p, p.evaluator()))
    app.processEvents()
    yield v, app
    v.close()


def in_view(v, point) -> np.ndarray:
    """Where a world point sits in the view, from VTK's own camera transform: (0, 0) is the centre and the
    edges are at ±1 (so this does not repeat the arithmetic it is checking)."""
    ren = v.plotter.renderer
    ren.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
    ren.WorldToView()
    return np.array(ren.GetViewPoint()[:2])


def on_focal_plane(v, x, y):
    """The world point the pixel (x, y) of the widget points at, on the camera's focal plane."""
    cam = v.plotter.camera
    w, h = v.plotter.interactor.width(), v.plotter.interactor.height()
    position, focus, up = np.array(cam.position), np.array(cam.focal_point), np.array(cam.up)
    look = focus - position
    distance = np.linalg.norm(look)
    look = look / distance
    right = np.cross(look, up)
    right = right / np.linalg.norm(right)
    half_h = cam.parallel_scale if cam.parallel_projection else distance * np.tan(np.radians(cam.view_angle) / 2)
    return focus + (x - w / 2) / (w / 2) * half_h * w / h * right \
                 + (h / 2 - y) / (h / 2) * half_h * np.cross(right, look)


def camera_state(v):
    cam = v.plotter.camera
    position, focus = np.array(cam.position), np.array(cam.focal_point)
    distance = np.linalg.norm(focus - position)
    return (focus - position) / distance, np.array(cam.up), distance, cam.view_angle


def move_the_view(v):
    """Rotate, zoom and pan, so the camera is nowhere near where it started."""
    v.default_view()
    v.plotter.camera.azimuth = 37
    v.plotter.camera.elevation = 21
    v._zoom(1.7)
    position, focus = np.array(v.plotter.camera.position), np.array(v.plotter.camera.focal_point)
    shift = 0.3 * np.linalg.norm(focus - position) * np.array([0.9, 0.36, 0.18])
    v.plotter.camera.position = tuple(position + shift)
    v.plotter.camera.focal_point = tuple(focus + shift)


@pytest.mark.parametrize("moved", [False, True], ids=["fresh view", "rotated, zoomed and panned"])
def test_the_box_lands_in_the_middle_of_the_view(view, moved):
    """The regression: zooming to a box worked from the view the model opened in, but once the view had
    been rotated, zoomed or panned, it jumped somewhere else — VTK's rubber-band style rebuilds the camera
    from the model's bounds instead of moving the current one."""
    v, app = view
    if moved:
        move_the_view(v)
    box = QRect(QPoint(150, 120), QPoint(450, 330))
    target = on_focal_plane(v, box.center().x(), box.center().y())
    before = camera_state(v)
    v._zoom_to_box(box)
    assert in_view(v, target) == pytest.approx([0, 0], abs=1e-6)     # the box centre is now the view centre
    after = camera_state(v)
    assert after[0] == pytest.approx(before[0], abs=1e-9)           # looking the same way,
    assert after[1] == pytest.approx(before[1], abs=1e-9)           # the same way up,
    assert after[2] == pytest.approx(before[2], rel=1e-9)           # from the same distance
    assert after[3] < before[3]                                     # through a narrower field of view


def test_the_box_fills_the_view_in_its_tighter_direction(view):
    """The zoom fits the box, keeping the aspect ratio: the narrower side of the box ends up at the edge
    of the view and nothing that was in the box is cut off."""
    v, app = view
    move_the_view(v)
    w, h = v.plotter.interactor.width(), v.plotter.interactor.height()
    box = QRect(QPoint(200, 150), QPoint(500, 360))
    angle = v.plotter.camera.view_angle
    v._zoom_to_box(box)
    assert v.plotter.camera.view_angle == pytest.approx(angle / min(w / box.width(), h / box.height()))


def test_a_click_is_not_a_zoom(view):
    """A click, or a sliver of a drag, leaves the view alone rather than zooming in enormously."""
    v, app = view
    move_the_view(v)
    before = (tuple(v.plotter.camera.position), tuple(v.plotter.camera.focal_point), v.plotter.camera.view_angle)
    v._zoom_to_box(QRect(QPoint(300, 300), QPoint(301, 302)))
    assert (tuple(v.plotter.camera.position), tuple(v.plotter.camera.focal_point),
            v.plotter.camera.view_angle) == before


def drag(v, app, start: QPoint, end: QPoint):
    """A left-button drag over the 3D view, as Qt delivers it."""
    target = v.plotter.interactor
    for kind, at, buttons in ((QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton),
                              (QEvent.Type.MouseMove, end, Qt.MouseButton.LeftButton),
                              (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton)):
        # the overload without a QPointingDevice is deprecated, but passing None for one crashes Qt
        app.sendEvent(target, QMouseEvent(kind, QPointF(at), Qt.MouseButton.LeftButton, buttons,
                                          Qt.KeyboardModifier.NoModifier))
        app.processEvents()


def test_a_drag_zooms_once_and_then_the_mouse_rotates_again(view):
    """Turning box zoom on arms it for one drag: that drag zooms (and rotates nothing), and the button
    switches itself off afterwards."""
    v, app = view
    move_the_view(v)
    direction, up, distance, angle = camera_state(v)
    v.box_zoom_action.setChecked(True)
    assert v.box_zoom_action.isChecked()
    drag(v, app, QPoint(200, 150), QPoint(500, 360))
    assert not v.box_zoom_action.isChecked()            # one shot
    assert not v._band.isVisible() and v._band_origin is None
    after = camera_state(v)
    assert after[3] < angle                             # it zoomed
    assert after[0] == pytest.approx(direction, abs=1e-9)   # and did not rotate on the way
    rotated = camera_state(v)
    drag(v, app, QPoint(200, 150), QPoint(500, 360))    # the same drag again, now plain trackball
    assert camera_state(v)[3] == rotated[3]             # no second zoom


def test_it_also_works_in_parallel_projection(view):
    """The view is perspective, but the zoom handles a parallel camera too: there the scale takes the place
    of the field of view."""
    v, app = view
    v.plotter.enable_parallel_projection()
    move_the_view(v)
    w, h = v.plotter.interactor.width(), v.plotter.interactor.height()
    box = QRect(QPoint(200, 150), QPoint(500, 360))
    target = on_focal_plane(v, box.center().x(), box.center().y())
    scale = v.plotter.camera.parallel_scale
    v._zoom_to_box(box)
    assert in_view(v, target) == pytest.approx([0, 0], abs=1e-6)
    assert v.plotter.camera.parallel_scale == pytest.approx(scale / min(w / box.width(), h / box.height()))
