"""Build a 3D geometry in gmsh/OpenCASCADE from the project feature list.

Features are applied in order, painter-style: an ``add`` feature carves its
volume out of everything before it and becomes a new domain with its own
material (later features win where they overlap); a ``subtract`` feature
removes its volume from everything before it, or only from one earlier object
or material (its ``on``). Finally all domains are
fragmented so the mesh is conforming across material interfaces.

OpenCASCADE uses absolute tolerances (~1e-8), so the model is built in a
scaled length unit ``L0`` chosen so coordinates are O(1); meshing converts
back to SI.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import gmsh
import numpy as np

from .expressions import Evaluator
from .project import AXES, Feature, Project, array_counts

_lock = threading.RLock()


@contextmanager
def gmsh_session():
    """Exclusive gmsh session (gmsh is a global singleton, not thread-safe)."""
    with _lock:
        gmsh.initialize(interruptible=False)
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            yield
        finally:
            gmsh.finalize()


@dataclass
class EvaluatedFeature:
    feature: Feature
    values: dict  # evaluated (SI) parameters


def evaluate_features(project: Project, ev: Evaluator) -> list[EvaluatedFeature]:
    out = []
    for f in project.objects:
        if not f.enabled:
            continue
        vals = {}
        for k, v in f.params.items():
            try:
                if k == "points":
                    vals[k] = np.array([[ev(c) for c in p] for p in v], dtype=float)
                elif k == "sections":
                    vals[k] = _loft_sections(v, ev)
                elif isinstance(v, list):
                    vals[k] = ev.vector(v)
                else:
                    vals[k] = ev(v)
            except Exception as exc:
                raise ValueError(f"object '{f.name}', {k}: {exc}") from exc
        out.append(EvaluatedFeature(f, vals))
    return out


def _loft_sections(sections, ev: Evaluator) -> np.ndarray:
    """Loft sections -> corners (sections, corners, xyz), SI."""
    if len(sections) < 2:
        raise ValueError("a loft needs at least 2 sections")
    out = []
    for n, sec in enumerate(sections, 1):
        pts = sec.get("points", [])
        if any(len(p) != 2 for p in pts):
            raise ValueError(f"section {n}: points need 2 coordinates (x, y)")
        if len(pts) < 3:
            raise ValueError(f"section {n}: a polygon needs at least 3 points")
        z = ev(sec.get("z", "0"))
        out.append([[ev(x), ev(y), z] for x, y in pts])
    if len({len(p) for p in out}) > 1:
        raise ValueError("all sections need the same number of points (corner i joins corner i of the next)")
    return np.array(out, dtype=float)


LENGTH_KEYS = ("size", "radius", "radius1", "radius2", "height", "base", "thickness")
MIN_RELATIVE_SIZE = 1e-5  # smaller features make OpenCASCADE/gmsh fail or hang


def _sizes(features: list[EvaluatedFeature], periods: dict[str, float]) -> list[float]:
    sizes = list(periods.values())
    for ef in features:
        v = ef.values
        for key in LENGTH_KEYS:
            if key in v:
                sizes.extend(np.abs(np.atleast_1d(v[key])).tolist())
        if "points" in v:
            sizes.append(float(np.ptp(v["points"], axis=0).max()))
        if "sections" in v:
            sizes.append(float(np.ptp(v["sections"].reshape(-1, 3), axis=0).max()))
        sizes.extend(np.abs(array_extent(ef)).tolist())
    return [s for s in sizes if s > 0]


def check_sizes(features: list[EvaluatedFeature], periods: dict[str, float]) -> None:
    """Reject features that are tiny compared to the model (zero sizes are allowed: skipped)."""
    sizes = _sizes(features, periods)
    if not sizes:
        return
    limit = MIN_RELATIVE_SIZE * max(sizes)
    for ef in features:
        for key in LENGTH_KEYS:
            if key in ef.values:
                for v in np.abs(np.atleast_1d(ef.values[key])):
                    if 0 < v < limit:
                        raise ValueError(
                            f"object '{ef.feature.name}': {key} = {v:.3g} m is too small compared to the "
                            f"model size ({max(sizes):.3g} m) — check its units")


def length_scale(features: list[EvaluatedFeature], periods: dict[str, float]) -> float:
    """Power of ten close to the characteristic size of the model."""
    sizes = _sizes(features, periods)
    if not sizes:
        return 1.0
    return 10.0 ** math.floor(math.log10(max(sizes)))


def array_shifts(ef: EvaluatedFeature) -> np.ndarray:
    """Translations (SI) of the copies of an arrayed feature, the feature itself (0, 0, 0) first."""
    v = ef.values
    if "array" not in v:
        return np.zeros((1, 3))
    counts = array_counts(v["array"], ef.feature.name)
    spacing = np.asarray(v.get("spacing", np.zeros(3)), dtype=float)
    if spacing.shape != (3,):
        raise ValueError(f"object '{ef.feature.name}': spacing needs 3 components (along x, y, z)")
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in counts], indexing="ij"), axis=-1).reshape(-1, 3)
    return idx * spacing


def array_extent(ef: EvaluatedFeature) -> np.ndarray:
    """Shift from the first to the last copy of an arrayed feature (SI, per axis)."""
    try:
        shifts = array_shifts(ef)
    except ValueError:
        return np.zeros(3)
    return shifts.max(axis=0) - shifts.min(axis=0)


def rotation_pivot(ef: EvaluatedFeature) -> np.ndarray:
    """The point a feature rotates about (SI): its center (base center for cylinders/cones); for a polygon
    prism the centroid of its corners at mid-height; for a loft the centroid of all its corners."""
    v = ef.values
    if ef.feature.type == "prism":
        return np.r_[v["points"].mean(axis=0), v["z0"] + v["height"] / 2]
    if ef.feature.type == "loft":
        return v["sections"].reshape(-1, 3).mean(axis=0)
    return np.asarray(v["center"], dtype=float)


def _make_volume(ef: EvaluatedFeature, s: float, lattice_vectors=()) -> list[tuple[int, int]]:
    """Create the OCC volume(s) for a feature, rotated if it has a rotation and repeated if it is an array (the
    copies fused into one object); coordinates divided by L0."""
    vol = _make_rotated(ef, s, lattice_vectors)
    shifts = array_shifts(ef)
    if not vol or len(shifts) == 1:
        return vol
    occ = gmsh.model.occ
    copies = list(vol)
    for d in shifts[1:]:
        c = occ.copy(vol)
        occ.translate(c, *(d * s))
        copies.extend(c)
    try:
        out, _ = occ.fuse(copies[:1], copies[1:])
    except Exception as exc:
        raise ValueError(f"object '{ef.feature.name}': joining the array copies failed in OpenCASCADE ({exc}). "
                         "Usually caused by nearly coincident faces.") from exc
    return [dt for dt in out if dt[0] == 3]


def _make_rotated(ef: EvaluatedFeature, s: float, lattice_vectors=()) -> list[tuple[int, int]]:
    """Create the OCC volume(s) for a feature, rotated if it has a rotation; coordinates divided by L0."""
    vol = _make_shape(ef, s, lattice_vectors)
    angles = np.radians(np.asarray(ef.values.get("rotation", np.zeros(3)), dtype=float))
    if vol and np.any(angles != 0):
        if angles.shape != (3,):
            raise ValueError(f"object '{ef.feature.name}': rotation needs 3 angles (about x, y, z)")
        p = rotation_pivot(ef) * s
        for axis, angle in zip(np.eye(3), angles):  # about global x, then y, then z
            if angle != 0:
                gmsh.model.occ.rotate(vol, *p, *axis, angle)
    return vol


def _make_shape(ef: EvaluatedFeature, s: float, lattice_vectors=()) -> list[tuple[int, int]]:
    """Create the OCC volume(s) for a feature; coordinates divided by L0 (s = 1/L0)."""
    occ = gmsh.model.occ
    t, v = ef.feature.type, ef.values
    if t == "cell":  # parallelogram spanned by a1, a2 (in the x-y plane), extruded along z
        from .lattice import cell_corners
        h = v["height"] * s
        if h == 0:
            return []
        inplane = [np.asarray(x) * s for x in lattice_vectors if abs(np.asarray(x)[2]) == 0]
        if len(inplane) < 2:
            raise ValueError(f"object '{ef.feature.name}': a lattice cell needs two lattice directions in the "
                             "x-y plane (periodic in x and y, or a hexagonal/custom lattice)")
        a1, a2 = inplane[:2]
        corners = cell_corners([a1, a2], v["center"] * s)
        corners[:, 2] = v["center"][2] * s - h / 2
        ptags = [occ.addPoint(*c) for c in corners]
        lines = [occ.addLine(ptags[i], ptags[(i + 1) % 4]) for i in range(4)]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        ext = occ.extrude([(2, surf)], 0, 0, h)
        return [dt for dt in ext if dt[0] == 3]
    if t == "block":
        size = np.abs(v["size"])
        if np.any(size == 0):
            return []
        corner = v["center"] - size / 2
        tag = occ.addBox(*(corner * s), *(size * s))
    elif t in ("cylinder", "cone"):
        axis = v["axis"]
        n = np.linalg.norm(axis)
        if n == 0:
            raise ValueError(f"object '{ef.feature.name}': zero axis vector")
        d = axis / n * v["height"] * s
        if v["height"] == 0:
            return []
        c = v["center"] * s
        if t == "cylinder":
            if v["radius"] <= 0:
                return []
            tag = occ.addCylinder(*c, *d, v["radius"] * s)
        else:
            if v["radius1"] <= 0 and v["radius2"] <= 0:
                return []
            tag = occ.addCone(*c, *d, v["radius1"] * s, v["radius2"] * s)
    elif t == "sphere":
        if v["radius"] <= 0:
            return []
        tag = occ.addSphere(*(v["center"] * s), v["radius"] * s)
    elif t == "triangle":
        b, h, th = abs(v["base"]) * s, abs(v["height"]) * s, abs(v["thickness"]) * s
        if b == 0 or h == 0 or th == 0:
            return []
        cx, cy, cz = v["center"] * s
        corners = [(cx - b / 2, cy - h / 2), (cx + b / 2, cy - h / 2), (cx, cy + h / 2)]
        ptags = [occ.addPoint(x, y, cz - th / 2) for x, y in corners]
        lines = [occ.addLine(ptags[i], ptags[(i + 1) % 3]) for i in range(3)]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        ext = occ.extrude([(2, surf)], 0, 0, th)
        return [dt for dt in ext if dt[0] == 3]
    elif t == "prism":
        pts = v["points"] * s
        z0, h = v["z0"] * s, v["height"] * s
        if h == 0 or len(pts) < 3:
            return []
        ptags = [occ.addPoint(x, y, z0) for x, y in pts]
        lines = [occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))]
        surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        ext = occ.extrude([(2, surf)], 0, 0, h)
        return [dt for dt in ext if dt[0] == 3]
    elif t == "loft":
        return _make_loft(ef, s)
    else:
        raise ValueError(f"unknown object type '{t}'")
    return [(3, tag)]


def _make_loft(ef: EvaluatedFeature, s: float) -> list[tuple[int, int]]:
    """Polygon sections joined by ruled (straight) side faces, OCC ThruSections."""
    occ = gmsh.model.occ
    secs = ef.values["sections"] * s
    z = secs[:, 0, 2]
    if np.any(np.diff(z) == 0):
        raise ValueError(f"object '{ef.feature.name}': consecutive loft sections need different z")

    def area(p):   # signed area of a polygon in x-y (> 0: counter-clockwise)
        x, y = p[:, 0], p[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    if any(area(p) == 0 for p in secs):
        return []
    wires = []
    for p in secs:
        if area(p) * area(secs[0]) < 0:  # same sense as the first section: corner i stays corner i
            p = np.r_[p[:1], p[:0:-1]]
        ptags = [occ.addPoint(*c) for c in p]
        lines = [occ.addLine(ptags[i], ptags[(i + 1) % len(ptags)]) for i in range(len(ptags))]
        wires.append(occ.addWire(lines))
    try:
        out = occ.addThruSections(wires, makeSolid=True, makeRuled=True)
    except Exception as exc:
        raise ValueError(f"object '{ef.feature.name}': joining the loft sections failed in OpenCASCADE ({exc}). "
                         "Check that no polygon crosses itself.") from exc
    return [dt for dt in out if dt[0] == 3]


FEATURE_TAG = 10001   # physical-group tags of objects (materials use 1, 2, ...)


def build(project: Project, ev: Evaluator) -> tuple[float, list[str]]:
    """Build the geometry in the current gmsh session.

    Creates one physical volume group per material (tag = index + 1) and one per object
    (tag = FEATURE_TAG + index, name "feature:<name>"; what is left of the feature after later ones).
    Returns (L0, material names in physical-group order).
    """
    from . import lattice as lat
    occ = gmsh.model.occ
    gmsh.model.add(project.name or "model")
    trans = lat.translations(project.lattice, ev)
    vectors = [v for _, v in trans]
    periods = {name: float(np.linalg.norm(v)) for name, v in trans}
    feats = evaluate_features(project, ev)
    check_sizes(feats, periods)
    L0 = length_scale(feats, periods)
    s = 1.0 / L0

    domains: list[tuple[list[tuple[int, int]], str, str]] = []   # (volumes, material, feature name)
    for ef in feats:
        vol = _make_volume(ef, s, vectors)
        if not vol:
            continue
        target = ef.feature.on if ef.feature.op == "subtract" else ""
        new_domains = []
        for tags, material, fname in domains:
            if tags and (not target or target in (fname, material)):
                try:
                    out, _ = occ.cut(tags, vol, removeObject=True, removeTool=False)
                except Exception as exc:
                    raise ValueError(
                        f"object '{ef.feature.name}': {ef.feature.op} failed in OpenCASCADE ({exc}). "
                        "Usually caused by nearly coincident faces or wrong units.") from exc
                tags = [dt for dt in out if dt[0] == 3]
            if tags:
                new_domains.append((tags, material, fname))
        domains = new_domains
        if ef.feature.op == "add":
            domains.append((vol, ef.feature.material, ef.feature.name))
        else:
            occ.remove(vol, recursive=True)

    all_tags = [dt for tags, _, _ in domains for dt in tags]
    if not all_tags:
        raise ValueError("geometry is empty")
    owner = [m for tags, m, _ in domains for _ in tags]
    owner_feature = [f for tags, _, f in domains for _ in tags]
    if len(all_tags) > 1:
        try:
            _, out_map = occ.fragment(all_tags, [])
        except Exception as exc:
            raise ValueError(f"geometry: joining material domains failed in OpenCASCADE ({exc})") from exc
    else:
        out_map = [all_tags]
    occ.synchronize()

    names: list[str] = []
    by_material: dict[str, list[int]] = {}
    for mat_name, outs in zip(owner, out_map):
        by_material.setdefault(mat_name, []).extend(t for d, t in outs if d == 3)
    for i, (mat_name, tags) in enumerate(by_material.items()):
        gmsh.model.addPhysicalGroup(3, sorted(set(tags)), tag=i + 1, name=mat_name)
        names.append(mat_name)
    by_feature: dict[str, list[int]] = {}
    for fname, outs in zip(owner_feature, out_map):
        by_feature.setdefault(fname, []).extend(t for d, t in outs if d == 3)
    for j, (fname, tags) in enumerate(by_feature.items()):
        gmsh.model.addPhysicalGroup(3, sorted(set(tags)), tag=FEATURE_TAG + j, name=f"feature:{fname}")

    if project.lattice.vectors:
        _set_periodic_vectors(trans, s)
    else:
        _set_periodic({ax: ev(p) for ax, p in project.lattice.periodic.items()}, s)
    return L0, names


def _set_periodic(periods: dict[str, float], s: float) -> None:
    """Make the meshes on opposite faces of the unit cell identical."""
    if not periods:
        return
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    lo, hi = np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])
    eps = 1e-6 * float(np.max(hi - lo))
    for ax, period in periods.items():
        i = AXES.index(ax)
        L = period * s
        if abs((hi[i] - lo[i]) - L) > max(eps, 1e-6 * max(L, 1e-30)):
            raise ValueError(
                f"periodic along {ax}: the geometry is {(hi[i] - lo[i]) / s * 1e6:.4g} µm long in {ax} but the "
                f"period is {period * 1e6:.4g} µm. Make the block size along {ax} equal to the period, or use a "
                f"'Lattice cell' object, which always matches the lattice.")
        shift = np.zeros(3)
        shift[i] = L
        box_lo, box_hi = lo - eps, hi + eps
        box_hi[i] = lo[i] + eps
        masters = gmsh.model.getEntitiesInBoundingBox(*box_lo, *box_hi, 2)
        box_lo[i], box_hi[i] = hi[i] - eps, hi[i] + eps
        slaves = gmsh.model.getEntitiesInBoundingBox(*box_lo, *box_hi, 2)
        affine = [1, 0, 0, shift[0], 0, 1, 0, shift[1], 0, 0, 1, shift[2], 0, 0, 0, 1]
        unmatched = []
        for _, m in masters:
            mb = np.array(gmsh.model.getBoundingBox(2, m)).reshape(2, 3) + shift
            match = None
            for _, sl in slaves:
                sb = np.array(gmsh.model.getBoundingBox(2, sl)).reshape(2, 3)
                if np.all(np.abs(sb - mb) < 10 * eps):
                    match = sl
                    break
            if match is None:
                unmatched.append(m)
            else:
                gmsh.model.mesh.setPeriodic(2, [match], [m], affine)
        if unmatched or len(masters) != len(slaves):
            raise ValueError(f"geometry is not periodic along {ax}: opposite faces do not match")


def _set_periodic_vectors(trans: list[tuple[str, np.ndarray]], s: float) -> None:
    """Periodic meshes for arbitrary lattice vectors: pair every boundary face S with the face
    S + a_i (same bounding box shifted by a_i)."""
    surfaces = [t for _, t in gmsh.model.getEntities(2)]
    boxes = {t: np.array(gmsh.model.getBoundingBox(2, t)).reshape(2, 3) for t in surfaces}
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    eps = 1e-6 * float(np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin]))
    for name, vec in trans:
        t = np.asarray(vec) * s
        affine = [1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1]
        pairs = 0
        for m in surfaces:
            target = boxes[m] + t
            for sl in surfaces:
                if sl != m and np.all(np.abs(boxes[sl] - target) < 10 * eps):
                    gmsh.model.mesh.setPeriodic(2, [sl], [m], affine)
                    pairs += 1
                    break
        if pairs == 0:
            raise ValueError(f"geometry is not periodic along {name}: no face maps onto another by {name}")
