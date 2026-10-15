"""Bravais lattices: the periodicity of a model ([lattice] in the model file), translation vectors,
reciprocal vectors, presets and k-paths.

A periodic model is described either by the axis shorthand
``periodic = {x = "a", y = "a"}`` (rectangular cells, periods along x, y, z)
or by explicit lattice vectors ``vectors = [[...], [...]]`` for any Bravais
lattice (hexagonal, oblique, ...). Internally both become up to three
"slots" a1, a2, a3; k-path points are given in reduced coordinates of the
matching reciprocal vectors b1, b2, b3 (b_i . a_j = 2 pi delta_ij).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .expressions import Evaluator

AXES = ("x", "y", "z")

# lattice presets (GUI): vectors as expressions of the lattice constant `a`
PRESETS: dict[str, list[list[str]]] = {
    "square": [["a", "0", "0"], ["0", "a", "0"]],
    "hexagonal": [["a", "0", "0"], ["a/2", "sqrt(3)*a/2", "0"]],
}

# standard Brillouin-zone paths, reduced coordinates (fractions of b1, b2, b3)
KPATHS: dict[str, tuple[dict[str, list[float]], list[str]]] = {
    "1d": ({"G": [0, 0, 0], "X": [0.5, 0, 0]}, ["G", "X"]),
    "square": ({"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0]}, ["G", "X", "M", "G"]),
    "rectangular": ({"G": [0, 0, 0], "X": [0.5, 0, 0], "S": [0.5, 0.5, 0], "Y": [0, 0.5, 0]},
                    ["G", "X", "S", "Y", "G"]),
    # a1, a2 at 60 degrees -> b1, b2 at 120 degrees: K (next to M = b1/2) is (2 b1 + b2) / 3
    "hexagonal": ({"G": [0, 0, 0], "M": [0.5, 0, 0], "K": [2 / 3, 1 / 3, 0]}, ["G", "M", "K", "G"]),
    "oblique": ({"G": [0, 0, 0], "X": [0.5, 0, 0], "Y": [0, 0.5, 0]}, ["G", "X", "Y", "G"]),
    "cubic": ({"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "R": [0.5, 0.5, 0.5]},
              ["G", "X", "M", "G", "R", "X"]),
}


@dataclass
class Lattice:
    """Periodicity of the structure, shared by every physics: it shapes the geometry (the `cell` feature) and
    solid mechanics makes the matching faces Bloch-periodic. Either the axis shorthand `periodic`
    ({"x": "a", "y": "a"}: rectangular cell = the bounding box) or lattice vectors a1[, a2[, a3]] for any
    Bravais lattice, e.g. hexagonal [["a", "0", "0"], ["a/2", "sqrt(3)*a/2", "0"]]."""
    periodic: dict = field(default_factory=dict)
    vectors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """[lattice]: the period per periodic axis (x = "a"), or `vectors`."""
        d = dict(self.periodic)
        if self.vectors:
            d["vectors"] = [list(v) for v in self.vectors]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Lattice":
        d = dict(d)
        vectors = d.pop("vectors", [])
        unknown = set(d) - set(AXES)
        if unknown:
            raise ValueError(f"[lattice]: unknown key(s) {', '.join(sorted(unknown))} (use x, y, z or vectors)")
        return cls(periodic=d, vectors=vectors)

    @property
    def is_periodic(self) -> bool:
        return bool(self.periodic or self.vectors)

    def periodic_axes(self) -> set[str]:
        """Axes along which bounding-box faces are periodic (no other condition allowed there)."""
        if not self.vectors:
            return set(self.periodic)
        return {ax for i, ax in enumerate(AXES)
                if any(str(v[i]).strip() not in ("0", "0.0", "") for v in self.vectors)}


def slots(lattice: Lattice, ev: Evaluator) -> list[np.ndarray | None]:
    """Lattice vectors a1, a2, a3 in SI (None where the structure is not periodic)."""
    if lattice.vectors:
        out = [ev.vector(v) for v in lattice.vectors]
        return out + [None] * (3 - len(out))
    out: list[np.ndarray | None] = []
    for i, ax in enumerate(AXES):
        if ax in lattice.periodic:
            v = np.zeros(3)
            v[i] = ev(lattice.periodic[ax])
            out.append(v)
        else:
            out.append(None)
    return out


def translations(lattice: Lattice, ev: Evaluator) -> list[tuple[str, np.ndarray]]:
    """(label, vector) for every periodic direction: 'x'/'y'/'z' (axis shorthand) or 'a1'.."""
    names = ["a1", "a2", "a3"] if lattice.vectors else list(AXES)
    return [(n, v) for n, v in zip(names, slots(lattice, ev)) if v is not None]


def reciprocal(vectors: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Reciprocal vectors within the span of the given ones: b_i . a_j = 2 pi delta_ij."""
    used = [i for i, v in enumerate(vectors) if v is not None]
    out: list[np.ndarray | None] = [None] * len(vectors)
    if not used:
        return out
    A = np.array([vectors[i] for i in used])              # (n, 3)
    B = 2 * np.pi * np.linalg.solve(A @ A.T, A)            # rows b_i, in span(A)
    for n, i in enumerate(used):
        out[i] = B[n]
    return out


def check(vectors: list[np.ndarray | None]) -> str | None:
    """Return an error message if the lattice vectors are degenerate."""
    used = [v for v in vectors if v is not None]
    if not used:
        return None
    A = np.array(used)
    if np.linalg.matrix_rank(A, tol=1e-9 * np.abs(A).max()) < len(used):
        return "lattice vectors are linearly dependent"
    return None


def classify(lattice: Lattice, ev: Evaluator) -> str:
    """Name of the standard k-path that suits the lattice."""
    vecs = [v for v in slots(lattice, ev) if v is not None]
    if len(vecs) <= 1:
        return "1d"
    if len(vecs) == 3:
        return "cubic"
    a1, a2 = vecs[:2]
    l1, l2 = np.linalg.norm(a1), np.linalg.norm(a2)
    cos = float(a1 @ a2 / (l1 * l2))
    if abs(l1 - l2) < 1e-9 * l1 and abs(abs(cos) - 0.5) < 1e-9:
        return "hexagonal"
    if abs(cos) < 1e-9:
        return "square" if abs(l1 - l2) < 1e-9 * l1 else "rectangular"
    return "oblique"


def cell_corners(vectors: list[np.ndarray], center: np.ndarray) -> np.ndarray:
    """Corners of the parallelogram spanned by two in-plane vectors, centred on `center`."""
    a1, a2 = vectors
    origin = center - (a1 + a2) / 2
    return np.array([origin, origin + a1, origin + a1 + a2, origin + a2])


def brillouin_zone_2d(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Polygon (n, 2) of the first Brillouin zone spanned by in-plane reciprocal vectors."""
    from scipy.spatial import Voronoi
    pts = np.array([(i * b1 + j * b2)[:2] for i in range(-2, 3) for j in range(-2, 3)])
    vor = Voronoi(pts)
    centre = int(np.argmin(np.linalg.norm(pts, axis=1)))
    region = vor.regions[vor.point_region[centre]]
    poly = vor.vertices[region]
    angle = np.arctan2(poly[:, 1], poly[:, 0])
    return poly[np.argsort(angle)]


# Corners of the irreducible wedge of the 2D Brillouin zone, in reduced coordinates of b1, b2: the standard
# high-symmetry points of each lattice. Meshing the wedge instead of the whole zone assumes the contents of
# the unit cell have the full point symmetry of the lattice (true for an empty cell or a centred hole, not
# for a tilted or off-centre inclusion), so the zone can be filled in from it by symmetry.
WEDGES: dict[str, list[list[float]]] = {
    "square": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],                   # G, X, M: an eighth of the zone
    "rectangular": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],  # G, X, S, Y: a quarter
    "hexagonal": [[0.0, 0.0], [0.5, 0.0], [2 / 3, 1 / 3]],            # G, M, K: a twelfth
}
WEDGE_LABELS: dict[str, tuple[str, ...]] = {"square": ("G", "X", "M"), "rectangular": ("G", "X", "S", "Y"),
                                            "hexagonal": ("G", "M", "K")}
ZONE_REGIONS = ("irreducible", "full")


def zone_mesh(b1: np.ndarray, b2: np.ndarray, n: int, region: str = "irreducible",
              kind: str = "square") -> tuple[np.ndarray, np.ndarray]:
    """Uniform triangular mesh of the first Brillouin zone (`region` "full") or of its irreducible wedge:
    points (P, 2) in reduced coordinates of b1, b2 and triangles (T, 3) of indices into them.

    Every edge of the region is divided into `n`, so a triangular wedge holds (n + 1)(n + 2) / 2 points.
    The mesh is built in reduced coordinates, which depend on the shape of the lattice but not its size: a
    sweep of the lattice constant keeps the same points and the same connectivity.
    """
    if n < 1:
        raise ValueError("a Brillouin-zone mesh needs at least one division per edge")
    if region == "full":
        corners, apex = _zone_polygon(b1, b2), np.zeros(2)
    elif kind in WEDGES:
        corners, apex = np.array(WEDGES[kind], dtype=float), None
    else:
        raise ValueError(f"no standard irreducible wedge for a {kind} lattice (use the whole zone)")
    return _mesh_polygon(corners, n, apex)


def _zone_polygon(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """The first Brillouin zone as a polygon (n, 2) in reduced coordinates of b1, b2 (its shape in reduced
    coordinates follows from the metric of the reciprocal vectors, not from the plane they lie in)."""
    from scipy.spatial import Voronoi
    gram = np.array([[b1 @ b1, b1 @ b2], [b1 @ b2, b2 @ b2]])
    chol = np.linalg.cholesky(gram)                 # reduced r -> Cartesian x = chol.T @ r, in some frame
    reduced = np.array([[i, j] for i in range(-2, 3) for j in range(-2, 3)], dtype=float)
    vor = Voronoi(reduced @ chol)
    centre = int(np.argmin(np.linalg.norm(reduced, axis=1)))
    poly = vor.vertices[vor.regions[vor.point_region[centre]]]
    poly = poly[np.argsort(np.arctan2(poly[:, 1], poly[:, 0]))]
    return np.linalg.solve(chol.T, poly.T).T


def _mesh_polygon(corners: np.ndarray, n: int, apex: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Uniform triangular mesh of a convex polygon: fanned into triangles from `apex` (by default the first
    corner, which is then not an interior point), each divided into n**2 similar triangles, shared points
    merged. The first point of the mesh is the apex."""
    corners = np.asarray(corners, dtype=float)
    if apex is None:
        apex, ring, wrap = corners[0], corners[1:], False     # a corner of the polygon: open fan
    else:
        apex, ring, wrap = np.asarray(apex, dtype=float), corners, True   # inside it: fan all the way round
    points, tris = [], []
    m = len(ring)
    for s in range(m if wrap else m - 1):
        p, t = _subdivide(apex, ring[s], ring[(s + 1) % m], n)
        tris.append(t + sum(len(q) for q in points))
        points.append(p)
    return _merge(np.vstack(points), np.vstack(tris))


def _subdivide(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A triangle divided into n**2 similar triangles: its points and their triangles."""
    index, points = {}, []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            index[(i, j)] = len(points)
            points.append(a + (b - a) * (i / n) + (c - a) * (j / n))
    tris = []
    for i in range(n):
        for j in range(n - i):
            tris.append((index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]))
            if i + j < n - 1:
                tris.append((index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]))
    return np.array(points), np.array(tris, dtype=np.int64)


def _merge(points: np.ndarray, tris: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Merge points that coincide (the fan triangles share edges), keeping the order of first appearance."""
    _, first, inverse = np.unique(np.round(points / tol).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    order = np.argsort(first)
    remap = np.empty(len(first), dtype=np.int64)
    remap[order] = np.arange(len(first))
    return points[first[order]], remap[inverse.ravel()][tris]
