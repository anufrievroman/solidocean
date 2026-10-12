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
