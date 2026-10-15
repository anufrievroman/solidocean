"""Derived quantities: scalar expressions over the fields of a solution, the generic engine.

A quantity is one number per solution (mode, time step, variable value), e.g. ``freq``, ``max(T)`` or
``mean(abs(uz)^2)``. Field variables live on the mesh and must be reduced with ``mean()``, ``integral()``,
``max()`` or ``min()``; integrals use the solver's quadrature, so they are exact for the finite-element
field. A second argument restricts a reduction to one object, one of the objects or a material:
``mean(T, "pad")``, ``integral(qz, "Au")``, or to part of the outer boundary, named like the selections of
conditions: ``"Film:zmax"`` (the face of the object's bounding box), ``"Film:all"`` (its whole outer surface),
``"Pad:xmin,xmax"``, or without an object the model's own ``"zmax"`` / ``"all"``. Over a surface, mean is the
area average and integral the surface integral, and the outward unit normal nx, ny, nz is a field too, so
``integral(qx*nx + qy*ny + qz*nz, "Film:zmax")`` is the heat leaving through that face. A name that is an
object or a material always means its volume. Line plots evaluate expressions point by point instead.

The engine knows only the coordinates x, y, z. Each physics defines its own field and scalar names
(PhysicsSpec.field_names / scalar_names) and passes the fields of a solution as a mapping: name -> array
at the quadrature (or sample) points, or name -> provider, a function returning a dict of several fields
that is called on first use (e.g. all strains and stresses from one gradient evaluation).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .expressions import CONSTANTS, ExpressionError, _parse, _fix_if
from .fem.elements import default_quadrature, shape_functions

COORDINATES = ("x", "y", "z")
NORMALS = ("nx", "ny", "nz")     # outward unit normal, on a surface only
SURFACE_HELP = ("; over a surface: mean(f, \"Film:zmax\") (a face of the object's bounding box), \"Film:all\" "
                "(its outer surface), \"zmax\" or \"all\" (of the model); nx, ny, nz: the outward normal there")
REDUCTIONS = ("mean", "integral", "max", "min")


def field_names() -> tuple[str, ...]:
    """Every field name: the coordinates and the fields of all registered physics."""
    from .physics import all_field_names
    return COORDINATES + NORMALS + all_field_names()


def scalar_names() -> tuple[str, ...]:
    from .physics import all_scalar_names
    return all_scalar_names()


_FUNCS = {
    "abs": np.abs, "real": np.real, "imag": np.imag, "conj": np.conj, "angle": np.angle,
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "floor": np.floor, "ceil": np.ceil,
}


_BINOPS = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
    ast.Pow: np.power, ast.Mod: np.mod,
}


_CMPOPS = {
    ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
}


def _clean(expr: str) -> str:
    """Expressions may span several lines in the GUI; line breaks are just spaces."""
    return " ".join(str(expr).split())


def uses_fields(expr: str, names=None) -> bool:
    """True if the expression needs field variables (a mesh and a solution), not only scalars."""
    names = set(field_names() if names is None else names)
    tree = _parse(_fix_if(_clean(expr)))
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(tree))


def check(expr: str, variables: Mapping[str, float], fields=None, scalars=None) -> None:
    """Raise ExpressionError if the expression uses unknown names or functions. fields / scalars: the names a
    physics provides (default: those of every physics)."""
    tree = _parse(_fix_if(_clean(expr)))
    known = set(COORDINATES) | set(NORMALS) | set(field_names() if fields is None else fields) \
        | set(scalar_names() if scalars is None else scalars) | set(CONSTANTS) | set(variables)
    funcs = set(_FUNCS) | set(REDUCTIONS) | {"if_", "if"}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Name) and n.func.id in funcs):
                raise ExpressionError(f"unknown function '{ast.unparse(n.func)}'")
            strings = [a for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if strings and (n.func.id not in REDUCTIONS or len(n.args) != 2 or n.args[1] is not strings[0]):
                raise ExpressionError("an object or surface name in quotes may only be the second argument of "
                                      "mean, integral, max or min")
        elif isinstance(n, ast.Name) and n.id not in known and n.id not in funcs:
            raise ExpressionError(f"unknown name '{n.id}'")


@dataclass
class Quadrature:
    """Per-element quadrature data of a mesh: weights (E, Q) and shape values (Q, nn)."""
    weights: np.ndarray
    N: np.ndarray
    points: np.ndarray     # (E, Q, 3) physical quadrature points
    cells: np.ndarray      # (E, nn)
    volume: float
    X: np.ndarray | None = None    # (E, nn, 3) element node coordinates (for gradients)
    dN: np.ndarray | None = None   # (Q, nn, 3) reference shape gradients
    mesh: object = None            # for object (feature / material) masks
    _G: list | None = None         # cached physical shape gradients per quadrature point
    _boundary: tuple | None = None     # cached outer boundary: (triangles, owner elements, local faces)
    _surfaces: dict | None = None      # cached SurfaceQuadrature per name

    @property
    def elements(self) -> np.ndarray:
        """The element each row of points belongs to (fields that need per-element properties)."""
        return np.arange(len(self.weights))

    @classmethod
    def of(cls, mesh) -> "Quadrature":
        qp, qw = default_quadrature(mesh.order)
        N, dN = shape_functions(mesh.order, qp)
        X = mesh.points[mesh.cells]                       # (E, nn, 3)
        J = np.einsum("eai,qaj->eqij", X, dN)             # (E, Q, 3, 3)
        w = np.abs(np.linalg.det(J)) * qw[None, :]
        pts = np.einsum("qa,eai->eqi", N, X)
        return cls(w, N, pts, mesh.cells, float(w.sum()), X, dN, mesh)

    def grad_at_points(self, nodal: np.ndarray) -> np.ndarray:
        """Gradient of a scalar nodal field at the quadrature points: (E, Q, 3)."""
        if self._G is None:  # computed once per mesh: transient problems evaluate many time steps
            self._G = []
            for q in range(len(self.dN)):
                J = np.einsum("eai,aj->eij", self.X, self.dN[q])
                self._G.append(np.einsum("aj,eji->eai", self.dN[q], np.linalg.inv(J)))
        Te = nodal[self.cells]                                   # (E, nn)
        out = np.empty(self.points.shape, dtype=np.result_type(nodal.dtype, float))
        for q, G in enumerate(self._G):
            out[:, q] = np.einsum("ea,eai->ei", Te, G)
        return out

    def region(self, name: str) -> np.ndarray:
        if self.mesh is None:
            raise ExpressionError("object selection needs a mesh")
        try:
            return self.mesh.region(name)
        except ValueError as exc:
            raise ExpressionError(str(exc)) from None

    def at_points(self, nodal: np.ndarray) -> np.ndarray:
        """Interpolate nodal values (N, ...) to the quadrature points (E, Q, ...)."""
        return np.einsum("qa,ea...->eq...", self.N, nodal[self.cells])

    def surface(self, name: str) -> "SurfaceQuadrature | None":
        """The part of the outer boundary a reduction's second argument names, or None if it names an object or
        a material (their volume)."""
        if self.mesh is None:
            raise ExpressionError("a surface needs a mesh")
        if self._surfaces is None:
            self._surfaces = {}
        if name not in self._surfaces:
            where = _surface_name(name, self.mesh)
            self._surfaces[name] = None if where is None else SurfaceQuadrature.of(self, name, *where)
        return self._surfaces[name]

    def boundary(self) -> tuple:
        """Outer boundary triangles, their elements and which face of them they are. Faces are matched by the
        positions of their nodes, so the nodes split at a thermal contact do not turn it into boundary."""
        if self._boundary is None:
            from .fem import surface
            mesh = self.mesh
            _, position = np.unique(mesh.points, axis=0, return_inverse=True)
            _, owner, local = surface.boundary_faces(position.reshape(-1)[mesh.cells], mesh.order, local=True)
            tris = np.take_along_axis(mesh.cells[owner], surface._face_locals(mesh.order)[local], axis=1)
            self._boundary = (tris, owner, local)
        return self._boundary


def _surface_name(name: str, mesh) -> tuple[str, list[str]] | None:
    """(object, faces) of a boundary named in a reduction, or None for an object or a material."""
    from .selection import ALL_FACES, FACES
    on, colon, faces = name.rpartition(":")
    if not colon:
        try:
            mesh.region(name)
            return None
        except ValueError:
            pass
    faces = [f.strip() for f in faces.split(",")]
    if not all(f in FACES + (ALL_FACES,) for f in faces):
        if not colon:
            return None          # neither an object nor faces: the object lookup reports it
        raise ExpressionError(f"bad face in '{name}' (use {', '.join(FACES)} or {ALL_FACES}, e.g. \"Film:zmax\")")
    return on.strip(), faces


_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)   # reference tet vertices


@dataclass
class SurfaceQuadrature:
    """Quadrature over boundary triangles, with the interface of Quadrature so a physics computes its fields
    there as in the volume: each point lies on a face of its element, where the element's own shape
    functions (and gradients) are evaluated. weights carry the area element; volume is the area."""
    weights: np.ndarray            # (F, Q)
    points: np.ndarray             # (F, Q, 3)
    normals: np.ndarray            # (F, Q, 3) outward unit normals
    N: np.ndarray                  # (F, Q, nn) shape values of the owner element
    dN: np.ndarray                 # (F, Q, nn, 3) reference shape gradients
    J: np.ndarray                  # (F, Q, 3, 3) Jacobians
    cells: np.ndarray              # (F, nn) owner element nodes
    elements: np.ndarray           # (F,) owner elements
    volume: float
    mesh: object = None
    _G: np.ndarray | None = None

    @classmethod
    def of(cls, quad: Quadrature, name: str, on: str, faces: list[str]) -> "SurfaceQuadrature":
        from types import SimpleNamespace
        from . import selection
        from .fem import surface
        mesh = quad.mesh
        tris, owner, local = quad.boundary()
        try:
            chosen = selection.select(mesh, selection.Boundary(tris, owner),
                                      SimpleNamespace(on=on, faces=faces, box=None), None)
        except ValueError as exc:
            raise ExpressionError(str(exc)) from None
        if len(chosen) == 0:
            raise ExpressionError(f"'{name}' selects no outer boundary (an object's surface excludes its "
                                  "interfaces with other objects)")
        owner, local = owner[chosen], local[chosen]
        tp, tw = surface._tri_rule(3)
        C, k = _CORNERS, np.arange(4)
        face = np.array(surface._FACES)                                  # (4, 3): face k lacks vertex k
        L = np.column_stack([1 - tp.sum(axis=1), tp])                    # (Q, 3)
        xi = np.einsum("qc,kci->kqi", L, C[face])                        # (4, Q, 3) reference points
        shapes = [shape_functions(mesh.order, xi[j]) for j in k]
        N = np.stack([n for n, _ in shapes])[local]                      # (F, Q, nn)
        dN = np.stack([d for _, d in shapes])[local]                     # (F, Q, nn, 3)
        cells = mesh.cells[owner]
        X = mesh.points[cells]                                           # (F, nn, 3)
        J = np.einsum("fai,fqaj->fqij", X, dN)
        tu = np.einsum("fqij,fj->fqi", J, (C[face[:, 1]] - C[face[:, 0]])[local])
        tv = np.einsum("fqij,fj->fqi", J, (C[face[:, 2]] - C[face[:, 0]])[local])
        n = np.cross(tu, tv)
        area = np.linalg.norm(n, axis=-1)
        inward = np.einsum("fqij,fqj->fqi", J, C[local][:, None] - xi[local])   # towards the missing vertex
        n *= np.where(np.einsum("fqi,fqi->fq", n, inward) > 0, -1.0, 1.0)[..., None] / area[..., None]
        w = area * tw[None]
        pts = np.einsum("fqa,fai->fqi", N, X)
        return cls(w, pts, n, N, dN, J, cells, owner, float(w.sum()), mesh)

    def at_points(self, nodal: np.ndarray) -> np.ndarray:
        return np.einsum("fqa,fa...->fq...", self.N, nodal[self.cells])

    def grad_at_points(self, nodal: np.ndarray) -> np.ndarray:
        if self._G is None:
            self._G = np.einsum("fqaj,fqji->fqai", self.dN, np.linalg.inv(self.J))
        return np.einsum("fa,fqai->fqi", nodal[self.cells], self._G)

    def region(self, name: str) -> np.ndarray:
        try:
            return self.mesh.region(name)[self.elements]
        except ValueError as exc:
            raise ExpressionError(str(exc)) from None


class PointSampler:
    """Fields at arbitrary points (e.g. along a line): each point is located in its element and nodal values
    are interpolated with that element's shape functions. Points outside the structure (holes, outside the
    model) give NaN."""

    def __init__(self, mesh, pts: np.ndarray, tol: float = 1e-7):
        from scipy.spatial import cKDTree
        self.mesh, self.pts = mesh, np.asarray(pts, dtype=float)
        X = mesh.points[mesh.cells[:, :4]]                       # corner nodes (E, 4, 3)
        self._X0 = X[:, 0]
        self._Jinv = np.linalg.inv(np.transpose(X[:, 1:] - X[:, :1], (0, 2, 1)))   # x - X0 = J xi
        n = len(self.pts)
        self.elem = np.full(n, -1)
        self.xi = np.zeros((n, 3))
        k = min(32, len(X))
        _, cand = cKDTree(X.mean(axis=1)).query(self.pts, k=k)
        cand = cand.reshape(n, k)
        xi, worst = self._reference(cand, self.pts[:, None, :])
        best = np.argmax(worst, axis=1)
        rows = np.arange(n)
        found = worst[rows, best] >= -tol
        self.elem[found] = cand[rows, best][found]
        self.xi[found] = xi[rows, best][found]
        for p in np.nonzero(~found)[0]:                          # rare: search all elements
            allx, w = self._reference(np.arange(len(X))[None, :], self.pts[p][None, None, :])
            e = int(np.argmax(w[0]))
            if w[0, e] >= -tol:
                self.elem[p], self.xi[p] = e, allx[0, e]
        self.inside = self.elem >= 0

    def _reference(self, cand, pts):
        """Reference coordinates (n, k, 3) of points in candidate elements, and their smallest barycentric
        coordinate (>= 0 inside)."""
        xi = np.einsum("nkij,nkj->nki", self._Jinv[cand], pts - self._X0[cand])
        lam = np.concatenate([1 - xi.sum(axis=-1, keepdims=True), xi], axis=-1)
        return xi, lam.min(axis=-1)

    def values(self, nodal: np.ndarray) -> np.ndarray:
        out = np.full(len(self.pts), np.nan, dtype=np.result_type(nodal.dtype, float))
        e = self.elem[self.inside]
        N, _ = shape_functions(self.mesh.order, self.xi[self.inside])
        out[self.inside] = np.einsum("pa,pa->p", N, nodal[self.mesh.cells[e]])
        return out

    def gradient(self, nodal: np.ndarray) -> np.ndarray:
        """(n, 3) gradient of a nodal scalar field (NaN outside)."""
        out = np.full((len(self.pts), 3), np.nan, dtype=np.result_type(nodal.dtype, float))
        e = self.elem[self.inside]
        _, dN = shape_functions(self.mesh.order, self.xi[self.inside])
        G = np.einsum("pai,pij->paj", dN, self._Jinv[e])         # physical shape gradients
        out[self.inside] = np.einsum("pa,paj->pj", nodal[self.mesh.cells[e]], G)
        return out


def evaluate_values(expr: str, scalars: Mapping[str, float], fields: Mapping) -> np.ndarray:
    """An expression of given field values (arrays of one shape, e.g. {"T": temperatures}), element-wise; no
    coordinates, no reductions. Used for properties that depend on a field."""
    ctx = _FieldContext(None, scalars, fields)
    return np.real(np.asarray(ctx.ev(_parse(_fix_if(_clean(expr))))))


def evaluate_points(expr: str, scalars: Mapping[str, float], sampler: PointSampler,
                    fields: Mapping | None = None) -> np.ndarray:
    """An expression evaluated point by point at the sampler's points (no reductions): (n,) array. fields:
    the solution's fields at the sampler's points (NaN outside the structure)."""
    ctx = _FieldContext(None, scalars, fields, nodal=sampler.pts)
    value = np.real(np.asarray(ctx.ev(_parse(_fix_if(_clean(expr))))))
    out = np.broadcast_to(value, (len(sampler.pts),)).astype(float).copy()
    out[~sampler.inside] = np.nan
    return out


class _FieldContext:
    def __init__(self, quad: Quadrature | None, scalars: Mapping[str, float], fields=None,
                 nodal: np.ndarray | None = None):
        self.quad, self.scalars = quad, scalars
        self.fields, self._lazy = {}, {}
        self.factory = fields if callable(fields) else None     # fields of any quadrature (surfaces)
        if self.factory is not None:
            fields = self.factory(quad)
        if nodal is not None:  # pointwise evaluation (nodes, points along a line)
            self.fields.update(x=nodal[:, 0], y=nodal[:, 1], z=nodal[:, 2])
        if quad is not None:
            self.fields.update(x=quad.points[..., 0], y=quad.points[..., 1], z=quad.points[..., 2])
            if isinstance(quad, SurfaceQuadrature):
                self.fields.update(nx=quad.normals[..., 0], ny=quad.normals[..., 1], nz=quad.normals[..., 2])
        for name, value in (fields or {}).items():
            (self._lazy if callable(value) else self.fields)[name] = value

    def ev(self, n):
        if isinstance(n, ast.Expression):
            return self.ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in self.fields and n.id in self._lazy:
                self.fields.update(self._lazy[n.id]())
            if n.id in self.fields:
                return self.fields[n.id]
            if n.id in NORMALS and n.id not in self.scalars:
                raise ExpressionError(f"'{n.id}' (the outward normal) exists only on a surface, e.g. "
                                      f"integral(qx*nx + qy*ny + qz*nz, \"Film:zmax\")")
            if n.id in field_names():
                raise ExpressionError(f"'{n.id}' is not available here")
            if n.id in self.scalars:
                return self.scalars[n.id]
            if n.id in CONSTANTS:
                return CONSTANTS[n.id]
            raise ExpressionError(f"unknown name '{n.id}'")
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](self.ev(n.left), self.ev(n.right))
        if isinstance(n, ast.UnaryOp):
            v = self.ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            return _CMPOPS[type(n.ops[0])](np.real(self.ev(n.left)), np.real(self.ev(n.comparators[0]))) * 1.0
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            f = n.func.id
            if f in REDUCTIONS and len(n.args) == 2 and isinstance(n.args[1], ast.Constant) \
                    and isinstance(n.args[1].value, str) and isinstance(self.quad, Quadrature):
                part = self.quad.surface(n.args[1].value)
                if part is not None:
                    return self._reduce_on(f, n.args[0], part)
            args = [self.ev(a) for a in n.args]
            if f in ("if_", "if") and len(args) == 3:
                return np.where(np.real(args[0]) != 0, args[1], args[2])
            if f in REDUCTIONS:
                return self.reduce(f, args)
            if f in _FUNCS:
                return _FUNCS[f](*args)
            raise ExpressionError(f"unknown function '{f}'")
        raise ExpressionError(f"unsupported syntax: {ast.unparse(n)[:40]}")

    def _reduce_on(self, f, arg, part: SurfaceQuadrature):
        """A reduction over part of the boundary: its argument evaluated on the surface's own points."""
        if self.factory is None:
            raise ExpressionError("reductions over a surface are not available here")
        inner = _FieldContext(part, self.scalars, self.factory)
        return inner.reduce(f, [inner.ev(arg)])

    def reduce(self, f, args):
        region = None
        if len(args) == 2 and isinstance(args[1], str):  # reduction over one object
            region = self.quad.region(args[1])
            args = args[:1]
        if f in ("max", "min") and len(args) == 2:  # elementwise of two values
            return (np.maximum if f == "max" else np.minimum)(np.real(args[0]), np.real(args[1]))
        if len(args) != 1:
            raise ExpressionError(f"{f}() takes a field and optionally an object name, e.g. {f}(T, \"pad\")")
        if self.quad is None:
            raise ExpressionError(f"{f}() reduces over the volume; here the quantity is evaluated point by point "
                                  "(e.g. T, qx, sqrt(qx^2+qy^2))")
        a = np.asarray(args[0])
        w = self.quad.weights
        if region is not None:
            w = w[region]
            a = a[region] if a.ndim else a
        volume = float(w.sum())
        if a.ndim == 0:  # a scalar: mean/max/min of a constant
            return a * volume if f == "integral" else a
        if f == "max":
            return np.max(np.real(a))
        if f == "min":
            return np.min(np.real(a))
        total = np.sum(w * a)
        return total if f == "integral" else total / volume


def evaluate_nodal(expr: str, points: np.ndarray, scalars: Mapping[str, float],
                   fields: Mapping | None = None) -> np.ndarray:
    """Evaluate an expression point by point at every node: (N,) array. Without fields: of x, y, z and the
    variables (e.g. initial values); with the fields of a solution at the nodes (physics' at_nodes) any
    quantity of it, for colouring the structure."""
    ctx = _FieldContext(None, scalars, fields, nodal=points)
    value = np.real(np.asarray(ctx.ev(_parse(_fix_if(_clean(expr))))))
    return np.broadcast_to(value, (len(points),)).astype(float).copy()


def corner_gradients(mesh, nodal: np.ndarray) -> np.ndarray:
    """Gradient of nodal values (N,) or (N, c) at the four corners of every element: (E, 4, 3) or
    (E, 4, c, 3). Gradients jump between elements; to_nodes() averages them for display."""
    _, dN = shape_functions(mesh.order, _CORNERS)              # (4, nn, 3)
    X = mesh.points[mesh.cells]
    vals = nodal[mesh.cells]                                     # (E, nn) or (E, nn, c)
    out = []
    for a in range(4):
        J = np.einsum("eai,aj->eij", X, dN[a])
        G = np.einsum("aj,eji->eai", dN[a], np.linalg.inv(J))   # (E, nn, 3)
        out.append(np.einsum("ea...,eaj->e...j", vals, G))
    return np.stack(out, axis=1)


def to_nodes(mesh, corner_values: np.ndarray) -> np.ndarray:
    """Values at the corners of every element (E, 4, ...) averaged over the elements sharing each corner node;
    mid-edge nodes (P2) get the mean of their two corners. (N, ...)."""
    shape = corner_values.shape[2:]
    total = np.zeros((mesh.n_nodes,) + shape, dtype=corner_values.dtype)
    count = np.zeros(mesh.n_nodes)
    for a in range(4):
        np.add.at(total, mesh.cells[:, a], corner_values[:, a])
        np.add.at(count, mesh.cells[:, a], 1)
    out = total / np.maximum(count, 1).reshape((-1,) + (1,) * len(shape))
    if mesh.order == 2:
        from .fem.elements import TET10_EDGES
        for n, (i, j) in enumerate(TET10_EDGES):
            out[mesh.cells[:, 4 + n]] = 0.5 * (out[mesh.cells[:, i]] + out[mesh.cells[:, j]])
    return out


def evaluate(expr: str, scalars: Mapping[str, float], quad: Quadrature | None = None, fields=None) -> float:
    """One number for one solution (mode, time step, variable value). fields: the solution's fields at the
    quadrature points, or a function of a quadrature returning them, which reductions over a surface need."""
    ctx = _FieldContext(quad, scalars, fields)
    value = np.asarray(ctx.ev(_parse(_fix_if(_clean(expr)))))
    if value.ndim > 0:
        raise ExpressionError("the expression is a field: reduce it with mean(), integral(), max() or min()")
    value = complex(value)
    if abs(value.imag) > 1e-9 * max(abs(value), 1e-300):
        raise ExpressionError("the result is complex: wrap it in abs(), real() or imag()")
    return value.real
