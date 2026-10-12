"""Heat conduction: rho cp dT/dt = div(k grad T) + Q, on the same tetrahedral meshes.

Weak form with test function v:
    int rho cp v dT/dt + int grad v . k grad T = int v Q + int_S v q_in
convective boundaries q_in = h (T_ext - T) and radiating ones q_in = eps sigma (T_amb^4 - T^4) (linearised
about the previous iterate, Newton). Properties may depend on T (Assembler). Fixed temperatures are imposed by
eliminating the constrained nodes. Scalar unknowns: one temperature per node.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .elements import default_quadrature, shape_functions
from .surface import boundary_faces, faces_on, surface_at_points, surface_terms  # noqa: F401  (boundary triangles, surface integrals)

CHUNK = 8192


def assemble(points, cells, cell_domain, k_list, rhoc_list, order):
    """Conductivity matrix K = int grad N . k grad N and capacity C = int rho cp N N (CSR), constant k (3x3)
    and rho cp per domain."""
    qp, qw = default_quadrature(order)
    N, dN = shape_functions(order, qp)
    n = len(points)
    nn = cells.shape[1]
    K = sp.csr_matrix((n, n))
    C = sp.csr_matrix((n, n))
    for d, (kt, rc) in enumerate(zip(k_list, rhoc_list)):
        dcells = cells[cell_domain == d]
        for start in range(0, len(dcells), CHUNK):
            c = dcells[start:start + CHUNK]
            X = points[c]
            Ke = np.zeros((len(c), nn, nn))
            Ce = np.zeros((len(c), nn, nn))
            for q in range(len(qw)):
                J = np.einsum("eai,aj->eij", X, dN[q])
                w = qw[q] * np.abs(np.linalg.det(J))
                G = np.einsum("aj,eji->eai", dN[q], np.linalg.inv(J))          # physical gradients
                Ke += w[:, None, None] * np.einsum("eai,ij,ebj->eab", G, kt, G, optimize=True)
                Ce += (w * rc)[:, None, None] * np.outer(N[q], N[q])[None]
            rows = np.repeat(c, nn, axis=1).ravel()
            cols = np.tile(c, (1, nn)).ravel()
            K = K + sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(n, n))
            C = C + sp.csr_matrix((Ce.ravel(), (rows, cols)), shape=(n, n))
    return (0.5 * (K + K.T)).tocsr(), (0.5 * (C + C.T)).tocsr()


class Assembler:
    """Re-assembles K and C quickly for coefficients that change between iterations (temperature-dependent
    properties): the element geometry (gradients and weights at the quadrature points) and the sparsity
    pattern are computed once. Large meshes keep the geometry only up to GEOMETRY_BYTES and recompute the
    rest chunk by chunk."""
    GEOMETRY_BYTES = 400e6

    def __init__(self, points, cells, order):
        self.points, self.cells, self.order = points, cells, order
        qp, self.qw = default_quadrature(order)
        self.N, self.dN = shape_functions(order, qp)
        n, (E, nn) = len(points), cells.shape
        keys = (np.repeat(cells, nn, axis=1).astype(np.int64) * n + np.tile(cells, (1, nn))).ravel()
        uniq, self.index = np.unique(keys, return_inverse=True)
        rows, self.indices = uniq // n, (uniq % n).astype(np.int32)
        self.indptr = np.searchsorted(rows, np.arange(n + 1)).astype(np.int32)
        self.n, self.nnz = n, len(uniq)
        per_element = len(self.qw) * (nn * 3 + 1) * 8
        self.cached = min(E, int(self.GEOMETRY_BYTES // per_element))
        self._geo = self._geometry(np.arange(self.cached)) if self.cached else None

    def _geometry(self, elems):
        """Weights (e, Q) and physical shape-function gradients (e, Q, nn, 3)."""
        X = self.points[self.cells[elems]]
        w, G = [], []
        for q in range(len(self.qw)):
            J = np.einsum("eai,aj->eij", X, self.dN[q])
            w.append(self.qw[q] * np.abs(np.linalg.det(J)))
            G.append(np.einsum("aj,eji->eai", self.dN[q], np.linalg.inv(J)))
        return np.stack(w, axis=1), np.stack(G, axis=1)

    def at_quadrature(self, T: np.ndarray) -> np.ndarray:
        """Nodal values at the quadrature points: (E, Q)."""
        return T[self.cells] @ np.asarray(self.N).T

    def _csr(self, Ke: np.ndarray):
        Ke = 0.5 * (Ke + Ke.transpose(0, 2, 1))
        data = np.bincount(self.index, weights=Ke.ravel(), minlength=self.nnz)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))

    def matrices(self, k: np.ndarray | None, rhoc: np.ndarray | None):
        """K for conductivities k (E, Q, 3, 3) and C for rho*cp (E, Q) at the quadrature points (None: skip)."""
        E, nn = self.cells.shape
        Ke = np.zeros((E, nn, nn)) if k is not None else None
        Ce = np.zeros((E, nn, nn)) if rhoc is not None else None
        for start in range(0, E, CHUNK):
            sl = slice(start, min(start + CHUNK, E))
            if sl.stop <= self.cached:
                w, G = self._geo[0][sl], self._geo[1][sl]
            else:
                w, G = self._geometry(np.arange(sl.start, sl.stop))
            if Ke is not None:
                Ke[sl] = np.einsum("eq,eqai,eqij,eqbj->eab", w, G, k[sl], G, optimize=True)
            if Ce is not None:
                Ce[sl] = np.einsum("eq,eq,qa,qb->eab", w, rhoc[sl], self.N, self.N, optimize=True)
        return (None if Ke is None else self._csr(Ke)), (None if Ce is None else self._csr(Ce))


def source_vector(points, cells, cell_domain, q_list, order, profile=None) -> np.ndarray:
    """F_a = int Q N_a over the domains with volumetric heat sources; profile: optional factor of the position,
    a function of the quadrature points (e, 3) -> (e,)."""
    qp, qw = default_quadrature(order)
    N, dN = shape_functions(order, qp)
    F = np.zeros(len(points))
    for d, Q in enumerate(q_list):
        if not Q:
            continue
        c = cells[cell_domain == d]
        X = points[c]
        fe = np.zeros(c.shape)
        for q in range(len(qw)):
            J = np.einsum("eai,aj->eij", X, dN[q])
            w = qw[q] * np.abs(np.linalg.det(J)) * Q
            if profile is not None:
                w = w * profile(np.einsum("a,eai->ei", N[q], X))
            fe += w[:, None] * N[q][None]
        np.add.at(F, c, fe)
    return F


# ---- time profiles ---------------------------------------------------------------
def pulse_integral(shape: str, t: np.ndarray, start: float, duration: float, period: float = 0.0,
                   t_max: float = 0.0) -> np.ndarray:
    """F(t) = int_{-inf}^t f, f = unit-peak pulse ('rect': on for `duration` from `start`; 'gauss': peak at
    `start`, FWHM `duration`), repeated every `period` (0 = once) up to t_max; for 'sine', f =
    sin(2 pi (t - start) / period) and F an antiderivative."""
    from scipy.special import erf
    t = np.asarray(t, dtype=float)
    if shape == "sine":
        return -period / (2 * np.pi) * np.cos(2 * np.pi * (t - start) / period)
    starts = [start] if period <= 0 else start + period * np.arange(int(max(t_max - start, 0) // period) + 2)
    F = np.zeros_like(t)
    for s0 in starts:
        if shape == "rect":
            F += np.clip(t - s0, 0.0, duration)
        else:
            sigma = duration / (2 * np.sqrt(2 * np.log(2)))
            F += sigma * np.sqrt(np.pi / 2) * (1 + erf((t - s0) / (sigma * np.sqrt(2))))
    return F


def pulse_average(shape, t0, t1, start, duration, period=0.0, t_max=0.0) -> float:
    """Mean of the unit-peak pulse (or the sine) over [t0, t1]: energy deposited per step is exact for any dt."""
    F = pulse_integral(shape, np.array([t0, t1]), start, duration, period, t_max)
    return float((F[1] - F[0]) / (t1 - t0))


# ---- solving ----------------------------------------------------------------------
def _solver(A, backend="auto"):
    from .eigen import _factorize
    return _factorize(A.tocsc() if not sp.isspmatrix_csc(A) else A, backend)


def stationary(A, b, fixed: np.ndarray, T_fixed: np.ndarray, backend: str = "auto") -> np.ndarray:
    """Solve A T = b with T[fixed] = T_fixed."""
    n = A.shape[0]
    T = np.zeros(n, dtype=np.result_type(A.dtype, b.dtype, np.asarray(T_fixed).dtype))   # complex: harmonic
    T[fixed] = T_fixed
    free = np.setdiff1d(np.arange(n), fixed)
    Aff = A[free][:, free]
    rhs = b[free] - A[free][:, fixed] @ T_fixed
    T[free] = _solver(Aff, backend)(rhs)
    return T


def transient(A, C, b, fixed, T_fixed, T0, dt, steps, theta=1.0, save_every=1, cancel=None, progress=None,
              t_start=0.0, backend="auto"):
    """theta scheme for C dT/dt + A T = b from t_start (where T = T0) in `steps` steps of dt; returns
    (step indices saved, T snapshots (n_saved, N)); step n is at time t_start + n dt.

    Time-dependent problems: b may be a function (t0, t1) -> load vector for the step from t0 to t1
    (already theta-weighted / step-averaged), and T_fixed a function t -> fixed temperatures.
    The matrix is factorised once (constant dt)."""
    n = A.shape[0]
    free = np.setdiff1d(np.arange(n), fixed)
    load = b if callable(b) else (lambda t0, t1: b)
    T_at = T_fixed if callable(T_fixed) else (lambda t: T_fixed)
    T = T0.copy()
    T[fixed] = T_at(t_start)
    Ml = (C / dt + theta * A).tocsr()
    Mr = (C / dt - (1 - theta) * A).tocsr()
    solve = _solver(Ml[free][:, free], backend)
    Mlf, Mrf = Ml[free][:, fixed], Mr[free]
    lift = Mlf @ T_at(t_start)
    saved, snaps = [0], [T.copy()]
    for step in range(1, steps + 1):
        if cancel is not None and cancel():
            break
        t0, t1 = t_start + (step - 1) * dt, t_start + step * dt
        if callable(T_fixed):
            Tf = T_at(t1)
            lift = Mlf @ Tf
        rhs = Mrf @ T + load(t0, t1)[free] - lift
        T[free] = solve(rhs)
        if callable(T_fixed):
            T[fixed] = Tf
        if step % save_every == 0 or step == steps:
            saved.append(step)
            snaps.append(T.copy())
        if progress:
            progress(step, steps)
    return np.array(saved), np.array(snaps)
