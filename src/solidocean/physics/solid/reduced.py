"""Fast frequency sweep of the harmonic solid problem by a reduced basis.

The system matrix of a harmonic solve is a polynomial in the frequency with fixed matrices,

    A(w) = Kc + i w beta K - (w^2 - i w alpha) M + i w D,    Kc = K + i K_loss,

so the solutions at a few frequencies (exact solves, the snapshots) span a small space in which every other
frequency is accurately solved: u(w) = V q(w) with the r x r system (V^H A(w) V) q = V^H b(w). The snapshots are
chosen adaptively: the residual |A(w) V q - b(w)| / |b(w)| is evaluated at every frequency of the sweep, and the
frequencies where it is above the tolerance get a snapshot, until none is. Frequencies that are still above it
when the basis reaches its largest size are solved exactly, so the tolerance holds for every one of them.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ...fem import eigen
from ..base import Cancelled

BLOCK = 32          # frequencies whose residuals are evaluated together
MAX_CACHE = 400e6   # bytes of right-hand sides kept between rounds
PER_ROUND = 4       # snapshots added per round at most (the worst, not next to each other)


def sweep(solver, evs, tol: float, max_basis: int):
    """Displacements (N, 3) of the harmonic solve at every evaluator of `evs`, or None when the sweep cannot be
    reduced (the matrices change along it, nothing is held, ...; the reason is logged). Returns (list, info)."""
    p = solver.project
    n_val = len(evs)
    keys = [solver.sweep_key(ev) for ev in evs]
    if any(k != keys[0] for k in keys):
        solver.log("Reduced sweep skipped: the sweep changes more than the frequency (geometry, materials or "
                   "constraints); solving every value directly")
        return None
    m = solver.model(evs[0])
    cons, n = m.constraints, m.mesh.n_nodes
    if cons.n_free == 0:
        return None
    T = cons.T().real.tocsr()
    D = solver.impedance_matrix(m, evs[0])
    Kc = m.K + 1j * m.K_loss if m.K_loss is not None else m.K.astype(complex)
    mats = [(T.T @ X @ T).tocsr() for X in (Kc, m.K, m.M, D)]          # Kc, K (beta), M, D on the free DOFs
    full = (Kc, m.K, m.M, D)

    drive = []
    for ev in evs:
        d = list(solver.harmonic_drive(m, ev))
        if drive:
            for k in (3, 4):                                           # a drive that does not change is kept once
                if np.array_equal(d[k], drive[-1][k]):
                    d[k] = drive[-1][k]
        drive.append(d)
    omega, alpha, beta = (np.array([d[i] for d in drive]) for i in range(3))
    theta = np.array([np.ones(n_val), 1j * omega * beta, -(omega ** 2 - 1j * omega * alpha), 1j * omega])   # (4, N)
    f_r = []                                                           # forces on the free DOFs
    for i, d in enumerate(drive):
        f_r.append(f_r[-1] if i and d[3] is drive[i - 1][3] else T.T @ d[3])
    g = [d[4] for d in drive]
    rhs_cache: dict[int, np.ndarray] = {}
    cache_rhs = n_val * cons.n_free * 16 <= MAX_CACHE

    def rhs(i: int) -> np.ndarray:
        """Right-hand side T^H (f - A g) of value i."""
        if i in rhs_cache:
            return rhs_cache[i]
        b = f_r[i].astype(complex)
        if np.any(g[i]):
            for coeff, X in zip(theta[:, i], full):
                b = b - coeff * (T.T @ (X @ g[i]))
        if cache_rhs:
            rhs_cache[i] = b
        return b

    def matrix(i: int):
        return sum(coeff * X for coeff, X in zip(theta[:, i], mats)).tocsc()

    n_free = cons.n_free
    exact: dict[int, np.ndarray] = {}

    def solve_exact(i: int) -> np.ndarray:
        if solver.cancel.is_set():
            raise Cancelled()
        ur = eigen._factorize(matrix(i), p.problem.solver)(rhs(i))
        if not np.all(np.isfinite(ur)):
            raise ValueError("harmonic solve failed: the drive frequency is (numerically) an eigenfrequency of a "
                             "lossless model; add a loss factor (solid.problem.loss or a material's eta) or "
                             "Rayleigh damping")
        exact[i] = ur
        return ur

    V = np.zeros((n_free, 0), dtype=complex)
    W = [np.zeros((n_free, 0), dtype=complex) for _ in mats]            # X_j V

    def extend(ur: np.ndarray):
        nonlocal V, W
        v = ur.astype(complex)
        scale = np.linalg.norm(v)
        for _ in range(2):                                               # Gram-Schmidt, twice
            v = v - V @ (V.conj().T @ v)
        if scale == 0 or np.linalg.norm(v) < 1e-10 * scale:
            return
        v = v / np.linalg.norm(v)
        V = np.column_stack([V, v])
        W = [np.column_stack([w, X @ v]) for w, X in zip(W, mats)]

    def rom():
        """Reduced solutions Q (r, N) of every value and their relative residuals."""
        r = V.shape[1]
        Xv = [V.conj().T @ w for w in W]                                 # (r, r) each
        Q = np.empty((r, n_val), dtype=complex)
        res = np.empty(n_val)
        for start in range(0, n_val, BLOCK):
            idx = range(start, min(start + BLOCK, n_val))
            B = np.column_stack([rhs(i) for i in idx])                   # (n_free, c)
            Ar = sum(theta[j, start:start + len(idx), None, None] * Xv[j][None] for j in range(4))
            try:
                q = np.linalg.solve(Ar, (V.conj().T @ B).T[..., None])[..., 0].T
            except np.linalg.LinAlgError:
                q = np.full((r, len(idx)), np.nan, dtype=complex)
            Q[:, start:start + len(idx)] = q
            R = sum(theta[j, start:start + len(idx)][None, :] * (W[j] @ q) for j in range(4)) - B
            with np.errstate(invalid="ignore", divide="ignore"):
                res[start:start + len(idx)] = np.linalg.norm(R, axis=0) / np.linalg.norm(B, axis=0)
        res[~np.isfinite(res)] = np.inf
        return Q, res

    pick = sorted({0, n_val // 2, n_val - 1})
    rounds = 0
    while True:
        for i in pick:
            extend(solve_exact(i))
        Q, res = rom()
        rounds += 1
        res[list(exact)] = 0.0
        bad = np.flatnonzero(res > tol)
        if len(bad) == 0 or V.shape[1] + 1 > max_basis:
            break
        order = bad[np.argsort(-res[bad])]
        pick = []
        for i in order:
            if all(abs(i - j) > 1 for j in pick) and len(pick) < min(PER_ROUND, max_basis - V.shape[1]):
                pick.append(int(i))
        if not pick:
            break
        solver.log(f"  reduced sweep: {V.shape[1]} snapshots, largest residual {res.max():.2g}")
    leftover = [int(i) for i in bad]
    for i in leftover:                                                   # above the tolerance at the largest basis
        solve_exact(i)
    ur_all = [exact[i] if i in exact else V @ Q[:, i] for i in range(n_val)]
    out = [(T @ ur + g[i]).reshape(n, 3) for i, ur in enumerate(ur_all)]
    done = res[[i for i in range(n_val) if i not in exact]]
    info = {"values": n_val, "exact_solves": len(exact), "basis": int(V.shape[1]),
            "max_residual": float(done.max()) if len(done) else 0.0}
    solver.log(f"Reduced sweep: {info['exact_solves']} exact solves for {n_val} values "
               f"(basis {info['basis']}, largest residual {info['max_residual']:.2g})")
    return out, info
