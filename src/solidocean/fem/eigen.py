"""Generalised Hermitian eigenproblem K x = lambda M x via shift-invert Lanczos."""

from __future__ import annotations

import os

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

DENSE_LIMIT = 600  # below this many DOFs use a dense solver


def available_backends() -> list[str]:
    out = []
    try:
        import mumps  # noqa: F401  (conda-forge: python-mumps)
        out.append("mumps")
    except ImportError:
        pass
    out.append("superlu")
    return out


def backend(requested: str = "auto") -> str:
    """Sparse direct solver: the requested one, else $SOLIDOCEAN_SOLVER, else the fastest available."""
    choice = requested.lower() if requested and requested != "auto" else os.environ.get("SOLIDOCEAN_SOLVER", "").lower()
    avail = available_backends()
    if choice:
        if choice not in avail:
            raise RuntimeError(f"solver backend '{choice}' not available (have: {', '.join(avail)})")
        return choice
    return avail[0]


class SingularMatrix(ValueError):
    """The system matrix could not be factorised. Raised as one type whatever the backend reported, so a
    caller that knows the mesh can add to it what the solver cannot know: which element is to blame."""


def _factorize(A: sp.spmatrix, which: str = "auto", permc: str = "MMD_AT_PLUS_A"):
    """Return a solve(b) function for sparse A. permc: the column ordering of SuperLU (COLAMD for saddle-point
    systems, whose zero block makes the default one fill in without end)."""
    try:
        if backend(which) == "mumps":
            import mumps
            ctx = mumps.Context()
            real = not np.iscomplexobj(A.data)
            # MUMPS 'symmetric' means complex-symmetric, so Hermitian matrices use general LU
            ctx.set_matrix(sp.triu(A, format="coo") if real else A.tocoo(), symmetric=real)
            ctx.factor()
            return ctx.solve
        lu = sla.splu(A.tocsc(), permc_spec=permc)
    except Exception as exc:
        raise SingularMatrix(f"the system matrix is singular: it could not be factorised ({exc})") from exc
    return lu.solve


def solve(K: sp.spmatrix, M: sp.spmatrix, n: int, sigma: float | None = None,
          backend_name: str = "auto", tol: float = 0.0, maxiter: int = 0):
    """Return the n eigenpairs (lambda, vectors) closest to sigma (lambda = omega^2).

    sigma=None searches near zero with a small negative shift, which keeps the
    shifted operator non-singular in the presence of rigid-body modes.
    tol/maxiter are passed to ARPACK (0 = its defaults).
    """
    dim = K.shape[0]
    n = min(n, dim)
    scale = float(np.mean(np.abs(K.diagonal())) / np.mean(np.abs(M.diagonal())))
    Ks = K / scale
    sig = -1e-6 if sigma is None or sigma <= 0 else sigma / scale

    if dim <= DENSE_LIMIT or n >= dim - 1:
        lam, vec = la.eigh(Ks.toarray(), M.toarray())
        order = np.argsort(np.abs(lam - sig))[:n]
    else:
        solve_fn = _factorize(Ks - sig * M, backend_name)
        op = sla.LinearOperator(K.shape, matvec=solve_fn, dtype=np.result_type(Ks.dtype, M.dtype))
        lam, vec = sla.eigsh(Ks, k=n, M=M, sigma=sig, which="LM", OPinv=op,
                             tol=tol, maxiter=maxiter or None)
        order = np.arange(len(lam))
    lam, vec = lam[order], vec[:, order]
    idx = np.argsort(lam)
    return lam[idx] * scale, vec[:, idx]


def frequencies(lam: np.ndarray) -> np.ndarray:
    """Eigenfrequencies in Hz from lambda = omega^2 (tiny negatives clipped to 0)."""
    return np.sqrt(np.clip(lam, 0, None)) / (2 * np.pi)
