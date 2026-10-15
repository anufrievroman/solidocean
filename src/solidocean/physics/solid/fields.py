"""Field variables of solid mechanics for quantities: displacement, strain, stress and von Mises stress, at
quadrature points (reductions) or at sample points (line plots)."""

from __future__ import annotations

import functools

import numpy as np

from ...fields import corner_gradients, to_nodes

DISPLACEMENT = ("ux", "uy", "uz")
STRAINS = ("exx", "eyy", "ezz", "eyz", "exz", "exy")    # tensor strain components (Voigt order)
STRESSES = ("sxx", "syy", "szz", "syz", "sxz", "sxy")   # Cauchy stress (Pa)
MECHANICS = STRAINS + STRESSES + ("mises",)
FIELD_NAMES = DISPLACEMENT + MECHANICS


def _mechanics(grad: np.ndarray, C: np.ndarray) -> dict:
    """Strains, stresses and the von Mises stress from displacement gradients grad[..., i, j] = du_i/dx_j and
    the stiffness C[..., 6, 6] (Voigt) at the same points (C broadcasts over trailing point axes)."""
    e = 0.5 * (grad + np.swapaxes(grad, -1, -2))
    voigt = np.stack([e[..., 0, 0], e[..., 1, 1], e[..., 2, 2],
                      2 * e[..., 1, 2], 2 * e[..., 0, 2], 2 * e[..., 0, 1]], axis=-1)
    sig = np.einsum("...ij,...j->...i", C, voigt)
    out = {name: voigt[..., n] / (2 if n >= 3 else 1) for n, name in enumerate(STRAINS)}
    out.update({name: sig[..., n] for n, name in enumerate(STRESSES)})
    s = np.real(sig)   # modes: at the displayed phase
    out["mises"] = np.sqrt(0.5 * ((s[..., 0] - s[..., 1]) ** 2 + (s[..., 1] - s[..., 2]) ** 2 +
                                  (s[..., 2] - s[..., 0]) ** 2) + 3 * (s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2))
    return out


def at_nodes(mesh, u: np.ndarray, cel: np.ndarray | None = None, keep_complex: bool = False) -> dict:
    """Fields of a displacement u (N, 3) at the nodes, for colouring the structure: the displacement, and
    (computed on first use) strains and stresses at element corners averaged over the elements sharing a
    node. Modes are shown at their displayed phase (real part); keep_complex keeps the complex amplitudes of a
    harmonic response, so that abs(uz) is its amplitude."""
    u = u if keep_complex else np.real(u)
    out = {"ux": u[:, 0], "uy": u[:, 1], "uz": u[:, 2]}
    if cel is not None:
        @functools.cache          # computed once, however many expressions use it
        def mechanics():
            corner = _mechanics(corner_gradients(mesh, u), cel[:, None])     # each (E, 4)
            return {name: to_nodes(mesh, v) for name, v in corner.items()}
        out.update({name: mechanics for name in MECHANICS})
    return out


def nodal_mises(mesh, u: np.ndarray, cel: np.ndarray) -> np.ndarray:
    """Von Mises stress (N,) at the nodes (corner values averaged over the elements sharing a node)."""
    return at_nodes(mesh, u, cel)["mises"]()["mises"]


def normalise(u: np.ndarray, quad) -> np.ndarray:
    """Scale a mode (N, 3) to mean(|u|^2) = 1 and rotate its phase to be as real as possible."""
    uq = quad.at_points(u)
    mean_sq = float(np.sum(quad.weights[..., None] * np.abs(uq) ** 2) / quad.volume)
    u = u / np.sqrt(mean_sq) if mean_sq > 0 else u
    return u * np.exp(-1j * np.angle(np.sum(u * u)) / 2)


def at_quadrature(quad, u: np.ndarray, cel: np.ndarray | None = None) -> dict:
    """Fields of a displacement u (N, 3) at the quadrature points; strains and stresses (computed on first use)
    need the stiffness of every element, cel (E, 6, 6)."""
    uq = quad.at_points(u)
    out = {"ux": uq[..., 0], "uy": uq[..., 1], "uz": uq[..., 2]}
    if cel is not None:
        def mechanics():
            grad = np.stack([quad.grad_at_points(u[:, c]) for c in range(3)], axis=-2)   # (E, Q, 3, 3)
            return _mechanics(grad, cel[quad.elements][:, None])
        out.update({name: mechanics for name in MECHANICS})
    return out


def at_points(sampler, u: np.ndarray, cel: np.ndarray | None = None) -> dict:
    """Fields of a displacement u (N, 3) at a sampler's points (NaN outside the structure)."""
    out = {"ux": sampler.values(u[:, 0]), "uy": sampler.values(u[:, 1]), "uz": sampler.values(u[:, 2])}
    if cel is not None:
        C = np.full((len(sampler.pts), 6, 6), np.nan)
        C[sampler.inside] = cel[sampler.elem[sampler.inside]]
        out.update(_mechanics(np.stack([sampler.gradient(u[:, c]) for c in range(3)], axis=-2), C))
    return out
