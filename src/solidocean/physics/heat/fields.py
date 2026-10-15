"""Field variables of heat transfer for quantities: temperature, its gradient and the heat flux, at quadrature
points (reductions) or at sample points (line plots).

Frequency-domain results are complex amplitudes T^ of the oscillation T(t) = Re(T^ exp(i omega t)): T and the
gradient and flux fields are then their real parts (the oscillation at phase 0), Tre and Tim the in-phase and
out-of-phase parts, Tamp = |T^| and Tphase its phase (degrees; negative: lagging the drive). For steady and
transient results Tre = Tamp = T, Tim = Tphase = 0."""

from __future__ import annotations

import numpy as np

FIELD_NAMES = ("T", "Tx", "Ty", "Tz", "qx", "qy", "qz", "Tre", "Tim", "Tamp", "Tphase")
HARMONIC = ("Tre", "Tim", "Tamp", "Tphase")


def _harmonic(Tc: np.ndarray) -> dict:
    """In-phase, out-of-phase, amplitude and phase (degrees) of complex values."""
    return dict(Tre=np.real(Tc), Tim=np.imag(Tc), Tamp=np.abs(Tc), Tphase=np.degrees(np.angle(Tc)))


def _flux(kel, elements: np.ndarray, T: np.ndarray, g: np.ndarray) -> np.ndarray:
    """-k grad T at points of the given elements (...) with temperatures T (...) and gradients g (..., 3); kel:
    a properties.Conductivity, or the (E, 3, 3) conductivity of every element."""
    k = kel.tensor(elements, T) if hasattr(kel, "tensor") else kel[elements]
    return -np.einsum("...ij,...j->...i", k, g)


def at_quadrature(quad, T: np.ndarray, kel=None) -> dict:
    """Fields of a nodal temperature T (N,); the gradient and the heat flux -k grad T (with kel, the
    conductivity, see _flux) are computed on first use."""
    Tr = np.real(T)

    def gradient():
        g = quad.grad_at_points(Tr)
        out = dict(Tx=g[..., 0], Ty=g[..., 1], Tz=g[..., 2])
        if kel is not None:
            Tq = quad.at_points(Tr)
            q = _flux(kel, np.broadcast_to(quad.elements[:, None], Tq.shape), Tq, g)
            out.update(qx=q[..., 0], qy=q[..., 1], qz=q[..., 2])
        return out

    def harmonic():
        return _harmonic(quad.at_points(T))
    out = {"T": lambda: {"T": quad.at_points(Tr)}}
    out.update({name: gradient for name in ("Tx", "Ty", "Tz") + (("qx", "qy", "qz") if kel is not None else ())})
    out.update({name: harmonic for name in HARMONIC})
    return out


def at_points(sampler, T: np.ndarray, kel=None) -> dict:
    """Fields of a nodal temperature T (N,) at a sampler's points (NaN outside the structure)."""
    Tr = np.real(T)
    g = sampler.gradient(Tr)
    Tp = sampler.values(Tr)
    out = dict(T=Tp, Tx=g[:, 0], Ty=g[:, 1], Tz=g[:, 2], **_harmonic(sampler.values(T)))
    if kel is not None:
        q = np.full((len(g), 3), np.nan)
        inside = sampler.inside
        q[inside] = _flux(kel, sampler.elem[inside], Tp[inside], g[inside])
        out.update(qx=q[:, 0], qy=q[:, 1], qz=q[:, 2])
    return out


def at_nodes(mesh, T: np.ndarray, kel=None) -> dict:
    """Fields of a nodal temperature T (N,) at the nodes, for colouring the structure: T, and (on first use)
    the gradient and heat flux at element corners averaged over the elements sharing a node."""
    import functools
    from ...fields import corner_gradients, to_nodes

    Tr = np.real(T)

    @functools.cache              # computed once, however many expressions use it
    def gradient():
        g = corner_gradients(mesh, Tr)                               # (E, 4, 3)
        out = {f"T{a}": to_nodes(mesh, g[..., i]) for i, a in enumerate("xyz")}
        if kel is not None:
            Tc = Tr[mesh.cells[:, :4]]                               # (E, 4) at the corners
            q = _flux(kel, np.broadcast_to(np.arange(len(Tc))[:, None], Tc.shape), Tc, g)
            out.update({f"q{a}": to_nodes(mesh, q[..., i]) for i, a in enumerate("xyz")})
        return out
    out = {"T": Tr, **_harmonic(T)}
    out.update({name: gradient for name in ("Tx", "Ty", "Tz") + (("qx", "qy", "qz") if kel is not None else ())})
    return out
