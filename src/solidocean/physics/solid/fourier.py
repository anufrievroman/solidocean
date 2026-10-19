"""Fourier components of the fields of a mode over the unit cell: the plane-wave amplitudes of a Bloch mode.

A mode of wavevector k is u(r) = e^{i k·r} p(r) with p lattice-periodic, so every field of the mode is a sum
of plane waves at the wavevectors k + G of the reciprocal lattice. This module computes their complex
amplitudes over the cell,

    ŝ(G) = (1/V) ∫_cell f(r) e^{−i (k + G)·r} dV,

for f each of the three displacement components and the six tensor strains, at the reciprocal vectors
G = h1 b1 + h2 b2 + h3 b3 of a grid of orders (|h_i| ≤ orders[i], `Fourier` in model.py). The phase runs
over k + G, not G alone: it is e^{−i k·r} that turns the Bloch field into the periodic one whose Fourier
series this is, so ŝ(G) is the amplitude of the plane-wave component at k + G, independent of which cell of
the structure the integral is taken over (the literal e^{−i G·r} would instead mix the components, and would
depend on that choice through a phase). At k = 0 the two coincide.

The integral and V run over the material in the cell, so ŝ(0) is the volume average of the field over it;
for a cell without holes that is the cell itself. Modes are mass-normalised (`mass_normalise`), which is what
makes the coefficients of different modes comparable.

Degeneracy: where several modes share a frequency, the eigenvectors are an arbitrary basis of that subspace,
so their individual coefficients mean nothing — only sums over the whole subspace do. `degeneracies` finds
those groups, and they are flagged beside the coefficients (fourier_degeneracy.csv).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import fields as solid_fields

FIELDS = solid_fields.DISPLACEMENT + solid_fields.STRAINS   # the fields transformed, in this column order
NORMALISATION = "mass: integral rho |u|^2 dV over the cell = 1 (SI), phase rotated so that sum u.u > 0"
# Two modes count as degenerate when their frequencies differ by less than this much of the highest
# frequency computed at that wavevector. Deliberately loose: a degeneracy split only by the mesh leaves the
# eigenvectors just as mixed as an exact one.
DEGENERACY_TOL = 1e-4
MAX_PHASE = 2_000_000     # entries of the plane-wave matrix held at once (the G grid is split to fit)


def orders_grid(orders) -> np.ndarray:
    """The integer orders (n_G, 3) of the requested grid: every h with |h_i| <= orders[i], b1 slowest."""
    ranges = [np.arange(-int(o), int(o) + 1) for o in orders]
    return np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)


def vectors(project, ev, orders) -> tuple[np.ndarray, np.ndarray]:
    """The orders (n_G, 3) of the grid and the reciprocal vectors G (n_G, 3) in rad/m they stand for at this
    value of the variable sweep (a swept lattice constant scales them)."""
    from ... import lattice as lat
    h = orders_grid(orders)
    recip = lat.reciprocal(lat.slots(project.lattice, ev))
    G = np.zeros((len(h), 3))
    for i, b in enumerate(recip):
        if b is not None:
            G += h[:, i, None] * b
    return h, G


def mass_normalise(u: np.ndarray, M) -> np.ndarray:
    """A mode (N, 3) scaled to unit modal mass, u^H M u = ∫ρ|u|² dV = 1 (SI), with its phase rotated so that
    Σ u·u is real and positive. Fixing both is what lets the components of different modes be compared."""
    flat = u.reshape(-1)
    mass = float(np.real(np.vdot(flat, M @ flat)))
    if mass > 0:
        u = u / np.sqrt(mass)
    return u * np.exp(-1j * np.angle(np.sum(u * u)) / 2)


def _field_values(quad, u: np.ndarray, cel: np.ndarray) -> np.ndarray:
    """The fields of a mode at the quadrature points, (9, n_quadrature) complex, in the order of FIELDS."""
    at = solid_fields.at_quadrature(quad, u, cel)
    strains = at[solid_fields.STRAINS[0]]()      # the lazy mechanics, computed once for all six
    rows = [at[n] for n in solid_fields.DISPLACEMENT] + [strains[n] for n in solid_fields.STRAINS]
    return np.stack([np.asarray(v, dtype=complex).reshape(-1) for v in rows])


def components(quad, modes, cel: np.ndarray, k: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Fourier components (n_modes, n_G, 9) of already-normalised modes (each (N, 3)) at one wavevector.

    ŝ(G) = Σ_q w_q f(r_q) e^{−i (k + G)·r_q} / V over the quadrature of the cell. The plane-wave factors do
    not depend on the mode, so they are built once per batch of reciprocal vectors and used for every mode.
    """
    out = np.empty((len(modes), len(G), len(FIELDS)), dtype=complex)
    r = quad.points.reshape(-1, 3)
    w = quad.weights.reshape(-1) / quad.volume
    step = max(1, MAX_PHASE // max(len(r), 1))
    for s in range(0, len(G), step):
        planes = np.exp(-1j * (r @ (k + G[s:s + step]).T)) * w[:, None]        # (n_quadrature, batch)
        for m, u in enumerate(modes):
            out[m, s:s + step] = (_field_values(quad, u, cel) @ planes).T
    return out


def degeneracies(freqs: np.ndarray) -> np.ndarray:
    """How many modes each mode shares its frequency with, (n_outer, n_inner, n_eigs): 1 where a mode stands
    alone, n inside a group of n modes within DEGENERACY_TOL of the highest frequency of that solve. The
    components of a group's individual modes depend on the arbitrary basis the eigensolver returned."""
    out = np.ones(freqs.shape, dtype=int)
    for i in range(freqs.shape[0]):
        for j in range(freqs.shape[1]):
            f = freqs[i, j]
            idx = np.nonzero(np.isfinite(f))[0]      # modes not solved for are NaN
            if len(idx) == 0:
                continue
            tol = DEGENERACY_TOL * float(np.max(f[idx]))
            start = 0
            for n in range(1, len(idx) + 1):         # frequencies arrive sorted: groups are runs
                if n == len(idx) or f[idx[n]] - f[idx[n - 1]] > tol:
                    out[i, j, idx[start:n]] = n - start
                    start = n
    return out


def _z(v: complex) -> str:
    """A complex number as one CSV field, '1.5e-09-2.1e-10j' (read back by complex() or numpy)."""
    return f"{v.real:.10g}{v.imag:+.10g}j"


def _rows(results):
    """(i, j) of every solve, in the order the tables are written."""
    for i in range(results.freqs.shape[0]):
        for j in range(results.freqs.shape[1]):
            yield i, j


def save_csv(results, project, path) -> Path:
    """fourier.csv: one row per solve, field and reciprocal vector, one complex column per mode."""
    path = Path(path)
    n_eigs = results.freqs.shape[2]
    outer = [results.outer_name] if results.outer_name else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(outer + ["kx [rad/m]", "ky [rad/m]", "kz [rad/m]", results.x_label, "field",
                            "h1", "h2", "h3", "Gx [rad/m]", "Gy [rad/m]", "Gz [rad/m]"] +
                   [f"mode {b + 1}" for b in range(n_eigs)])
        for i, j in _rows(results):
            head = ([f"{results.outer_values[i]:.10g}"] if outer else []) + \
                [f"{v:.10g}" for v in results.k[i, j]] + [f"{results.x[j]:.10g}"]
            for c, name in enumerate(FIELDS):
                for g, h in enumerate(results.fourier_orders):
                    w.writerow(head + [name] + [str(int(v)) for v in h] +
                               [f"{v:.10g}" for v in results.fourier_g[i, g]] +
                               [_z(results.fourier[i, j, b, g, c]) for b in range(n_eigs)])
    return path


def save_degeneracy_csv(results, path) -> Path:
    """fourier_degeneracy.csv: one row per solve, one column per mode, holding how many modes share that
    mode's frequency (1 = a simple mode, whose components stand on their own)."""
    path = Path(path)
    groups = degeneracies(results.freqs)
    n_eigs = results.freqs.shape[2]
    outer = [results.outer_name] if results.outer_name else []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(outer + ["kx [rad/m]", "ky [rad/m]", "kz [rad/m]", results.x_label] +
                   [f"degenerate modes at mode {b + 1}" for b in range(n_eigs)])
        for i, j in _rows(results):
            w.writerow(([f"{results.outer_values[i]:.10g}"] if outer else []) +
                       [f"{v:.10g}" for v in results.k[i, j]] + [f"{results.x[j]:.10g}"] +
                       [str(int(v)) for v in groups[i, j]])
    return path
