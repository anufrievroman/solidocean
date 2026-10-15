"""Spectral quantities of a wavevector mesh: what the bands over the Brillouin zone say about frequency,
rather than about one wavevector.

Every band is sampled on the same triangulation of the zone (physics/solid/modal.py, lattice.zone_mesh), so
over each triangle the band is the plane through its three corner frequencies. The level set f_j(k) = f of
that plane is a straight segment, and its gradient is constant, which turns every integral over the zone
into a sum over (band, triangle) pairs the level crosses:

    D(f)   = (1/(2 pi)^2) S sum_j,m  L_jm(f) / |grad f_jm|          states per unit area per Hz
    <v>(f) = 2 pi sum_j,m L_jm(f) / sum_j,m L_jm(f) / |grad f_jm|   mean group velocity of those states
    p(f,T) = h f n(f,T) S sum_j,m L_jm(f) / (2 pi^2)                heat flux per unit width per Hz

with L_jm(f) the length of the segment in triangle m of band j and S the number of copies of the meshed
region in the zone (8 for the wedge of a square lattice, 12 for a hexagonal one, 1 for the whole zone),
taken from the area the mesh covers. The group velocity |v| = 2 pi |grad_k f| follows from the same
gradients (k is in rad/m).

Everything here is per unit frequency f, which is what the plots draw and what the files record. The
published form of these, and the one the panels and the model-file reference show, is per unit angular
frequency omega = 2 pi f: a density per unit f is 2 pi times the one per unit omega, h f is the h-bar omega
of those equations, and the velocity is the same either way.

The heat flux is the emission of a convex boundary averaged over its directions: averaging (v . n) over the
directions of n gives |v| / pi, which cancels the group velocity against the Jacobian of the change of
variable from k to f and leaves the contour length alone. That average is also what makes a mesh of the
irreducible wedge enough: a flux in one chosen direction is not symmetric under the point group, so it
could not be recovered by multiplying the wedge by S.

A run computes a fixed number of bands at every wavevector, so the set is only complete below the *lowest*
value the highest of them takes over the zone (`complete_to`) — usually at Gamma, where the bands of a
membrane are folded down and crowd together. Above that line states are missing and every spectrum sags
towards zero, which looks like physics and is not: the highest frequency reached somewhere in the zone says
nothing about where the set stops being complete.

Integrating the flux over frequency gives what the whole band structure carries, against temperature:

    P(T) = int p(f,T) df            power radiated per unit width of the boundary
    G(T) = dP/dT = int h f (dn/dT) S sum L / (2 pi^2) df      ballistic thermal conductance

Neither needs the mesh again: the contour lengths do not depend on the temperature, so they are summed once
and every temperature is a quadrature over the same sampled spectrum. Both are truncated at the highest
computed band, so they are lower bounds once k_B T / h approaches it.

Reference: the methods of Zen et al., Nat. Commun. 5:3435 (2014) (literature/), whose DOS is this sum with
S = 8, divided by the membrane thickness to read per unit volume, and whose Eq. (1) is P(T).
"""

from __future__ import annotations

import numpy as np

H_PLANCK = 6.62607015e-34       # J s
K_BOLTZMANN = 1.380649e-23      # J / K

SPECTRA = ("dos", "velocity", "flux")
TRANSPORT = ("conductance", "power")
TRANSPORT_TITLES = {"conductance": "Thermal conductance", "power": "Radiated power"}
TRANSPORT_LABELS = {"conductance": "Thermal conductance (W K$^{-1}$ m$^{-1}$)",
                    "power": "Radiated power (W m$^{-1}$)"}
TRANSPORT_HELP = {
    "conductance": "The ballistic thermal conductance G = dP/dT of the structure, per unit length of the "
                   "boundary heat enters through: what the band structure carries when nothing scatters.",
    "power": "The power that boundary radiates into the structure at each temperature, per unit of its "
             "length; the conductance is its derivative.",
}
SPECTRUM_TITLES = {"dos": "Density of states", "velocity": "Average group velocity",
                   "flux": "Spectral heat flux"}
SPECTRUM_LABELS = {"dos": "Density of states (Hz$^{-1}$ m$^{-2}$)",
                   "velocity": "Average group velocity (m/s)",
                   "flux": "Spectral heat flux (W Hz$^{-1}$ m$^{-1}$)"}
SPECTRUM_HELP = {
    "dos": "States per unit frequency and unit area of the membrane: divide by its thickness for the "
           "per-volume density of states.",
    "velocity": "The magnitude of the group velocity averaged over the states at that frequency (weighted "
                "like the density of states), not over the wavevectors.",
    "flux": "The energy a convex boundary at the given temperature radiates into the structure, per unit "
            "frequency and unit length of that boundary, averaged over its directions.",
}


class NoCompleteRange(ValueError):
    """The computed bands have no frequency range in which every state is present."""


def multiplicity(reduced: np.ndarray, tris: np.ndarray) -> float:
    """How many copies of the meshed region fill the zone, from the area it covers (the zone has unit area
    in reduced coordinates): 1 for the whole zone, 8 for the wedge of a square lattice, 12 for a hexagonal
    one. It assumes what the wedge itself assumes — that the cell has the point symmetry of the lattice."""
    a, b, c = reduced[tris[:, 0]], reduced[tris[:, 1]], reduced[tris[:, 2]]
    u, v = b - a, c - a
    area = float(0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum())
    return 1.0 / area if area > 0 else 1.0


def _planes(k: np.ndarray, tris: np.ndarray, values: np.ndarray):
    """Per triangle: its corner positions sorted by value, the sorted values, and |grad| of the plane
    through them (0 where the triangle is degenerate)."""
    points = k[tris]                                   # (M, 3, 2)
    corner = values[tris]                              # (M, 3)
    order = np.argsort(corner, axis=1)
    rows = np.arange(len(tris))[:, None]
    points, corner = points[rows, order], corner[rows, order]
    e1, e2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    d1, d2 = corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    safe = np.where(det == 0, 1.0, det)
    grad = np.stack([(d1 * e2[:, 1] - d2 * e1[:, 1]) / safe,
                     (-d1 * e2[:, 0] + d2 * e1[:, 0]) / safe], axis=1)
    grad[det == 0] = 0.0
    return points, corner, np.linalg.norm(grad, axis=1)


def _segment_lengths(points: np.ndarray, corner: np.ndarray, level: float) -> np.ndarray:
    """Length of the level set inside every triangle whose range contains `level` (0 for the others). The
    corners are sorted by value, so the level crosses either the two edges leaving the lowest corner or the
    two edges arriving at the highest one."""
    low, mid, high = corner[:, 0], corner[:, 1], corner[:, 2]
    inside = (low < level) & (level < high)
    out = np.zeros(len(corner))
    if not inside.any():
        return out
    p = points[inside]
    a, b, c = low[inside], mid[inside], high[inside]
    below = level <= b                                  # the level cuts off the lowest corner
    t_ac = ((level - a) / (c - a))[:, None]
    q1 = p[:, 0] + t_ac * (p[:, 2] - p[:, 0])           # on the longest edge either way
    t_ab = np.where(b > a, (level - a) / np.where(b > a, b - a, 1.0), 0.0)[:, None]
    t_bc = np.where(c > b, (level - b) / np.where(c > b, c - b, 1.0), 0.0)[:, None]
    q2 = np.where(below[:, None], p[:, 0] + t_ab * (p[:, 1] - p[:, 0]),
                  p[:, 1] + t_bc * (p[:, 2] - p[:, 1]))
    out[inside] = np.linalg.norm(q1 - q2, axis=1)
    return out


def contour_sums(k: np.ndarray, tris: np.ndarray, freqs: np.ndarray,
                 levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Summed over every band and triangle the level crosses: the length of the iso-frequency contour, and
    that length divided by |grad f| (the measure the density of states counts with).

    k: (P, 2) wavevectors in rad/m; tris: (M, 3); freqs: (P, B) one column per band; levels: (L,) in Hz.
    """
    length = np.zeros(len(levels))
    weighted = np.zeros(len(levels))
    for band in range(freqs.shape[1]):
        values = freqs[:, band]
        if not np.all(np.isfinite(values)):
            continue                                    # a band the eigensolver did not reach everywhere
        points, corner, grad = _planes(k, tris, values)
        usable = grad > 0
        lo, hi = corner[:, 0].min(), corner[:, 2].max()
        for n, level in enumerate(levels):
            if not lo < level < hi:
                continue
            seg = _segment_lengths(points, corner, float(level))
            length[n] += seg.sum()
            weighted[n] += (seg[usable] / grad[usable]).sum()
    return length, weighted


def bose_einstein(freq: np.ndarray, temperature: float) -> np.ndarray:
    """Occupation n(f, T) of a mode of frequency f (Hz); 0 at T = 0."""
    if temperature <= 0:
        return np.zeros_like(freq)
    x = np.clip(H_PLANCK * freq / (K_BOLTZMANN * temperature), 1e-12, 700.0)
    return 1.0 / np.expm1(x)


def _bands_of(results, outer: int, band: int, bands: int) -> np.ndarray:
    """The columns of frequencies a spectrum is summed over."""
    if not results.is_kmesh:
        raise ValueError("spectra are computed from a wavevector mesh (a coherent heat conduction problem)")
    n_bands = results.freqs.shape[2]
    first = min(max(band, 0), n_bands - 1)
    last = n_bands if bands < 1 else min(first + bands, n_bands)
    return results.freqs[outer, :, first:last]


def complete_to(results, outer: int = 0, band: int = 0, bands: int = 0) -> float:
    """The highest frequency (Hz) whose states are all accounted for: the lowest value the topmost included
    band takes over the zone. Above it the run simply stopped computing bands, so a spectrum drawn there is
    missing states rather than reporting that there are none."""
    freqs = _bands_of(results, outer, band, bands)
    top = freqs[:, -1]
    return float(np.nanmin(top)) if np.any(np.isfinite(top)) else 0.0


def highest(results, outer: int = 0, band: int = 0, bands: int = 0) -> float:
    """The highest frequency computed anywhere in the zone — not where the bands stop being complete."""
    freqs = _bands_of(results, outer, band, bands)
    return float(np.nanmax(freqs)) if np.any(np.isfinite(freqs)) else 0.0


def sampled(results, bins: int = 200, outer: int = 0, band: int = 0, bands: int = 0,
            incomplete: bool = False):
    """The frequencies a mesh is sampled at and the sums every spectrum is built from: (levels, total
    contour length, that length over |grad f|, copies of the meshed region in the zone).

    The levels stop where the bands stop being complete; `incomplete` carries them to the highest frequency
    computed anywhere instead, where the spectra are a lower bound.
    """
    freqs = _bands_of(results, outer, band, bands)
    top = highest(results, outer, band, bands) if incomplete else complete_to(results, outer, band, bands)
    if top <= 0:
        raise NoCompleteRange(
            "the topmost computed band falls to zero somewhere in the zone, so no range of frequencies has "
            "all of its states: compute more eigenfrequencies, or carry the spectrum past the complete range")
    levels = np.linspace(0.0, top, max(int(bins), 2) + 1)[1:]          # 0 carries no states
    length, weighted = contour_sums(results.k[outer][:, :2], results.ktris, freqs, levels)
    return levels, length, weighted, multiplicity(results.kred, results.ktris)


def spectrum(results, kind: str, bins: int = 200, temperature: float = 1.0, outer: int = 0,
             band: int = 0, bands: int = 0, incomplete: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(frequencies in Hz, values) of a spectral quantity of a wavevector mesh: "dos", "velocity" or
    "flux". `bands` bands from `band` up are included (0 = all of them), for value `outer` of the variable
    sweep; `temperature` (K) only matters to the flux."""
    if kind not in SPECTRA:
        raise ValueError(f"unknown spectrum '{kind}' (one of {', '.join(SPECTRA)})")
    levels, length, weighted, copies = sampled(results, bins, outer, band, bands, incomplete)
    if kind == "dos":
        return levels, copies * weighted / (2 * np.pi) ** 2
    if kind == "velocity":
        return levels, np.where(weighted > 0, 2 * np.pi * length / np.where(weighted > 0, weighted, 1.0), 0.0)
    energy = H_PLANCK * levels * bose_einstein(levels, temperature)
    return levels, copies * energy * length / (2 * np.pi ** 2)


def save_table(results, project, pdef, path) -> "Path":
    """The curve of a spectrum or conductance plot as a CSV."""
    import csv
    from pathlib import Path

    from ...expressions import UNITS
    path = Path(path)
    if pdef.kind == "conductance":
        from ...results.modal import temperatures_of
        temperatures, values = transport(results, pdef.transport, temperatures_of(project, pdef),
                                         bins=pdef.bins, outer=max(pdef.at, 0), band=pdef.band,
                                         bands=pdef.bands)
        header = {"conductance": "thermal conductance [W/(K*m)]", "power": "radiated power [W/m]"}
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["temperature [K]", header[pdef.transport]])
            for t, v in zip(temperatures, values):
                writer.writerow([f"{t:.10g}", f"{v:.10g}"])
        return path
    temperature = project.evaluator()(pdef.temperature)
    freq, values = spectrum(results, pdef.spectrum, bins=pdef.bins, temperature=temperature,
                            outer=max(pdef.at, 0), band=pdef.band, bands=pdef.bands)
    unit = project.frequency_unit
    scale = UNITS[unit]
    header = {"dos": "density of states [1/(Hz*m^2)]", "velocity": "group velocity [m/s]",
              "flux": f"heat flux at {temperature:.6g} K [W/(Hz*m)]"}[pdef.spectrum]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f"frequency [{unit}]", header])
        for f, v in zip(freq, values):
            writer.writerow([f"{f / scale:.10g}", f"{v:.10g}"])
    return path


def d_occupation_dT(freq: np.ndarray, temperature: float) -> np.ndarray:
    """dn/dT of the Bose-Einstein occupation at frequency f (Hz); 0 at T = 0."""
    if temperature <= 0:
        return np.zeros_like(freq)
    x = np.clip(H_PLANCK * freq / (K_BOLTZMANN * temperature), 1e-12, 700.0)
    return (x / temperature) * np.exp(-x) / np.expm1(-x) ** 2


def transport(results, kind: str, temperatures: np.ndarray, bins: int = 200, outer: int = 0,
              band: int = 0, bands: int = 0, incomplete: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(temperatures in K, values) of the radiated power P(T) in W/m or the ballistic thermal conductance
    G(T) = dP/dT in W/(K m), integrating the heat flux over the computed bands. The contour lengths do not
    depend on the temperature, so the mesh is summed once for the whole curve."""
    if kind not in TRANSPORT:
        raise ValueError(f"unknown quantity '{kind}' (one of {', '.join(TRANSPORT)})")
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    levels, length, _, copies = sampled(results, bins, outer, band, bands, incomplete)
    weight = copies * H_PLANCK * levels * length / (2 * np.pi ** 2)
    occupation = bose_einstein if kind == "power" else d_occupation_dT
    values = np.array([np.trapezoid(weight * occupation(levels, float(t)), levels) for t in temperatures])
    return temperatures, values


def thermal_frequency(temperature: float) -> float:
    """k_B T / h (Hz): the scale of the occupied frequencies. The computed bands must reach well above it
    for an integral over them to stand for the whole spectrum."""
    return K_BOLTZMANN * temperature / H_PLANCK
