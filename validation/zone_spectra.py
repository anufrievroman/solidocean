"""Spectra of an unpatterned membrane over the Brillouin zone, against the rotated Rayleigh-Lamb branches."""

from __future__ import annotations

import numpy as np

import common
from common import Report, Timer, figure, figures, quiet
from lamb_plate import NU, c_t, lamb_frequencies

TITLE = "Density of states of a membrane over the Brillouin zone"
ABOUT = r"""
The same free plate as the Lamb case, but asked a different question: not where the branches lie, but how
many states each frequency carries. An unpatterned membrane is given an artificial square cell of
side $$a = 0.4\,t$$, its bands are computed over a mesh of the irreducible wedge of that cell's Brillouin
zone, and the spectra are summed from the wedge (physics/solid/spectra.py).

The reference needs no mesh. The plate is isotropic, so its exact dispersion is radially symmetric: every
Rayleigh-Lamb and shear-horizontal branch $$\omega_j(k)$$, rotated about the origin, is a surface whose
iso-frequency contour is a circle of circumference $$2\pi k_j(\omega)$$. Summing those circles gives,
per unit angular frequency,

$$D(\omega) = \frac{1}{(2\pi)^2}\sum_j \frac{2\pi k_j}{|\mathrm{d}\omega/\mathrm{d}k|_j},\qquad
\langle v\rangle(\omega) = \frac{\sum_j k_j}{\sum_j k_j/|\mathrm{d}\omega/\mathrm{d}k|_j},$$

evaluated on a dense grid of the exact branches. The two routes share nothing: one is a finite-element
eigenproblem sampled over a two-dimensional zone of an invented cell, the other a one-dimensional analytic
dispersion rotated in k-space. They must still agree, because the artificial cell only folds the states into
its zone and folding moves no state. The comparison runs to $$\omega t / c_\mathrm{T} = 10$$, which is below
the frequency where the sixteen computed bands stop being complete — the lowest value the topmost of them
takes over the zone, not the highest frequency any of them reaches.

What is left over is the price of sampling. Bands are numbered by ascending frequency at each wavevector, so
where two branches cross, the surfaces kink; a mesh of flat triangles cuts those corners and flattens the
gradients. The density of states weights each contour by the gradient and feels this only weakly, while the
group velocity and everything built on the contour length alone (the heat flux, the conductance) carry a
systematic deficit. Both shrink as the wavevector mesh is refined, and both are the argument for sorting the
surfaces into branches, which this release does not do. The last check is the quantity those spectra are
for: the ballistic conductance at 0.1 K, with both routes integrated over the same window, so that what is
compared is the sampling of the zone and not the states left above it.
"""

A_FRAC = 0.4             # side of the artificial cell, in thicknesses
NUM_EIGS = 16            # bands per wavevector: enough that none is missing below OMEGA_MAX
OMEGA_MAX = 10.0         # compared range of omega t / c_T
CHECK_T = 0.1            # K: the temperature the conductance is compared at
THICKNESS = 1e-6


def project(n: int):
    """The plate with a wavevector mesh of n divisions per edge of the irreducible wedge."""
    from solidocean.lattice import Lattice
    from solidocean.physics.solid.model import KMesh, SolidPhysics, SolidProblem
    from solidocean.project import Feature, Material, MeshSettings, Project
    return Project(
        name="Membrane over the zone", physics_kind="solid",
        description="Validation: the spectra of an unpatterned plate, summed over a mesh of its zone",
        variables={"t": "1[um]", "a": f"{A_FRAC}*t"},
        materials=[Material("M", {"model": "isotropic", "rho": "2700[kg/m^3]", "E": "70[GPa]", "nu": str(NU)})],
        objects=[Feature("block", "Membrane", material="M",
                         params={"center": ["0", "0", "0"], "size": ["a", "a", "t"]})],
        mesh=MeshSettings(max_size="t/5", order=2),
        lattice=Lattice(periodic={"x": "a", "y": "a"}),
        solid=SolidPhysics(problem=SolidProblem(type="conductance", num_eigs=NUM_EIGS, kmesh=KMesh(n=n))))


def rotated_lamb(levels: np.ndarray, kt_max: float = 16.0, samples: int = 2400):
    """The exact branches rotated about the origin: (total contour length, that length over |df/dk|) at
    every level, summed over the branches — what the mesh has to reproduce."""
    kt = np.linspace(1e-9, kt_max, samples)
    curves = [lamb_frequencies(x, OMEGA_MAX + 1.0, NU) for x in kt]
    width = max(len(c) for c in curves)
    table = np.full((len(kt), width), np.nan)
    for i, c in enumerate(curves):
        table[i, :len(c)] = c
    k = kt / THICKNESS
    freqs = table * c_t() / (2 * np.pi * THICKNESS)
    length, weighted = np.zeros(len(levels)), np.zeros(len(levels))
    for b in range(width):
        branch = freqs[:, b]
        for i in range(len(k) - 1):
            low, high = branch[i], branch[i + 1]
            if not (np.isfinite(low) and np.isfinite(high)) or low == high:
                continue
            crossed = (levels > min(low, high)) & (levels < max(low, high))
            if not crossed.any():
                continue
            slope = (high - low) / (k[i + 1] - k[i])         # df/dk on this piece of the branch
            radius = k[i] + (levels[crossed] - low) / slope
            length[crossed] += 2 * np.pi * radius
            weighted[crossed] += 2 * np.pi * radius / abs(slope)
    return length, weighted


def l1_error(got: np.ndarray, want: np.ndarray, levels: np.ndarray) -> float:
    """Relative difference of two spectra over the compared range."""
    return float(np.trapezoid(np.abs(got - want), levels) / np.trapezoid(want, levels))


def spectra_of(results, levels: np.ndarray):
    """(density of states, average group velocity, contour length) of a run, sampled at `levels`."""
    from solidocean.physics.solid import spectra
    sampled, length, weighted, copies = spectra.sampled(results, bins=4000)
    dos = np.interp(levels, sampled, copies * weighted / (2 * np.pi) ** 2)
    velocity = np.interp(levels, sampled,
                         np.where(weighted > 0, 2 * np.pi * length / np.where(weighted > 0, weighted, 1.0), 0.0))
    return dos, velocity, np.interp(levels, sampled, copies * length)


def conductance(levels: np.ndarray, contour: np.ndarray, temperature: float) -> float:
    """The ballistic conductance carried by the states in the compared window (W / (K m)): the heat flux
    h f (dn/dT) summed over the contours, integrated over frequency."""
    from solidocean.physics.solid import spectra
    weight = spectra.H_PLANCK * levels * contour / (2 * np.pi ** 2)
    return float(np.trapezoid(weight * spectra.d_occupation_dT(levels, temperature), levels))


def run(quick: bool = False) -> Report:
    from solidocean.physics.solid import modal
    from solidocean.physics.solid.solver import Solver
    r = Report("zone_spectra", TITLE, ABOUT)
    with Timer() as tm:
        top = OMEGA_MAX * c_t() / (2 * np.pi * THICKNESS)
        levels = np.linspace(top / 300, top, 300)
        length, weighted = rotated_lamb(levels, samples=1200 if quick else 2400)
        dos_exact = weighted / (2 * np.pi) ** 2
        velocity_exact = np.where(weighted > 0, 2 * np.pi * length / np.where(weighted > 0, weighted, 1.0), 0.0)

        divisions = [4, 8] if quick else [4, 8, 12]
        points, dos_errors, velocity_errors = [], [], []
        finest = None
        for n in divisions:
            p = project(n)
            if n == divisions[-1]:
                common.save_model(p, "zone_spectra")
            results = modal.run(Solver(p, log=quiet))
            dos, velocity, contour = spectra_of(results, levels)
            from solidocean.physics.solid import spectra
            window = spectra.complete_to(results)        # the bands are all present only below this
            assert window > top, f"only {NUM_EIGS} bands: complete to {window:.3g} Hz, compared to {top:.3g} Hz"
            points.append(results.freqs.shape[1])
            dos_errors.append(l1_error(dos, dos_exact, levels))
            velocity_errors.append(l1_error(velocity, velocity_exact, levels))
            finest = (dos, velocity, contour)

        omega = 2 * np.pi * levels * THICKNESS / c_t()
        fig, (left, right) = figures(2, 9.4, 3.8)
        for ax, got, exact, label in ((left, finest[0], dos_exact, "Density of states, $D$ (Hz$^{-1}$m$^{-2}$)"),
                                      (right, finest[1], velocity_exact,
                                       r"Average group velocity, $\langle v\rangle$ (m/s)")):
            ax.plot(omega, exact, "-", lw=1.1, color=common.REF_COLOR, label="Rotated Lamb branches")
            ax.plot(omega, got, "-", lw=1.1, color=common.FEM_COLOR, label="SolidOcean, zone mesh")
            ax.set_xlabel(r"Frequency, $\omega t / c_\mathrm{T}$")
            ax.set_ylabel(label)
            ax.set_xlim(0, OMEGA_MAX)
            ax.set_ylim(bottom=0)
        left.legend(loc="upper left")
        r.figure(fig, "zone_spectra.png",
                 f"Spectra of an unpatterned plate from a mesh of the irreducible wedge ({points[-1]} "
                 f"wavevectors, {NUM_EIGS} bands) against the exact branches rotated about the origin. The "
                 f"density of states follows the reference including its van Hove steps; the group velocity "
                 f"sits low, where crossings of the unsorted surfaces flatten the gradients.")

        fig, ax = figure()
        for errors, colour, label in ((dos_errors, common.FEM_COLOR, "Density of states"),
                                      (velocity_errors, common.SHORE[2], "Average group velocity")):
            ax.plot(points, errors, "o-", color=colour, label=label)
        common.convergence_axes(ax, "Wavevectors in the wedge", "Relative difference of the spectra")
        ax.legend()
        r.figure(fig, "zone_convergence.png",
                 "Both spectra approach the exact ones as the wavevector mesh is refined; the finite-element "
                 "mesh is unchanged throughout, so what converges here is the sampling of the zone.")

        got = conductance(levels, finest[2], CHECK_T)
        want = conductance(levels, length, CHECK_T)
        r.notes.append(
            "Wavevectors " + " / ".join(str(n) for n in points) + ": the density of states differs from the "
            "exact one by " + ", ".join(f"{e * 100:.1f} %" for e in dos_errors) + ", the group velocity by "
            + ", ".join(f"{e * 100:.1f} %" for e in velocity_errors) + f". Over the same window the states "
            f"below it carry {got:.3g} W/(K m) at {CHECK_T:g} K, against {want:.3g} W/(K m) exactly "
            f"({100 * (got / want - 1):+.0f} %).")
        r.check(f"ballistic conductance at {CHECK_T:g} K", abs(got / want - 1), 0.15)
        r.check("density of states, finest mesh", dos_errors[-1], 0.08)
        r.check("average group velocity, finest mesh", velocity_errors[-1], 0.12)
        r.check("density of states improves with the mesh", dos_errors[-1] / dos_errors[0], 0.8)
        r.check("group velocity improves with the mesh", velocity_errors[-1] / velocity_errors[0], 0.8)
    r.seconds = tm.seconds
    return r
