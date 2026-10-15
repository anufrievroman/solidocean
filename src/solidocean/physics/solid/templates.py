"""Solid mechanics templates: eigenfrequency (bulk, plates, beams, phononic crystals), coherent heat
conduction (bands over the Brillouin zone), stationary (cantilever, tensile test) and frequency-domain
(driven waves) models. Naming inside a model: first word capitalised."""

from __future__ import annotations

import copy

from ... import lattice as lat
from ...lattice import Lattice
from ...project import Feature, MeshSettings, VariableSweep, Project, Problem
from ...plots import PlotStyle, ResultPlot
from ...templates import Template, library_material
from .model import KMesh, KPath, SolidCondition, SolidPhysics, SolidProblem
from .plots import (band_map, band_surfaces, colored_plot, conductance_plot, field_plot,
                    spectrum_plot)


def _si(**overrides):
    return library_material("Si", **overrides)


def _lib(name: str, **overrides):
    return library_material(name, **overrides)


HEX = [["a", "0", "0"], ["a/2", "sqrt(3)*a/2", "0"]]


def _kpath(kind: str, n: int = 12) -> KPath:
    points, path = lat.KPATHS[kind]
    return KPath(copy.deepcopy(points), list(path), n)


def _cell(height: str = "t") -> Feature:
    return Feature("cell", "Cell", material="Si", params={"center": ["0", "0", "0"], "height": height})


def _hole(name: str, x: str, y: str) -> Feature:
    return Feature("cylinder", name, op="subtract",
                   params={"center": [x, y, "-t/2"], "axis": ["0", "0", "1"], "radius": "r", "height": "t"})


def _plots(colored: bool = True) -> list[ResultPlot]:
    return [ResultPlot()] + ([colored_plot()] if colored else []) + [field_plot()]


# ---- solid mechanics -------------------------------------------------------------
def empty() -> Project:
    return Project(name="Untitled", variables={"a": "1[um]"}, materials=[_si()],
                   objects=[Feature("block", "Block", material="Si",
                                     params={"center": ["0", "0", "0"], "size": ["a", "a", "a"]})],
                   mesh=MeshSettings(max_size="a/5"))


def bulk() -> Project:
    return Project(
        name="Bulk crystal", description="Bulk Si: cubic cell periodic in x, y and z",
        variables={"a": "100[nm]"}, materials=[_si()],
        objects=[Feature("block", "Cell", material="Si",
                          params={"center": ["0", "0", "0"], "size": ["a", "a", "a"]})],
        mesh=MeshSettings(max_size="a/3"), lattice=Lattice(periodic={"x": "a", "y": "a", "z": "a"}),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=12, kpath=_kpath("cubic", 10))), plots=_plots(False))


def membrane() -> Project:
    return Project(
        name="Membrane (plate)", description="Free-standing Si plate (Lamb waves); the in-plane period is a short "
                                             "auxiliary length, keep it well below the shortest wavelength of interest",
        variables={"t": "220[nm]", "a": "50[nm]"}, materials=[_si()],
        objects=[Feature("block", "Membrane", material="Si",
                          params={"center": ["0", "0", "0"], "size": ["a", "a", "t"]})],
        mesh=MeshSettings(max_size="a/2"), lattice=Lattice(periodic={"x": "a", "y": "a"}),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=12, kpath=KPath({"G": [0, 0, 0], "K": [0.1, 0, 0]}, ["G", "K"], 20))),
        plots=_plots())


def beam() -> Project:
    return Project(
        name="Beam (waveguide)", description="Suspended Si beam (waveguide) along x; the period a is a short "
                                             "auxiliary length",
        variables={"w": "300[nm]", "t": "220[nm]", "a": "50[nm]"}, materials=[_si()],
        objects=[Feature("block", "Beam", material="Si",
                          params={"center": ["0", "0", "0"], "size": ["a", "w", "t"]})],
        mesh=MeshSettings(max_size="a/2"), lattice=Lattice(periodic={"x": "a"}),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=16, kpath=KPath({"G": [0, 0, 0], "K": [0.1, 0, 0]}, ["G", "K"], 20))),
        plots=_plots())


def square_crystal() -> Project:
    return Project(
        name="Square-lattice hole-based phononic crystal",
        description="Square lattice of holes (radius r, period a) in a Si membrane of thickness t",
        variables={"a": "400[nm]", "t": "200[nm]", "r": "0.43*a"}, materials=[_si()],
        objects=[_cell(), _hole("Hole", "0", "0")],
        mesh=MeshSettings(max_size="a/8", min_size="a/40"), lattice=Lattice(periodic={"x": "a", "y": "a"}),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=16, kpath=_kpath("square"))), plots=_plots())


def hexagonal_crystal() -> Project:
    return Project(
        name="Hexagonal-lattice hole-based phononic crystal",
        description="Hexagonal (triangular) lattice of holes in a Si membrane. The unit cell lies between holes: "
                    "a hole sits on each of its corners (four quarter-like pieces), so every cell edge cuts through "
                    "holes and any radius up to touching holes (r < a/2) gives the right structure",
        variables={"a": "400[nm]", "t": "200[nm]", "r": "0.43*a"}, materials=[_si()],
        # corners of the cell spanned by a1 = a(1, 0), a2 = a(1/2, sqrt(3)/2), centred on the origin
        objects=[_cell(), _hole("Hole 1", "3*a/4", "sqrt(3)*a/4"), _hole("Hole 2", "-3*a/4", "-sqrt(3)*a/4"),
                  _hole("Hole 3", "a/4", "-sqrt(3)*a/4"), _hole("Hole 4", "-a/4", "sqrt(3)*a/4")],
        mesh=MeshSettings(max_size="a/8", min_size="a/40"), lattice=Lattice(vectors=copy.deepcopy(HEX)),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=16, kpath=_kpath("hexagonal"))), plots=_plots())


def honeycomb_crystal() -> Project:
    return Project(
        name="Honeycomb hole-based phononic crystal",
        description="Honeycomb lattice: hexagonal lattice with two holes per cell in a Si membrane",
        variables={"a": "400[nm]", "t": "200[nm]", "r": "0.17*a"}, materials=[_si()],
        objects=[_cell(), _hole("Hole 1", "a/4", "sqrt(3)*a/12"), _hole("Hole 2", "-a/4", "-sqrt(3)*a/12")],
        mesh=MeshSettings(max_size="a/8", min_size="a/40"), lattice=Lattice(vectors=copy.deepcopy(HEX)),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=16, kpath=_kpath("hexagonal"))), plots=_plots())


def pillar_crystal() -> Project:
    return Project(
        name="Pillar-based phononic crystal",
        description="Si membrane with an aluminium pillar on top of each cell (square lattice); the pillar is "
                    "bonded to the membrane",
        variables={"a": "400[nm]", "t": "200[nm]", "r": "0.43*a", "h": "200[nm]"},
        materials=[_si(), _lib("Al")],
        objects=[_cell(),
                  Feature("cylinder", "Pillar", material="Al",
                          params={"center": ["0", "0", "t/2"], "axis": ["0", "0", "1"], "radius": "r",
                                  "height": "h"})],
        mesh=MeshSettings(max_size="a/8", min_size="a/40"), lattice=Lattice(periodic={"x": "a", "y": "a"}),
        solid=SolidPhysics(problem=SolidProblem(num_eigs=16, kpath=_kpath("square"))), plots=_plots())


# ---- solid mechanics: stationary ---------------------------------------------------
def zone_crystal() -> Project:
    """A study of the ballistic thermal conductance of a perforated membrane, complete enough to be read at
    0.1 K: the bands over the irreducible wedge, the spectra they carry and the conductance they give."""
    hole = Feature("cylinder", "Hole", op="subtract",
                   params={"center": ["0", "0", "-t/2"], "axis": ["0", "0", "1"], "radius": "r", "height": "t"})
    return Project(
        name="Coherent heat conduction in phononic crystal",
        description="Square lattice of holes in a SiN membrane (the hole fills pi r^2 / a^2 = 0.7 of the "
                    "cell). The bands are computed over a mesh of the irreducible wedge of the Brillouin "
                    "zone and summed into the density of states, the group velocity, the radiated flux and "
                    "the ballistic conductance. 60 bands per wavevector keep the set complete past 40 GHz, "
                    "well above the 19 GHz the occupied phonons reach at 0.1 K. A finer wavevector "
                    "mesh smooths the spectra; it costs one eigenvalue solve per wavevector",
        variables={"a": "300[nm]", "t": "150[nm]", "r": "0.472*a"},
        materials=[_lib("SiN")],
        objects=[Feature("cell", "Cell", material="SiN", params={"center": ["0", "0", "0"], "height": "t"}), hole],
        mesh=MeshSettings(max_size="a/8", min_size="a/40"), lattice=Lattice(periodic={"x": "a", "y": "a"}),
        solid=SolidPhysics(problem=SolidProblem(type="conductance", num_eigs=60, kmesh=KMesh(n=16))),
        plots=[band_surfaces(bands=8), band_map(),
               spectrum_plot(spectrum="dos"), spectrum_plot(spectrum="velocity"),
               _flux_plot(), _conductance_plot(), field_plot()])


def _flux_plot() -> ResultPlot:
    """The spectral heat flux radiated at the temperature the study is aimed at."""
    plot = spectrum_plot(spectrum="flux")
    plot.temperature = "0.1[K]"
    return plot


def _conductance_plot() -> ResultPlot:
    """The conductance over the range the computed bands can speak for (see the plot's own note)."""
    plot = conductance_plot()
    plot.t_min, plot.t_max = "0.02[K]", "0.15[K]"
    return plot


def cantilever() -> Project:
    return Project(
        name="Cantilever under a probe force",
        description="Si cantilever along x (a <100> direction), clamped at x = 0, pushed down by a force F spread "
                    "over a short patch of its top face at the free end (a box selection), e.g. an AFM tip. "
                    "Deflection vs. length, compared with Euler-Bernoulli beam theory",
        variables={"L": "10[um]", "w": "1[um]", "t": "220[nm]", "F": "1[nN]", "lp": "0.5[um]",
                    "E_x": "130.1[GPa]", "a_F": "L - lp/2"},
        materials=[_si()],
        objects=[Feature("block", "Beam", material="Si", params={"center": ["L/2", "0", "0"],
                                                                 "size": ["L", "w", "t"]})],
        mesh=MeshSettings(max_size="t/2"),
        solid=SolidPhysics(problem=SolidProblem(type="stationary"), conditions=[
            SolidCondition("fixed", ["xmin"], name="Clamp"),
            SolidCondition("load", ["zmax"], name="Probe force", load_type="total", load=["0", "0", "-F"],
                           box={"min": ["L - lp", "-w/2", "0"], "max": ["L", "w/2", "t/2"]})]),
        problem=Problem(sweep=VariableSweep("L", "range(5, 2.5, 15) * 1[um]")),
        plots=[field_plot(),
               ResultPlot("Deflection along the beam", "uz / 1[nm]", kind="line", start=["0", "0", "0"],
                          end=["L", "0", "0"], y_label="uz (nm)", style=PlotStyle(markers=False)),
               ResultPlot("Tip deflection", "-min(uz) / 1[nm]", kind="quantity", y_label="Tip deflection (nm)"),
               ResultPlot("Ratio to beam theory", "-min(uz) / (F*a_F^2*(3*L - a_F) / (6*E_x*w*t^3/12))",
                          kind="quantity", y_label="FEM / Euler-Bernoulli")])


def tensile() -> Project:
    return Project(
        name="Tensile test of a crystal bar",
        description="Si bar stretched along x by a prescribed displacement (strain e0); symmetry planes at the "
                    "minimum faces let it contract freely. The crystal is rotated by theta (degrees) about z: "
                    "Young's modulus and Poisson's ratio along x vs. crystal direction, exact along <100> (0 deg) "
                    "and <110> (45 deg); in between, the symmetry planes slightly restrain the shear that a "
                    "rotated crystal couples to tension (E within 0.5 %)",
        variables={"L": "2[um]", "w": "0.5[um]", "e0": "1e-3", "theta": "0"},
        materials=[_si(rotation_deg=["0", "0", "theta"])],
        objects=[Feature("block", "Bar", material="Si", params={"center": ["L/2", "w/2", "w/2"],
                                                                "size": ["L", "w", "w"]})],
        mesh=MeshSettings(max_size="w/2"),
        solid=SolidPhysics(problem=SolidProblem(type="stationary"), symmetry=["xmin", "ymin", "zmin"], conditions=[
            SolidCondition("displacement", ["xmax"], name="Pull", displacement=["e0*L", "", ""])]),
        problem=Problem(sweep=VariableSweep("theta", "range(0, 7.5, 90)")),
        plots=[field_plot(),
               ResultPlot("Young's modulus along x", "mean(sxx) / mean(exx) / 1[GPa]", kind="quantity",
                          y_label="E (GPa)", x_label="Crystal rotation about z (deg)"),
               ResultPlot("Poisson's ratio", "-mean(eyy) / mean(exx)", kind="quantity", y_label="ν_xy",
                          x_label="Crystal rotation about z (deg)")])


def driven_beam() -> Project:
    """Longitudinal waves driven through a phononic-crystal section of a long beam, absorbing ends."""
    def block(name, x0, length):
        return Feature("block", name, material="Si", params={"center": [f"{x0} + ({length})/2", "w/4", "t/4"],
                                                               "size": [length, "w/2", "t/2"]})
    eta = "eta_max * (if(x < La, ((La - x)/La)^2, 0) + if(x > Ltot - La, ((x - Ltot + La)/La)^2, 0))"
    return Project(
        name="Driven waves through a phononic beam", physics_kind="solid",
        description="Longitudinal waves driven at one end of a Si beam (width w, thickness t) pass through N cells "
                    "of holes (period a, radius r): the frequency-domain response to an end displacement u0, swept "
                    "over frequency. Inside the band gaps of these waves (about 1.4-2.2 and 3.1-4.1 GHz) the "
                    "transmission drops by orders of magnitude. Both ends absorb the waves (a loss factor growing "
                    "over the length La: material property eta), so no standing waves form. A quarter of the beam "
                    "(symmetry planes y = 0 and z = 0: a symmetric, longitudinal drive)",
        variables={"a": "1[um]", "w": "1[um]", "t": "0.22[um]", "r": "0.3*a", "N": "8", "La": "8[um]",
                    "Lc": "3[um]", "Ltot": "2*La + 2*Lc + N*a", "x1": "La + Lc", "u0": "1[nm]", "eta_max": "1",
                    "f": "2[GHz]"},
        materials=[_si(eta=eta)],
        objects=[block("Beam", "0", "Ltot"),
                  block("Input", "La + Lc/4", "Lc/2"),
                  block("Output", "x1 + N*a + Lc/4", "Lc/2"),
                  Feature("cylinder", "Holes", op="subtract",
                          params={"center": ["x1 + a/2", "0", "-t"], "axis": ["0", "0", "1"], "radius": "r",
                                  "height": "2*t", "array": ["N", "1", "1"], "spacing": ["a", "0", "0"]})],
        mesh=MeshSettings(max_size="t/2"),
        solid=SolidPhysics(symmetry=["ymin", "zmin"], conditions=[
            SolidCondition("displacement", ["xmin"], displacement=["u0", "", ""], name="Drive")],
            problem=SolidProblem(type="frequency", frequency="f")),
        problem=Problem(sweep=VariableSweep("f", "linspace(1, 4.5, 36) * 1[GHz]")),
        plots=[field_plot("Displacement amplitude"),
               ResultPlot("Transmission", 'mean(abs(ux), "Output") / u0', kind="quantity",
                          y_label="Output amplitude / drive amplitude", style=PlotStyle(y_log=True)),
               ResultPlot("Input amplitude", 'mean(abs(ux), "Input") / u0', kind="quantity",
                          y_label="Input amplitude / drive amplitude", style=PlotStyle(y_log=True))])


TEMPLATES = {  # in display order
    # solid mechanics: eigenfrequencies, phonon dispersion
    "empty": Template("Empty model", empty, "solid"),
    "bulk": Template("Bulk crystal", bulk, "solid"),
    "membrane": Template("Membrane (plate)", membrane, "solid"),
    "beam": Template("Beam (waveguide)", beam, "solid"),
    "square": Template("Square-lattice hole-based phononic crystal", square_crystal, "solid"),
    "hexagonal": Template("Hexagonal-lattice hole-based phononic crystal", hexagonal_crystal, "solid"),
    "honeycomb": Template("Honeycomb hole-based phononic crystal", honeycomb_crystal, "solid"),
    "pillars": Template("Pillar-based phononic crystal", pillar_crystal, "solid"),
    # solid mechanics: coherent heat conduction (bands over the zone)
    "zone_crystal": Template("Coherent heat conduction in phononic crystal", zone_crystal, "solid"),
    # solid mechanics: stationary (loads, prescribed displacements)
    "cantilever": Template("Cantilever under a probe force", cantilever, "solid"),
    "tensile": Template("Tensile test of a crystal bar", tensile, "solid"),
    # solid, frequency domain
    "driven_beam": Template("Driven waves through a phononic beam", driven_beam, "solid"),
}
