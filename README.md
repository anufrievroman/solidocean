<img src="src/solidocean/gui/resources/icon.png" alt="SolidOcean icon" width="100">

# SolidOcean

SolidOcean is a free and open-source finite element modelling tool for calculating and analysing phonon
dispersion relations, mechanical waves, and heat transfer problems.

Essentially, it is a Qt user interface that unites several open-source tools for creating a mesh (with the
Gmsh mesher), solving differential equations with boundary conditions (with the MUMPS solver), and analysing
the results (with Matplotlib). Every model is a simple human-readable TOML file that can also be run headless
from the command line, making it ideal for scripting and working with agents.

![SolidOcean: the mesh of a silicon unit cell with an aluminium pillar](docs/screenshot.jpeg)

## What it can do

**Solid mechanics problems**
- **Eigenfrequencies** and mode shapes calculations with phonon dispersion along chosen wavevector paths in
  selected Brillouin zones.
- **Stationary** displacement calculation with strain, stress and von Mises stress under fixed or prescribed
  displacements and boundary loads.
- **Frequency domain** studies with the driven response to oscillating loads or displacements, with loss
  factors and absorbing regions.

**Heat transfer problems**
- **Stationary** and **transient** problems with fixed temperature, heat flux, convection and radiation
  boundaries, volume heat sources with pulses and spatial profiles, thermal boundary resistances and thin
  resistive layers at interfaces.
- **Frequency domain** problems with the temperature oscillation driven at one frequency, as in the 3ω and
  FDTR methods.

## Installation

All dependencies are available from [conda-forge](https://conda-forge.org/download/) for macOS, Linux and
Windows, which is the recommended way to install them (any conda distribution works;
[Miniforge](https://conda-forge.org/download/) is the smallest one preconfigured for conda-forge):

```bash
git clone https://github.com/anufrievroman/solidocean.git
cd solidocean
conda env create -f environment.yml
conda activate solidocean
pip install -e . --no-deps
```

This is an editable install, so a `git pull` is enough to update it. Run `solidocean` to launch the
interface (the first launch is a bit slow on macOS). On macOS, `solidocean --install-app` then adds
`SolidOcean.app` to `~/Applications`, so it can be opened from Spotlight or the Dock.

The solver is [MUMPS](https://mumps-solver.org) when `python-mumps` is installed (the conda environment
installs it), otherwise SciPy's SuperLU, which is roughly ten times slower. Without conda,
`pip install -e ".[gui,dev]"` works in a virtual environment, but `python-mumps` is not on PyPI and the
`gmsh` wheels are not available for every platform.

## Usage

```bash
solidocean                                # the interface (File → New offers ready-made models)
solidocean --template square slab.toml    # a new model file from a template
solidocean slab.toml                      # run it headless: CSV, NPZ and the model in slab/
solidocean --set "r=0.35*a" slab.toml     # override a variable from the command line
solidocean --install-app                  # macOS: add SolidOcean.app to ~/Applications
```

Templates: `empty`, `bulk`, `membrane`, `beam`, `square`, `hexagonal`, `honeycomb`, `pillars`,
`cantilever`, `tensile`, `driven_beam` (solid mechanics); `heat_empty`, `heat_bar`, `heat_convection`,
`heat_pad`, `heat_radiation`, `heat_diode`, `heat_contact`, `heat_3omega`, `heat_fdtr` (heat transfer).
The model file format is described in [docs/input-format.md](docs/input-format.md).

## Models

Any variable of a model (a dimension, a material property, a frequency) can be swept, and results are
evaluated with expressions of the fields. Boundary conditions apply to faces of the model or of one object,
optionally only inside a box. Geometry is built from blocks, cylinders, cones, spheres, prisms and lattice
cells (added or subtracted, rotated, arrayed) and meshed with linear or quadratic tetrahedra, finer where
needed.
Materials may be anisotropic, rotated and temperature-dependent, with a built-in library of Si, diamond, Ge,
GaAs, 3C-SiC, 4H-SiC, Al, Au, Cr and SiO2.

## Validation

[docs/validation.md](docs/validation.md) compares SolidOcean with exact solutions, including mesh and
time-step convergence: Lamb waves in a plate, the vibrations of a free sphere, cantilever modes against beam
theories, the resonances of a driven bar with losses, Lamé's thick-walled cylinder, a semi-infinite solid
heated by a flux, the spreading resistance of a disk heater, boundary resistances in a layer stack, and the
3ω and FDTR responses. Regenerate it with `python validation/run.py`; the code tests run with `pytest`.

## Disclaimer

SolidOcean is an independent project. It is a graphical interface to open-source tools that already exist
(listed below), uniting their workflow, from geometry to results, under one roof as a lab tool for academic
research. It is provided as is, without any warranty; check the results of your models against known
solutions before relying on them.

## License

Copyright (C) 2026 Anufriev Roman.

SolidOcean is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. It is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; see [LICENSE](LICENSE) for the full text.

SolidOcean builds mainly on these open-source projects, each under its own license:

| Project | Used for | License |
|---|---|---|
| [Gmsh](https://gmsh.info) | geometry and meshing | GPL-2.0-or-later |
| [MUMPS](https://mumps-solver.org) | sparse direct solver | CeCILL-C |
| [NumPy](https://numpy.org) | arrays and assembly | BSD-3-Clause |
| [SciPy](https://scipy.org) | sparse matrices and eigensolver | BSD-3-Clause |
| [Matplotlib](https://matplotlib.org) | plots | Matplotlib license (PSF-based) |
| [Qt for Python (PySide6)](https://doc.qt.io/qtforpython/) | interface | LGPL-3.0 |
| [PyVista](https://pyvista.org) and [VTK](https://vtk.org) | 3D views | MIT, BSD-3-Clause |
