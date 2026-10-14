<img src="src/solidocean/gui/resources/icon.png" alt="SolidOcean icon" width="100">

# SolidOcean

SolidOcean is a free and open-source finite element modelling tool for calculating and analysing phonon
dispersion relations, mechanical waves, and heat transfer problems.

Essentially, it is a Qt user interface that unites several open-source tools for creating a mesh (with the
Gmsh mesher), solving differential equations with boundary conditions (with the MUMPS solver), and analysing
the results (with Matplotlib). Every model is a simple human-readable TOML file that can also be run headless
from the command line, making it ideal for scripting and working with agents.

**Documentation: [anufrievroman.gitbook.io/solidocean](https://anufrievroman.gitbook.io/solidocean)**

![SolidOcean: the mesh of a silicon unit cell with an aluminium pillar](https://raw.githubusercontent.com/anufrievroman/solidocean-docs/main/docs/.gitbook/assets/screenshot.jpeg)

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

The [formalism](https://anufrievroman.gitbook.io/solidocean/physics) pages describe the equations, elements
and solvers, and the [tutorials](https://anufrievroman.gitbook.io/solidocean/tutorials) walk through example
models.

## Installation

All dependencies are available from [conda-forge](https://conda-forge.org/download/) for macOS, Linux and
Windows:

```bash
git clone https://github.com/anufrievroman/solidocean.git
cd solidocean
conda env create -f environment.yml
conda activate solidocean
pip install -e . --no-deps
```

See [Installation](https://anufrievroman.gitbook.io/solidocean/installation) for the details, the macOS app
and installing without conda.

## Usage

Run `solidocean` to open the interface, or `solidocean model.toml` to run a model headless. The
[Usage](https://anufrievroman.gitbook.io/solidocean/usage) page covers the command line, the interface and the
files a run writes, and the [Model file](https://anufrievroman.gitbook.io/solidocean/model-file) reference
describes every section and key of a model.

## Validation

SolidOcean is compared with exact solutions of elasticity and heat conduction, including mesh and time-step
convergence. See [Validation](https://anufrievroman.gitbook.io/solidocean/validation) for the summary and the
full report. The code tests run with `pytest`; the validation suite with `python validation/run.py`
(see [Testing and validation](https://anufrievroman.gitbook.io/solidocean/technical/testing)).

## Contributing

How the code is organised, and how to add a physics, is described in
[Architecture](https://anufrievroman.gitbook.io/solidocean/technical/architecture).

## Disclaimer

SolidOcean is an independent project, a graphical interface to existing open-source tools that unites their
workflow under one roof as a lab tool for academic research. It is provided as is, without any warranty;
check the results of your models against known solutions before relying on them.

## License

Copyright (C) 2026 Anufriev Roman. SolidOcean is free software under the
[GNU General Public License v3.0 or later](LICENSE). It builds on Gmsh, MUMPS, NumPy, SciPy, Matplotlib,
Qt for Python, PyVista and VTK, each under its own license; see
[Licenses](https://anufrievroman.gitbook.io/solidocean/licenses) for the full statements.
