# Architecture

SolidOcean is organised so that each physics (solid mechanics, heat transfer, …) is an isolated package
and the shared code — model file, geometry, mesh, numerical kernels, CLI and GUI — never tests which
physics it is dealing with. Shared code finds a physics through a registry and talks to it through a small
interface.

## Layers

```
src/solidocean/
  expressions.py  units.py          expressions with units, dimensions, display formatting
  materials.py                      elastic and thermal material models, library
  geometry.py  meshing.py  lattice.py
  selection.py                      boundary selections of conditions (on / faces / box), any physics
  fem/                              numerical kernels: elements, elastic assembly, heat assembly, boundary
                                    surfaces, Bloch reduction, eigen and linear solvers
  fields.py                         expression engine for quantities (physics-neutral, see below)
  plots.py                          results plot definitions (ResultPlot, PlotStyle)
  project.py                        the model: variables, materials, geometry, lattice, mesh, shared study,
                                    output, plots, and one settings section per physics
  tomlio.py                         compact TOML writing (no defaults, short arrays on one line)
  physics/                          one subpackage per physics + the registry
    base.py                         PhysicsSpec: the interface of a physics
    solid/  heat/                   settings model, validation, solver, results, field variables,
                                    plots, templates, spec (in __init__.py)
  results/                          results shared by all physics
    style.py                        plot style, axes, PNG writing
    field.py                        FieldResults (one field per variable value and time step) and
                                    their plots: vs time, vs variable, along a line
    modal.py                        plots of modal results (dispersion)
  templates.py                      collects the templates of every physics
  cli.py                            headless runs: spec.solver(...).run(), results.write_outputs()
  gui/
    main_window.py                  window shell: tree, panels, file ops, tasks, geometry and mesh
    results_view.py                 running studies, plots, the selected solution and its field, exports
    physics/                        GUI adapter per physics (PhysicsUI): tree nodes, panels, context
                                    menu, condition highlights, field display
    panels/                         settings panels: common widgets, model, study, results, solid, heat
    viewer.py  plot_canvas.py       3D view (PyVista) and 2D plots (matplotlib)
```

Dependencies point downwards: physics packages use the shared layers; shared layers use a physics only
through `physics.get(kind)` / `physics.of(project)` (core) and `gui.physics.of(project)` (GUI). No module
outside `physics/<name>/` and `gui/{physics,panels}/<name>.py` mentions a specific physics.

## A physics

`physics/<name>/` contains:

| module | content |
|---|---|
| `model.py` | settings dataclasses with `to_dict()` (`tomlio.without_defaults`) / `from_dict()`, the physics' own study settings, `validate_settings(project, ev, materials)` |
| `solver.py` | the solver: `run(quantities)` returns results; caches mesh and matrices between sweep steps |
| results | `FieldResults` subclass (implement `point_values`, `compute`, `save`), or an own results type with `write_outputs(project, dir, plots)` (e.g. solid `modal.ModalResults`) |
| `fields.py` | the field variables of quantities: `at_quadrature(quad, …)` and `at_points(sampler, …)` return name → array, or name → provider (computed on first use) |
| `plots.py` | plot kinds, titles, field view, defaults |
| `templates.py` | `TEMPLATES`: name → `templates.Template` |
| `__init__.py` | `SPEC`, a `PhysicsSpec` subclass tying it together (section, field and scalar names, validation, solver, plots, quantity help, templates) |

Its GUI lives in `gui/panels/<name>.py` (panels and the actions they trigger) and `gui/physics/<name>.py`
(a `PhysicsUI` subclass: tree nodes, which panel belongs to which node, context menu, condition highlights,
how a solution's field is drawn, plot help).

Field results are the common case (one field per variable value and time step): they get quantities vs
time or the swept variable, line profiles, CSV/NPZ output and the GUI's selection and field view for free.

## Physics of a model

`project.physics_kind` (`[project] physics` in the file) names the model's physics setup. Today a setup is one
physics; `physics.active(project)` lists the physics of the setup, and the GUI tree shows them under
"Physics" (each with its conditions), so a coupled setup (thermal expansion: heat transfer driving solid
mechanics; Joule heating: electric currents heating) only has to add a setup name that lists several physics,
each keeping its own settings table (`[heat]`, `[solid]`, …), plus its coupling settings. Model files of a
single physics stay valid.

## Adding a physics

1. `physics/<name>/` as above; register `SPEC` in `physics/__init__.py`.
2. Add its settings as a field of `Project` (attribute name = `SPEC.section` = the TOML table = the kind).
3. `gui/panels/<name>.py` and `gui/physics/<name>.py`; register the adapter in `gui/physics/__init__.py`
   (an assertion there checks that every physics has one).
4. Templates in `physics/<name>/templates.py`; `tests/test_templates.py` solves every template.
5. An analytic test of the solver, and the input fields in `docs/input-format.md`.

## Conventions that keep it maintainable

- The model file is the model: the GUI edits `Project` and never holds state that cannot be saved.
- Everything internal and every result file is SI; the preferred units (`project.length_unit`, `frequency_unit`, `time_unit`; `project.display_units`) are display only.
- A physics never imports another physics. Shared helpers go to the shared layers (`selection.py`,
  `results/`, `gui/panels/common.py`).
- The model file has the shared sections (project, variables, materials, geometry, `[lattice]`, mesh,
  `[study]` with the variable sweep and linear solver, output, plots) and one table per physics named
  after it (`[solid]`, `[heat]`), holding all its settings including its own study. Only the active
  physics' table is saved; values equal to their defaults are left out and short arrays stay on one line
  (`tomlio.py`).
