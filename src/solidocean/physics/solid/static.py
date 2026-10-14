"""Stationary and frequency-domain problems of solid mechanics: displacement, strain and stress under loads and
prescribed displacements (constant, or their harmonic amplitudes: complex fields), one solution per
variable-sweep value. Plotted like any field results (results/field.py); quantities are expressions of the
displacement, strains and stresses (physics/solid/fields.py)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ... import fields, outputs
from ...results.field import FieldResults
from ..base import Cancelled
from . import fields as solid_fields


@dataclass
class StaticResults(FieldResults):
    U: list = field(default_factory=list)                    # per variable value: (N, 3) displacement (m)
    cels: dict = field(default_factory=dict, repr=False)     # per variable value: (E, 6, 6) element stiffness
    timed: bool = False                                      # one solution per value: no time column

    def point_values(self, i: int, j: int, expr: str, scalars, sampler) -> np.ndarray:
        return fields.evaluate_points(expr, scalars, sampler, solid_fields.at_points(sampler, self.U[i], self.cels[i]))

    def compute(self, project, quantity: str) -> np.ndarray:
        return compute_quantity(self, project, quantity)

    def save(self, directory: Path, contents=outputs.CONTENTS) -> list[Path]:
        """quantities.csv ('tables') and displacement.npz ('data': nodal displacements and meshes, SI)."""
        directory.mkdir(parents=True, exist_ok=True)
        files = self.save_quantities(directory) if "tables" in contents else []
        if "data" not in contents:
            return files
        names = list(self.quantities)
        arrays = {}
        for i, u in enumerate(self.U):
            arrays[f"u{i}"] = u
            if i in self.meshes:
                arrays[f"points{i}"] = self.meshes[i].points
                arrays[f"cells{i}"] = self.meshes[i].cells
        path = directory / "displacement.npz"
        np.savez(path, outer_values=self.outer_values, **{f"q{n}": self.quantities[q] for n, q in enumerate(names)},
                 **arrays, meta=json.dumps({"kind": self.kind, "outer_name": self.outer_name, "quantities": names,
                                            "info": self.info}))
        return files + [path]


def run(solver, quantities=()) -> StaticResults:
    """Stationary problem (with an optional variable sweep) for solver.project."""
    p = solver.project
    errors = p.validate()
    if errors:
        raise ValueError("invalid project:\n  " + "\n  ".join(errors))
    base = p.evaluator()
    t_start = time.perf_counter()
    sw = p.problem.active_sweep
    outer_name, outer_values = (sw.variable, base.array(sw.values)) if sw else ("", np.array([np.nan]))
    quantities = [q.strip() for q in quantities if q.strip()]
    res = StaticResults(outer_name=outer_name, outer_values=outer_values,
                        kind="frequency" if p.solid.problem.harmonic else "stationary")
    qvals = {q: np.full((len(outer_values), 1), np.nan) for q in quantities}
    quads = {}
    for i, val in enumerate(outer_values):
        if solver.cancel.is_set():
            raise Cancelled()
        ev = base.with_overrides({outer_name: val}) if outer_name else base
        if outer_name:
            solver.log(f"{outer_name} = {val:.6g}")
        u = solver.solve_harmonic(ev) if p.solid.problem.harmonic else solver.solve_static(ev)
        m = solver._model
        res.U.append(u)
        res.meshes[i], res.cels[i] = m.mesh, m.cel
        if quantities:
            quad = quads.setdefault(id(m.mesh), fields.Quadrature.of(m.mesh))
            values = solid_fields.at_quadrature(quad, u, m.cel)
            for q in quantities:
                qvals[q][i, 0] = fields.evaluate(q, ev.values, quad, values)
        solver.log(f"  largest displacement {np.linalg.norm(np.abs(u), axis=1).max():.4g} m"
                   + (" (amplitude)" if np.iscomplexobj(u) else ""))
        solver.progress(i + 1, len(outer_values))
    from ...units import dimension
    outer_dim = dimension(p.variables[outer_name], base) if outer_name else None
    res.quantities = qvals
    res.info = {"project": p.name, "elapsed_s": round(time.perf_counter() - t_start, 2),
                "n_dofs": int(solver._model.constraints.n_free) if solver._model else 0,
                "outer_dim": list(outer_dim) if outer_dim else None}
    solver.log(f"Done in {res.info['elapsed_s']} s")
    return res


def compute_quantity(results: StaticResults, project, quantity: str) -> np.ndarray:
    """Evaluate a quantity on the stored displacement fields."""
    q = quantity.strip()
    base = project.evaluator()
    out = np.full((len(results.outer_values), 1), np.nan)
    quads = {}
    for i, u in enumerate(results.U):
        ev = base.with_overrides({results.outer_name: results.outer_values[i]}) if results.outer_name else base
        quad = quads.setdefault(id(results.meshes[i]), fields.Quadrature.of(results.meshes[i]))
        out[i, 0] = fields.evaluate(q, ev.values, quad, solid_fields.at_quadrature(quad, u, results.cels[i]))
    results.quantities[q] = out
    return out
