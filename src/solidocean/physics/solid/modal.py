"""Eigenfrequency problem of solid mechanics: modes at every point of a k-path and/or a variable sweep, the
modal results (frequencies, quantities per mode, mode shapes kept in memory) and quantities evaluated on them.
Plots: results/modal.py."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np

from ... import fields, outputs
from ...expressions import Evaluator, UNITS
from ..base import Cancelled
from . import fields as solid_fields

MODE_BUDGET = 1.5e9  # bytes of mode shapes kept in memory for post-processing


@dataclass
class ModalResults:
    freqs: np.ndarray                 # (n_outer, n_inner, n_eigs) Hz
    k: np.ndarray                     # (n_outer, n_inner, 3) rad/m
    x: np.ndarray                     # (n_inner,) plot coordinate for the inner sweep
    x_label: str
    ticks: list[tuple[float, str]] = field(default_factory=list)
    outer_name: str = ""
    outer_values: np.ndarray = field(default_factory=lambda: np.zeros(1))
    info: dict = field(default_factory=dict)
    # derived quantities: expression -> (n_outer, n_inner, n_eigs)
    quantities: dict = field(default_factory=dict)
    # mode shapes kept in memory (not saved): (i, j) -> (n_eigs, N, 3) complex64; meshes[i] for sweep value i
    modes: dict = field(default_factory=dict, repr=False)
    meshes: dict = field(default_factory=dict, repr=False)
    cels: dict = field(default_factory=dict, repr=False)      # (E, 6, 6) stiffness per element (stresses)
    # wavevector mesh: the triangles (T, 3) joining the k points of the inner axis and their reduced
    # coordinates (n_inner, 2). Both are the same for every value of the outer sweep (only the reciprocal
    # vectors scale with it), so the connectivity is stored once.
    ktris: np.ndarray | None = None
    kred: np.ndarray | None = None
    # Fourier components of the fields of every mode over the unit cell (fourier.py), when the problem asks
    # for them: (n_outer, n_inner, n_eigs, n_G, n_fields) complex, the integer orders of the reciprocal
    # vectors (n_G, 3) and the vectors themselves (n_outer, n_G, 3) rad/m (a swept lattice scales them).
    fourier: np.ndarray | None = None
    fourier_orders: np.ndarray | None = None
    fourier_g: np.ndarray | None = None

    @property
    def is_kpath(self) -> bool:
        """The inner axis sweeps the wavevector: a k-path or a list of wavevectors (a mesh of the zone is its
        own kind, and a single solve has no inner axis). `ticks` names its points, a list has none."""
        return self.freqs.shape[1] > 1 and not self.is_kmesh

    @property
    def is_kmesh(self) -> bool:
        return self.ktris is not None

    @property
    def has_all_modes(self) -> bool:
        return len(self.modes) == self.freqs.shape[0] * self.freqs.shape[1]

    def values(self, quantity: str) -> np.ndarray | None:
        """Array of a quantity if available ('freq' always is)."""
        q = quantity.strip()
        return self.freqs if q == "freq" else self.quantities.get(q)

    def save_table(self, path: Path, quantity: str, unit_scale: float = 1.0, unit: str = "") -> Path:
        """CSV of a quantity: one row per solve, one column per mode."""
        vals = self.values(quantity)
        path = Path(path)
        n_eigs = vals.shape[2]
        label = f" [{unit}]" if unit else ""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(([self.outer_name] if self.outer_name else []) +
                       ["kx [rad/m]", "ky [rad/m]", "kz [rad/m]", self.x_label] +
                       [f"{quantity} mode {b + 1}{label}" for b in range(n_eigs)])
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    w.writerow(([f"{self.outer_values[i]:.10g}"] if self.outer_name else []) +
                               [f"{v:.10g}" for v in self.k[i, j]] + [f"{self.x[j]:.10g}"] +
                               [f"{v / unit_scale:.10g}" for v in vals[i, j]])
        return path

    def save_csv(self, path: Path, freq_unit: str = "GHz") -> Path:
        """One row per solve: [variable], k (rad/m), path coordinate, frequencies in freq_unit."""
        path = Path(path)
        scale = UNITS[freq_unit]
        n_eigs = self.freqs.shape[2]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            head = ([self.outer_name] if self.outer_name else []) + \
                ["kx [rad/m]", "ky [rad/m]", "kz [rad/m]", self.x_label] + \
                [f"f{i + 1} [{freq_unit}]" for i in range(n_eigs)]
            w.writerow(head)
            for i in range(self.freqs.shape[0]):
                for j in range(self.freqs.shape[1]):
                    row = ([f"{self.outer_values[i]:.10g}"] if self.outer_name else []) + \
                        [f"{v:.10g}" for v in self.k[i, j]] + [f"{self.x[j]:.10g}"] + \
                        [f"{f / scale:.10g}" for f in self.freqs[i, j]]
                    w.writerow(row)
        return path

    def save(self, directory: Path, freq_unit: str = "GHz", contents=outputs.CONTENTS) -> list[Path]:
        """dispersion.csv ('tables': one row per solve) and results.npz ('data': frequencies,
        wavevectors and quantities in SI)."""
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        if "tables" in contents:
            files.append(self.save_csv(directory / "dispersion.csv", freq_unit))
        if "data" in contents:
            npz_path = directory / "results.npz"
            names = list(self.quantities)
            mesh = {"ktris": self.ktris, "kred": self.kred} if self.is_kmesh else {}
            if self.fourier is not None:
                mesh.update(fourier=self.fourier, fourier_orders=self.fourier_orders, fourier_g=self.fourier_g)
            np.savez(npz_path, freqs=self.freqs, k=self.k, x=self.x, outer_values=self.outer_values, **mesh,
                     **{f"q{n}": self.quantities[q] for n, q in enumerate(names)},
                     meta=json.dumps({"x_label": self.x_label, "ticks": self.ticks, "quantities": names,
                                      "outer_name": self.outer_name, "info": self.info}))
            files.append(npz_path)
        return files

    def write_outputs(self, project, directory: Path, plots) -> list[Path]:
        """Everything a run writes, as [output] contents says: dispersion.csv, results.npz, a CSV per
        other quantity, and the model with its run record. Plots stay in the GUI ("Save plot…")."""
        from ...results.style import slug
        out = project.output
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = self.save(directory, project.frequency_unit, out.contents)
        if out.wants("tables"):
            for pl in plots:
                if pl.kind in ("spectrum", "conductance"):
                    from . import spectra
                    if self.is_kmesh:
                        written.append(spectra.save_table(self, project, pl, directory / f"{slug(pl.name)}.csv"))
                    continue
                extra = pl.color_by if pl.kind == "colored" else pl.quantity
                # dispersion.csv already holds the frequencies; a quantity of a plot added after the run
                # has no values yet
                if extra.strip() == "freq" or self.values(extra) is None:
                    continue
                path = directory / f"{slug(pl.name)}.csv"
                if path in written:      # a plot named like the frequencies file
                    path = directory / f"{slug(pl.name)}_{slug(extra)}.csv"
                written.append(self.save_table(path, extra))
            if self.fourier is not None:
                from . import fourier as fourier_mod
                written.append(fourier_mod.save_csv(self, project, directory / "fourier.csv"))
                written.append(fourier_mod.save_degeneracy_csv(self, directory / "fourier_degeneracy.csv"))
        if out.wants("model"):
            written += outputs.write_record(project, directory, self)
        return written

    @classmethod
    def load(cls, path: Path) -> "ModalResults":
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        quantities = {q: d[f"q{n}"] for n, q in enumerate(meta.get("quantities", []))}
        return cls(d["freqs"], d["k"], d["x"], meta["x_label"], [tuple(t) for t in meta["ticks"]],
                   meta["outer_name"], d["outer_values"], meta["info"], quantities,
                   ktris=d["ktris"] if "ktris" in d else None, kred=d["kred"] if "kred" in d else None,
                   **{n: d[n] for n in ("fourier", "fourier_orders", "fourier_g") if n in d})


class NeedsModes(Exception):
    """The quantity uses displacement fields but the mode shapes are not in memory."""


def compute_quantity(results: ModalResults, project, quantity: str) -> np.ndarray:
    """Evaluate a quantity from frequencies, wavevectors and (if needed) stored mode shapes."""
    q = quantity.strip()
    if q == "freq":
        return results.freqs
    needs = fields.uses_fields(q)
    if needs and not results.has_all_modes:
        raise NeedsModes(q)
    base = project.evaluator()
    out = np.full(results.freqs.shape, np.nan)
    quads = {}
    for i in range(results.freqs.shape[0]):
        ev = base.with_overrides({results.outer_name: results.outer_values[i]}) if results.outer_name else base
        cel = results.cels.get(i)
        if needs:
            mesh = results.meshes[i]
            quad = quads.setdefault(id(mesh), fields.Quadrature.of(mesh))
        for j in range(results.freqs.shape[1]):
            u = results.modes.get((i, j)) if needs else None
            for b in range(results.freqs.shape[2]):
                ub = solid_fields.normalise(u[b].astype(complex), quad) if needs else None
                out[i, j, b] = fields.evaluate(q, _scalars(ev, results.freqs[i, j, b], results.k[i, j], b),
                                               quad if needs else None,
                                               partial(solid_fields.at_quadrature, u=ub, cel=cel) if needs else None)
    results.quantities[q] = out
    return out


def _scalars(ev: Evaluator, f: float, k: np.ndarray, b: int) -> dict:
    return {**ev.values, "freq": f, "omega": 2 * np.pi * f, "kx": k[0], "ky": k[1], "kz": k[2],
            "k": float(np.linalg.norm(k)), "mode": b + 1}


def kpath_vectors(project, ev: Evaluator):
    """k vectors (n, 3), cumulative path length (n,), ticks for the configured wavevector sweep: the points
    of the named path, or exactly the listed wavevectors (which have no named ticks)."""
    from ... import lattice as lat
    kp = project.solid.problem.active_kpath
    recip = lat.reciprocal(lat.slots(project.lattice, ev))

    def k_of(reduced):
        return sum((r * b for r, b in zip(reduced, recip) if b is not None), np.zeros(3))

    if kp.mode == "list":
        ks = [k_of(ev.vector(point)) for point in kp.k]
        xs = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))]) if len(ks) > 1 \
            else np.zeros(len(ks))
        return np.array(ks), xs, []

    corners = [k_of(np.asarray(kp.points[name], dtype=float)) for name in kp.path]
    ks, xs, ticks = [corners[0]], [0.0], [(0.0, kp.path[0])]
    dist = 0.0
    for s in range(1, len(corners)):
        a, c = corners[s - 1], corners[s]
        seg = np.linalg.norm(c - a)
        for t in np.linspace(0, 1, kp.n + 1)[1:]:
            ks.append(a + t * (c - a))
            xs.append(dist + t * seg)
        dist += seg
        ticks.append((dist, kp.path[s]))
    return np.array(ks), np.array(xs), ticks


def kmesh_vectors(project, ev: Evaluator):
    """The k points (P, 3) of the configured wavevector mesh, their reduced coordinates (P, 2) and the
    triangles (T, 3) joining them. The mesh covers the irreducible wedge of the 2D Brillouin zone or the
    whole zone (lattice.zone_mesh)."""
    from ... import lattice as lat
    km = project.solid.problem.active_kmesh
    recip = [b for b in lat.reciprocal(lat.slots(project.lattice, ev)) if b is not None]
    if len(recip) != 2:
        raise ValueError("a wavevector mesh needs a lattice periodic in exactly two directions")
    b1, b2 = recip
    reduced, tris = lat.zone_mesh(b1, b2, km.n, km.region, lat.classify(project.lattice, ev))
    return reduced[:, :1] * b1 + reduced[:, 1:] * b2, reduced, tris


def run(solver, quantities=(), store_modes: bool = False) -> ModalResults:
    """Eigenfrequency problem: for every variable-sweep value, the k-path (or the single k) is solved; quantities
    are expressions evaluated for every mode; store_modes keeps mode shapes in memory (within MODE_BUDGET bytes)
    for later evaluation."""
    p = solver.project
    quantities = [q.strip() for q in quantities if q.strip() and q.strip() != "freq"]
    field_qs = [q for q in quantities if fields.uses_fields(q)]
    errors = p.validate()
    if errors:
        raise ValueError("invalid project:\n  " + "\n  ".join(errors))
    base = p.evaluator()
    t_start = time.perf_counter()

    sw = p.problem.active_sweep
    if sw:
        outer_name, outer_values = sw.variable, base.array(sw.values)
    else:
        outer_name, outer_values = "", np.array([np.nan])

    n_eigs = p.solid.problem.num_eigs
    if p.solid.problem.active_kpath:
        n_inner = len(kpath_vectors(p, base)[0])
    elif p.solid.problem.active_kmesh:
        n_inner = len(kmesh_vectors(p, base)[0])
    else:
        n_inner = 1
    total = len(outer_values) * n_inner
    freqs = np.full((len(outer_values), n_inner, n_eigs), np.nan)
    kvecs = np.zeros((len(outer_values), n_inner, 3))
    qvals = {q: np.full(freqs.shape, np.nan) for q in quantities}
    modes, meshes, cels = {}, {}, {}
    keep = store_modes
    step = 0
    x, x_label, ticks = np.zeros(1), "", []
    kred, ktris = None, None
    fo = p.solid.problem.active_fourier
    fcoef = forders = fg = None
    if fo:
        from . import fourier as fourier_mod
        forders = fourier_mod.orders_grid(fo.orders)        # the same orders at every value of the sweep
        fcoef = np.full((len(outer_values), n_inner, n_eigs, len(forders), len(fourier_mod.FIELDS)),
                        np.nan, dtype=complex)
        fg = np.zeros((len(outer_values), len(forders), 3))
        solver.log(f"Fourier components of every mode at {len(forders)} reciprocal vectors")
    for i, val in enumerate(outer_values):
        ev = base.with_overrides({outer_name: val}) if outer_name else base
        if outer_name:
            solver.log(f"{outer_name} = {val:.6g}")
        if p.solid.problem.active_kpath:
            ks, x, ticks = kpath_vectors(p, ev)
            # a list of wavevectors has the same coordinate (the distance travelled in k space) but no path
            x_label = "k path [rad/m]" if ticks else "wavevector [rad/m]"
        elif p.solid.problem.active_kmesh:
            ks, kred, ktris = kmesh_vectors(p, ev)
            x, x_label = np.arange(len(ks), dtype=float), "k point"
        else:
            ks = solver.k_of(ev)[None]
            x, x_label = np.zeros(1), "k"
        if fo:
            fg[i] = fourier_mod.vectors(p, ev, fo.orders)[1]
        for j, k in enumerate(ks):
            if solver.cancel.is_set():
                raise Cancelled()
            if keep or field_qs or fo:
                f, u = solver.solve_point(ev, k, modes=True)
                mesh = solver._model.mesh
                meshes[i], cels[i] = mesh, solver._model.cel
                if keep and not modes and u.nbytes / 2 * total > MODE_BUDGET:
                    keep = False
                    solver.log(f"Mode shapes too large to keep in memory "
                             f"({u.nbytes / 2 * total / 1e9:.1f} GB); they will be recomputed when needed")
                if keep:
                    modes[(i, j)] = u.astype(np.complex64)
                quad = None
                if field_qs or fo:
                    quad = solver.__dict__.setdefault("_quad", {}).get(id(mesh))
                    if quad is None:
                        quad = solver._quad[id(mesh)] = fields.Quadrature.of(mesh)
            else:
                f, u, quad = solver.solve_point(ev, k), None, None
            freqs[i, j, :len(f)] = f
            kvecs[i, j] = k
            if fo:
                normalised = [fourier_mod.mass_normalise(u[b], solver._model.M) for b in range(len(f))]
                fcoef[i, j, :len(f)] = fourier_mod.components(quad, normalised, solver._model.cel, k, fg[i])
            for q in quantities:
                for b in range(len(f)):
                    ub = solid_fields.normalise(u[b], quad) if q in field_qs else None
                    qvals[q][i, j, b] = fields.evaluate(
                        q, _scalars(ev, f[b], k, b), quad,
                        partial(solid_fields.at_quadrature, u=ub, cel=solver._model.cel) if ub is not None else None)
            step += 1
            solver.progress(step, total)
        solver.log(f"  lowest: {', '.join(f'{v / 1e9:.4g}' for v in freqs[i, 0, :min(6, n_eigs)])} GHz")
    from ...units import dimension
    outer_dim = dimension(p.variables[outer_name], base) if outer_name else None
    info = {"project": p.name, "elapsed_s": round(time.perf_counter() - t_start, 2),
            "n_dofs": int(solver._model.constraints.n_free) if solver._model else 0,
            "outer_dim": list(outer_dim) if outer_dim else None}
    if fo:
        info["fourier"] = {"orders": [int(o) for o in fo.orders], "vectors": len(forders),
                           "fields": list(fourier_mod.FIELDS), "normalisation": fourier_mod.NORMALISATION,
                           "degeneracy_tol": fourier_mod.DEGENERACY_TOL}
        groups = fourier_mod.degeneracies(freqs)
        n_degenerate = int(np.count_nonzero(groups > 1))
        if n_degenerate:
            solver.log(f"{n_degenerate} of {groups.size} modes are degenerate: the components of a degenerate "
                       f"mode depend on the basis the eigensolver picked, only their sum over the group does not")
    solver.log(f"Done in {info['elapsed_s']} s")
    return ModalResults(freqs, kvecs, x, x_label, ticks, outer_name, outer_values, info, qvals,
                   modes if keep else {}, meshes, cels, ktris, kred, fcoef, forders, fg)
