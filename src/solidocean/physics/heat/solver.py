"""Heat transfer solver: stationary or transient temperature fields, with variable sweeps.

Execution: an optional outer variable sweep; for each value the mesh and the matrices are (re)built only
when the geometry, mesh or material properties change. Quantities (expressions of T, its gradient and the
heat flux, physics/heat/fields.py) are evaluated for every saved time step.

Nonlinear models (radiating boundaries, temperature-dependent properties) are solved by iteration: the
properties are taken at the previous iterate (Picard) and the radiation is linearised about it (Newton),
until the temperature change is below the problem's `tol`; in a transient problem at every time step.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import scipy.sparse as sp

from ... import fields
from ... import meshing, outputs, selection
from ...expressions import Evaluator
from ...fem import heat
from ...geometry import evaluate_features
from ...results.field import FieldResults
from ..base import Cancelled, LogFn
from . import fields as heat_fields
from . import interfaces
from .interfaces import Contacts
from .properties import Conductivity, Properties

SIGMA = 5.670374419e-8      # Stefan-Boltzmann constant, W/(m^2 K^4)


@dataclass
class HeatModel:
    mesh: meshing.Mesh
    K: object                   # conductivity matrix incl. thermal contacts (temperature-dependent: at
    C: object                   # REFERENCE_T, see matrices()); capacity matrix (likewise)
    cond: Conductivity          # conductivity of every element, for the heat flux
    tris: np.ndarray            # boundary triangles
    props: Properties
    owner: np.ndarray | None = None   # element of each boundary triangle
    quad: fields.Quadrature | None = None
    assembler: heat.Assembler | None = None   # re-assembly of temperature-dependent matrices
    contacts: Contacts | None = None
    volume: tuple = ()          # (K, C) of the domains alone, when contacts are re-assembled separately


@dataclass
class Radiation:
    tris: np.ndarray            # boundary triangles
    eps: float                  # emissivity
    T_amb: object               # ambient temperature (K), or a function (t0, t1) -> it


@dataclass
class HeatResults(FieldResults):
    T: list = field(default_factory=list)                   # per variable value: (n_t, N) nodal temperatures (K)
    kels: dict = field(default_factory=dict, repr=False)    # per variable value: properties.Conductivity

    def point_values(self, i: int, j: int, expr: str, scalars, sampler) -> np.ndarray:
        return fields.evaluate_points(expr, scalars, sampler, heat_fields.at_points(sampler, self.T[i][j], self.kels[i]))

    def compute(self, project, quantity: str) -> np.ndarray:
        return compute_quantity(self, project, quantity)

    def save(self, directory: Path, contents=outputs.CONTENTS) -> list[Path]:
        """quantities.csv ('tables') and temperature.npz ('data': nodal temperatures, times and meshes, SI)."""
        directory.mkdir(parents=True, exist_ok=True)
        files = self.save_quantities(directory) if "tables" in contents else []
        if "data" not in contents:
            return files
        names = list(self.quantities)
        arrays = {}
        for i, T in enumerate(self.T):
            arrays[f"T{i}"] = T
            if i in self.meshes:
                arrays[f"points{i}"] = self.meshes[i].points
                arrays[f"cells{i}"] = self.meshes[i].cells
        path = directory / "temperature.npz"
        np.savez(path, times=self.times, outer_values=self.outer_values,
                 **{f"q{n}": self.quantities[q] for n, q in enumerate(names)}, **arrays,
                 meta=json.dumps({"kind": self.kind, "outer_name": self.outer_name, "quantities": names,
                                  "info": self.info}))
        return files + [path]


class HeatSolver:
    def __init__(self, project, log: LogFn = print, progress=None, cancel: threading.Event | None = None):
        self.project = project
        self.log = log
        self.progress = progress or (lambda i, n: None)
        self.cancel = cancel or threading.Event()
        self._key = self._mesh_key = None
        self._mesh = self._split = None
        self._model: HeatModel | None = None

    def _geometry_key(self, ev: Evaluator):
        """Changes only when the mesh must be rebuilt (geometry, mesh settings)."""
        p = self.project
        feats = [(f.feature.type, f.feature.op, f.feature.material,
                  tuple((k, np.round(np.atleast_1d(v), 15).tobytes()) for k, v in sorted(f.values.items())))
                 for f in evaluate_features(p, ev)]
        mesh = p.mesh.key(ev)
        from ... import lattice as lat
        periods = tuple((n, tuple(np.round(v, 18))) for n, v in lat.translations(p.lattice, ev))
        return repr((feats, mesh, periods))

    def model(self, ev: Evaluator) -> HeatModel:
        """Mesh and matrices; a sweep of material properties (e.g. k) re-assembles on the same mesh."""
        mesh_key = self._geometry_key(ev)
        p = self.project
        t0 = time.perf_counter()
        if mesh_key != self._mesh_key or self._mesh is None:
            self.log("Meshing…")
            mesh = meshing.generate(p, ev)
            tris, owner = heat.boundary_faces(mesh.cells, mesh.order, owners=True)
            self._mesh, self._mesh_key = (mesh, tris, owner), mesh_key
            self.log(f"Mesh: {mesh.info['n_elements']} elements, {mesh.n_nodes} nodes (temperature DOFs), "
                     f"order {mesh.order} [{time.perf_counter() - t0:.1f} s]")
        mesh, tris, owner = self._mesh
        split = None
        if p.heat.interfaces:               # duplicate the interface nodes: T may jump across them
            split_key = (mesh_key, interfaces.selection_key(p))
            if self._split is None or self._split[0] != split_key:
                self._split = (split_key, interfaces.split(mesh, p.heat.interfaces, tris, owner))
                s = self._split[1]
                self.log(f"Thermal contacts: {s.mesh.n_nodes - mesh.n_nodes} interface nodes duplicated")
            split = self._split[1]
            mesh, tris, owner = split.mesh, split.tris, split.owner
        props = Properties(p, ev, mesh.material_names)
        contacts = Contacts(p, ev, split)
        key = (mesh_key, interfaces.selection_key(p), props.key(), contacts.key())
        if key == self._key and self._model is not None:
            return self._model
        k_list, rhoc = props.constant()     # temperature-dependent ones at REFERENCE_T, re-assembled per iterate
        K, C = heat.assemble(mesh.points, mesh.cells, mesh.cell_domain, k_list, rhoc, mesh.order)
        volume = (K, C)
        if contacts.items:
            Kc, Cc = contacts.matrices(mesh.n_nodes)
            K, C = K + Kc, C + Cc
        quad = self._model.quad if self._model is not None and self._model.mesh is mesh else None
        self._model = HeatModel(mesh, K, C, Conductivity(props, mesh.cell_domain), tris, props, owner=owner,
                                quad=quad, contacts=contacts, volume=volume)
        self._key = key
        return self._model

    def matrices(self, m: HeatModel, T: np.ndarray | None, capacity: bool = True):
        """K and C (None unless `capacity`) with the properties at the temperatures T; only those that depend
        on T are re-assembled (domains and thermal contacts separately)."""
        props, ct = m.props, m.contacts
        re_k = props.k_varies or ct.k_varies
        re_c = capacity and (props.c_varies or ct.c_varies)
        if not (re_k or re_c):
            return m.K, (m.C if capacity else None)
        mesh = m.mesh
        Kv, Cv = m.volume
        if (props.k_varies or (capacity and props.c_varies)):
            if m.assembler is None:
                m.assembler = heat.Assembler(mesh.points, mesh.cells, mesh.order)
            Tq = m.assembler.at_quadrature(T)
            domain = np.broadcast_to(mesh.cell_domain[:, None], Tq.shape)
            re_kv, re_cv = props.k_varies, capacity and props.c_varies
            K, C = m.assembler.matrices(props.conductivity(domain, Tq) if re_kv else None,
                                        props.capacity(domain, Tq) if re_cv else None)
            Kv, Cv = (K if re_kv else Kv), (C if re_cv else Cv)
        if ct.items:
            Kc, Cc = ct.matrices(mesh.n_nodes, T if ct.varies else None, capacity)
            Kv = Kv + Kc
            Cv = Cv + Cc if capacity else Cv
        return Kv, (Cv if capacity else None)

    def radiation_terms(self, m: HeatModel, radiation: list[Radiation], T: np.ndarray, t0=0.0, t1=0.0):
        """Matrix and load of the radiating boundaries linearised about T: emissivity sigma (T_amb^4 - T^4)
        ~ emissivity sigma (T_amb^4 + 3 T0^4) - 4 emissivity sigma T0^3 T (exact at T = T0)."""
        mesh = m.mesh
        n = mesh.n_nodes
        M, f = sp.csr_matrix((n, n)), np.zeros(n)
        for r in radiation:
            T0 = np.abs(heat.surface_at_points(mesh.points, r.tris, mesh.order, T))
            Ta = r.T_amb(t0, t1) if callable(r.T_amb) else r.T_amb
            Mr, fr = heat.surface_terms(mesh.points, r.tris, mesh.order, coeff=4 * r.eps * SIGMA * T0 ** 3,
                                        load=r.eps * SIGMA * (Ta ** 4 + 3 * T0 ** 4), n_nodes=n)
            M, f = M + Mr, f + fr
        return M, f

    def _faces(self, m: HeatModel, c, ev: Evaluator) -> np.ndarray:
        """Boundary triangles of a condition's selection (see selection.py)."""
        return m.tris[selection.select(m.mesh, selection.Boundary(m.tris, m.owner), c, ev)]

    def _profile(self, ev: Evaluator, expr, pulse=None, transient=True):
        """Constant value, or a function (t0, t1) -> step value for time-dependent expressions / pulses."""
        timed = bool(re.search(r"\btime\b", str(expr)))
        pulsed = transient and pulse is not None and bool(pulse.shape)
        if not transient or not (timed or pulsed):
            return ev.with_overrides({"time": 0.0})(expr) if timed else ev(expr)
        st = self.project.heat.problem
        theta = st.theta
        const = None if timed else ev(expr)
        sine = pulsed and pulse.shape == "sine"
        if pulsed:
            args = (pulse.shape, ev(pulse.start), 0.0 if sine else ev(pulse.duration),
                    ev(pulse.period) if pulse.period else 0.0, ev(st.t_end))
            amplitude = ev(pulse.amplitude) if sine else 0.0

        def value(t0, t1):
            v = const if const is not None else \
                theta * ev.with_overrides({"time": t1})(expr) + (1 - theta) * ev.with_overrides({"time": t0})(expr)
            if sine:              # oscillation about the value
                v += amplitude * heat.pulse_average(args[0], t0, t1, *args[1:])
            elif pulsed:          # pulse multiplying the value (its peak)
                v *= heat.pulse_average(args[0], t0, t1, *args[1:])
            return v
        return value

    @staticmethod
    def _amplitude(ev: Evaluator, c) -> float:
        """Oscillation amplitude of a flux or source in the frequency domain."""
        expr = c.pulse.amplitude if c.pulse.shape == "sine" else c.value
        return ev.with_overrides({"time": 0.0})(expr)

    def system(self, ev: Evaluator, m: HeatModel, transient: bool = False, harmonic: bool = False):
        """Boundary matrix (convection; the conduction matrix is added by the caller), load, fixed nodes and
        their temperatures, and the radiating boundaries, for the current conditions and sources.

        load is a vector, or (transient, time-dependent) a function (t0, t1) -> vector; likewise the
        fixed temperatures are an array or a function t -> array.

        harmonic (frequency domain): the load holds only the oscillation amplitudes of the fluxes and sources
        (the sine amplitude of an oscillating one, else its value); ambient temperatures are steady."""
        p, mesh = self.project, m.mesh
        n = mesh.n_nodes
        A = sp.csr_matrix((n, n))
        b = np.zeros(n)
        radiation = []
        timed = []                                   # (unit load vector, profile function)
        fixed_of: dict[int, int] = {}                # node -> index into fixed_values
        fixed_values = []
        for c in p.heat.conditions:
            tris = self._faces(m, c, ev)
            if c.type == "temperature":
                fixed_values.append(self._profile(ev, c.value, transient=transient))
                for node in np.unique(tris):
                    fixed_of[int(node)] = len(fixed_values) - 1
            elif c.type == "flux":
                load = 1.0
                if c.profile.shape:                  # spatial factor at the surface quadrature points
                    xq = np.stack([heat.surface_at_points(mesh.points, tris, mesh.order, mesh.points[:, i])
                                   for i in range(3)], axis=-1)
                    load = c.profile.factor(ev, xq)
                _, f = heat.surface_terms(mesh.points, tris, mesh.order, load=load, n_nodes=n)
                value = self._amplitude(ev, c) if harmonic else self._profile(ev, c.value, c.pulse, transient)
                if callable(value):
                    timed.append((f, value))
                else:
                    b += value * f
            elif c.type == "convection":
                h = ev(c.h)
                M, f = heat.surface_terms(mesh.points, tris, mesh.order, coeff=h, load=h, n_nodes=n)
                A = A + M
                if harmonic:
                    continue
                Te = self._profile(ev, c.T_ext, transient=transient)
                if callable(Te):
                    timed.append((f, Te))
                else:
                    b += Te * f
            elif c.type == "radiation":
                radiation.append(Radiation(tris, ev(c.emissivity), self._profile(ev, c.T_ext, transient=transient)))
        for q in p.heat.sources:
            unit = [1.0 if name == q.material else 0.0 for name in mesh.material_names]
            if not any(unit):
                continue
            f = heat.source_vector(mesh.points, mesh.cells, mesh.cell_domain, unit, mesh.order,
                                   profile=(lambda X, q=q: q.profile.factor(ev, X)) if q.profile.shape else None)
            value = self._amplitude(ev, q) if harmonic else self._profile(ev, q.value, q.pulse, transient)
            if callable(value):
                timed.append((f, value))
            else:
                b += value * f
        fixed = np.array(sorted(fixed_of), dtype=np.int64)
        which = np.array([fixed_of[i] for i in fixed], dtype=np.int64)

        def T_fixed_at(t0, t1=None):
            vals = np.array([v(t0, t0) if callable(v) else v for v in fixed_values]) if fixed_values else np.zeros(0)
            return vals[which] if len(which) else np.zeros(0)

        if any(callable(v) for v in fixed_values):
            T_fixed = T_fixed_at
        else:
            T_fixed = T_fixed_at(0.0)
        if timed:
            def load(t0, t1):
                out = b.copy()
                for f, value in timed:
                    out += value(t0, t1) * f
                return out
            return A.tocsr(), load, fixed, T_fixed, radiation
        return A.tocsr(), b, fixed, T_fixed, radiation

    def solve(self, ev: Evaluator):
        """(times (n_t,), T (n_t, N)) for the current variable values."""
        p = self.project
        m = self.model(ev)
        st = p.heat.problem
        backend = p.problem.solver
        if st.type == "frequency":
            A_bc, b, fixed, _, radiation = self.system(ev, m, harmonic=True)
            return np.zeros(1), self._harmonic(ev, m, A_bc, b, fixed, radiation, backend)[None]
        A_bc, b, fixed, T_fixed, radiation = self.system(ev, m, transient=st.type == "transient")
        nonlinear = bool(radiation) or m.props.varies or m.contacts.varies
        if st.type == "stationary":
            sink = len(fixed) > 0 or radiation or any(c.type == "convection" for c in p.heat.conditions)
            if not sink:
                if nonlinear:
                    raise ValueError("a stationary model with temperature-dependent properties needs a heat sink "
                                     "(fixed temperature, convection or radiation)")
                return np.zeros(1), self._insulated_steady_state(ev, m, m.K + A_bc, b, backend)[None]
            if not nonlinear:
                return np.zeros(1), heat.stationary(m.K + A_bc, b, fixed, T_fixed, backend)[None]
            return np.zeros(1), self._stationary_nonlinear(ev, m, A_bc, b, fixed, T_fixed, radiation, backend)[None]
        t_start, t_end = ev(st.t_start), ev(st.t_end)
        dt = (t_end - t_start) / st.steps
        T0 = fields.evaluate_nodal(p.heat.initial, m.mesh.points, {**ev.values, "time": t_start})
        if nonlinear:
            saved, snaps = self._transient_nonlinear(m, A_bc, b, fixed, T_fixed, radiation, T0, t_start, dt, backend)
        else:
            saved, snaps = heat.transient(m.K + A_bc, m.C, b, fixed, T_fixed, T0, dt, st.steps, st.theta,
                                          st.save_every, cancel=self.cancel.is_set, progress=self._step_progress,
                                          t_start=t_start, backend=backend)
        if self.cancel.is_set():
            raise Cancelled()
        return t_start + saved * dt, snaps

    def _converged(self, T_new, T, it: int, where: str = "") -> bool:
        st = self.project.heat.problem
        change = float(np.max(np.abs(T_new - T)))
        if change <= st.tol * float(np.max(np.abs(T_new))):
            return True
        if it + 1 >= st.max_iter:
            raise ValueError(f"nonlinear heat transfer{where}: no convergence in {st.max_iter} iterations (last "
                             f"temperature change {change:.3g} K). Use smaller time steps, a better initial "
                             "temperature, or a larger heat.problem.max_iter")
        return False

    def _stationary_nonlinear(self, ev, m: HeatModel, A_bc, b, fixed, T_fixed, radiation, backend):
        """Fixed-point iteration from the initial values: properties at the previous iterate, radiation
        linearised about it."""
        T = fields.evaluate_nodal(self.project.heat.initial, m.mesh.points, {**ev.values, "time": 0.0})
        for it in range(self.project.heat.problem.max_iter):
            if self.cancel.is_set():
                raise Cancelled()
            K, _ = self.matrices(m, T, capacity=False)
            Mr, fr = self.radiation_terms(m, radiation, T)
            T_new = heat.stationary(K + A_bc + Mr, b + fr, fixed, T_fixed, backend)
            done = self._converged(T_new, T, it)
            T = T_new
            if done:
                self.log(f"  nonlinear: converged in {it + 1} iterations")
                return T
        return T

    def _transient_nonlinear(self, m: HeatModel, A_bc, b, fixed, T_fixed, radiation, T0, t_start, dt, backend):
        """theta scheme with the properties and the radiation of the end of each step, iterated to convergence
        within the step: C*(T1 - T0)/dt + theta R(T1) + (1 - theta) R(T0) = load, R(T) = K(T) T + A_bc T - q_rad(T)."""
        st = self.project.heat.problem
        theta = st.theta
        load = b if callable(b) else (lambda t0, t1: b)
        T_at = T_fixed if callable(T_fixed) else (lambda t: T_fixed)
        T = T0.copy()
        T[fixed] = T_at(t_start)
        saved, snaps, most = [0], [T.copy()], 0
        for step in range(1, st.steps + 1):
            if self.cancel.is_set():
                break
            t0, t1 = t_start + (step - 1) * dt, t_start + step * dt
            r0 = 0.0
            if theta < 1:                             # the explicit part: R(T0) at the start of the step
                K0, _ = self.matrices(m, T, capacity=False)
                Mr0, fr0 = self.radiation_terms(m, radiation, T, t0, t0)
                r0 = (K0 + A_bc + Mr0) @ T - fr0
            Tf, f = T_at(t1), load(t0, t1)
            Ti = T.copy()
            for it in range(st.max_iter):
                K, C = self.matrices(m, Ti)
                Mr, fr = self.radiation_terms(m, radiation, Ti, t1, t1)
                lhs = C / dt + theta * (K + A_bc + Mr)
                rhs = C @ T / dt + theta * fr - (1 - theta) * r0 + f
                T_new = heat.stationary(lhs, rhs, fixed, Tf, backend)
                done = self._converged(T_new, Ti, it, f" at t = {t1:.4g} s")
                Ti = T_new
                if done:
                    most = max(most, it + 1)
                    break
            T = Ti
            if step % st.save_every == 0 or step == st.steps:
                saved.append(step)
                snaps.append(T.copy())
            self._step_progress(step, st.steps)
        self.log(f"  nonlinear: at most {most} iterations per time step")
        return np.array(saved), np.array(snaps)

    def _harmonic(self, ev, m: HeatModel, A_bc, b, fixed, radiation, backend):
        """Complex amplitude of the temperature oscillation, (K + A_bc + R + i omega C) T = b, with T = 0 at fixed
        temperatures. A small-signal response: temperature-dependent properties are taken at the initial
        temperature and radiation is linearised about it (sink 4 eps sigma T0^3)."""
        T0 = self._T0(ev, m)
        K, C = self.matrices(m, T0)
        Mr, _ = self.radiation_terms(m, radiation, T0)
        omega = 2 * np.pi * ev(self.project.heat.problem.frequency)
        A = (K + A_bc + Mr).astype(complex) + 1j * omega * C
        return heat.stationary(A, b.astype(complex), fixed, np.zeros(len(fixed), complex), backend)

    def _T0(self, ev, m: HeatModel) -> np.ndarray:
        return fields.evaluate_nodal(self.project.heat.initial, m.mesh.points, {**ev.values, "time": 0.0})

    def _insulated_steady_state(self, ev: Evaluator, m: HeatModel, A, b, backend):
        """Stationary problem without a heat sink (no fixed temperature, no convection): the temperature level
        is set by energy conservation, i.e. the heat content stays that of the initial values. A steady state
        exists only if the net heat input is zero."""
        net, scale = float(b.sum()), float(np.abs(b).sum())   # sum of the load vector = total heat input (W)
        if scale > 0 and abs(net) > 1e-9 * scale:
            raise ValueError(
                f"no steady state: the model has no heat sink (fixed temperature or convection) but a net heat "
                f"input of {net:.4g} W, so its temperature keeps {'rising' if net > 0 else 'falling'}. Add a heat "
                "sink, or use a transient problem.")
        n = m.mesh.n_nodes
        T = heat.stationary(A, b - net / n, np.array([0]), np.array([0.0]), backend)  # level fixed below
        T0 = fields.evaluate_nodal(self.project.heat.initial, m.mesh.points, {**ev.values, "time": 0.0})
        c = m.C @ np.ones(n)                       # heat capacity per node
        T += (c @ T0 - c @ T) / c.sum()            # same heat content as the initial temperature
        self.log("  no heat sink: temperature level from the initial values (energy conservation)")
        return T

    def _step_progress(self, i, n):
        self.progress(self._done * n + i, self._total * n)

    def run(self, quantities=()) -> HeatResults:
        p = self.project
        errors = p.validate()
        if errors:
            raise ValueError("invalid project:\n  " + "\n  ".join(errors))
        base = p.evaluator()
        t_start = time.perf_counter()
        sw = p.problem.active_sweep
        outer_name, outer_values = (sw.variable, base.array(sw.values)) if sw else ("", np.array([np.nan]))
        quantities = [q.strip() for q in quantities if q.strip()]
        self._total = len(outer_values)
        temps, meshes, kels, qvals, times = [], {}, {}, {q: [] for q in quantities}, np.zeros(1)
        for i, val in enumerate(outer_values):
            self._done = i
            if self.cancel.is_set():
                raise Cancelled()
            ev = base.with_overrides({outer_name: val}) if outer_name else base
            if outer_name:
                self.log(f"{outer_name} = {val:.6g}")
            times, T = self.solve(ev)
            m = self._model
            temps.append(T)
            meshes[i], kels[i] = m.mesh, m.cond
            if np.iscomplexobj(T) and m.props.k_varies:      # flux of the oscillation: k at the initial temperature
                T0e = self._T0(ev, m)[m.mesh.cells[:, :4]].mean(axis=1)
                kels[i] = m.props.conductivity(m.mesh.cell_domain, T0e)
            if quantities and m.quad is None:
                m.quad = fields.Quadrature.of(m.mesh)
            for q in quantities:
                qvals[q].append([fields.evaluate(q, {**ev.values, "time": t}, m.quad,
                                                 heat_fields.at_quadrature(m.quad, Tj, m.cond))
                                 for t, Tj in zip(times, T)])
            if np.iscomplexobj(T):
                self.log(f"  temperature amplitude: up to {np.abs(T[-1]).max():.6g} K")
            else:
                self.log(f"  T: {T[-1].min():.6g} .. {T[-1].max():.6g} K")
            if p.heat.problem.type == "stationary":
                self.progress(i + 1, len(outer_values))
        info = {"project": p.name, "elapsed_s": round(time.perf_counter() - t_start, 2),
                "n_dofs": int(self._model.mesh.n_nodes) if self._model else 0}
        from ...units import dimension
        outer_dim = dimension(p.variables[outer_name], base) if outer_name else None
        info["outer_dim"] = list(outer_dim) if outer_dim else None
        self.log(f"Done in {info['elapsed_s']} s")
        return HeatResults(kind=p.heat.problem.type, times=times, T=temps, outer_name=outer_name,
                           outer_values=outer_values, quantities={q: np.array(v) for q, v in qvals.items()},
                           info=info, meshes=meshes, kels=kels)


def compute_quantity(results: HeatResults, project, quantity: str) -> np.ndarray:
    """Evaluate a quantity on stored temperature fields."""
    q = quantity.strip()
    base = project.evaluator()
    out = np.full((len(results.outer_values), len(results.times)), np.nan)
    quads = {}
    for i, T in enumerate(results.T):
        ev = base.with_overrides({results.outer_name: results.outer_values[i]}) if results.outer_name else base
        mesh = results.meshes[i]
        quad = quads.setdefault(id(mesh), fields.Quadrature.of(mesh))
        for j, t in enumerate(results.times):
            out[i, j] = fields.evaluate(q, {**ev.values, "time": t}, quad,
                                        heat_fields.at_quadrature(quad, T[j], results.kels[i]))
    results.quantities[q] = out
    return out
