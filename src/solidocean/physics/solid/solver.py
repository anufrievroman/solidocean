"""Solid mechanics solver: mesh, stiffness and mass matrices, constraints (symmetry, periodicity, fixed and
prescribed displacements on selections) and loads, cached between the steps of a sweep; one eigenproblem at
a wavevector (solve_point), one stationary solve (solve_static) or one harmonic solve (solve_harmonic)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ... import materials as mat
from ... import meshing, selection
from ...expressions import Evaluator
from ...fem import assembly, bloch, eigen, surface
from ...geometry import evaluate_features
from ..base import Cancelled, LogFn, ProgressFn  # noqa: F401  (Cancelled: raised by the studies)


@dataclass
class Model:
    """Discretised model for one geometry/material state."""
    mesh: meshing.Mesh
    K: object
    M: object
    constraints: bloch.Constraints
    boundary: selection.Boundary | None = None
    cel: np.ndarray | None = None      # (E, 6, 6) stiffness of every element (stresses)
    K_loss: object = None              # sum of eta_e K_e (harmonic response: K + i K_loss), None if lossless


class Solver:
    """Runs studies for a project, caching the discretised model between steps."""

    def __init__(self, project, log: LogFn = print,
                 progress: ProgressFn | None = None, cancel: threading.Event | None = None):
        self.project = project
        self.log = log
        self.progress = progress or (lambda i, n: None)
        self.cancel = cancel or threading.Event()
        self._model_key = None
        self._model: Model | None = None

    # ---- model -----------------------------------------------------------
    def _geometry_key(self, ev: Evaluator):
        p = self.project
        feats = evaluate_features(p, ev)
        vals = [(f.feature.type, f.feature.op, f.feature.material,
                 tuple((k, np.round(np.atleast_1d(v), 15).tobytes()) for k, v in sorted(f.values.items())))
                for f in feats]
        mesh = p.mesh.key(ev)
        from ... import lattice as lat
        periods = tuple((n, tuple(np.round(v, 18))) for n, v in lat.translations(p.lattice, ev))
        return repr((vals, mesh, periods))

    def _constraint_key(self, ev: Evaluator):
        """Changes when the constrained DOFs change (not their prescribed values)."""
        ph = self.project.solid
        held = [(c.type, selection.key(c, ev), c.free_components()) for c in ph.conditions if c.type != "load"]
        return repr((tuple(ph.symmetry), tuple(ph.antisymmetry), held))

    def _material_key(self, ev: Evaluator):
        st = self.project.solid.study
        loss = (ev(st.loss), tuple(mat.loss_factor(m.props) for m in self.project.materials)) if st.harmonic else ()
        return repr((tuple((m.name, mat.evaluate(m.props, ev)[0].tobytes(), mat.evaluate(m.props, ev)[1])
                           for m in self.project.materials), loss))

    def _loss_weights(self, mesh, ev: Evaluator) -> np.ndarray:
        """Loss factor of every element (at its centroid): the study's plus its material's eta."""
        from ... import fields
        p = self.project
        eta = np.full(len(mesh.cells), ev(p.solid.study.loss))
        centroids = mesh.points[mesh.cells[:, :4]].mean(axis=1)
        for d, name in enumerate(mesh.material_names):
            expr = mat.loss_factor(p.material(name).props)
            sel = mesh.cell_domain == d
            if expr and np.any(sel):
                eta[sel] += fields.evaluate_nodal(expr, centroids[sel], ev.values)
        return eta

    def model(self, ev: Evaluator) -> Model:
        """Mesh, matrices and constraints. Each is rebuilt only when what it depends on changes: a sweep of
        material properties re-assembles on the same mesh (re-meshing would add mesh noise to the sweep), and
        a change of the constraints only rebuilds the reduction."""
        mesh_key = self._geometry_key(ev)
        key = (mesh_key, self._material_key(ev), self._constraint_key(ev))
        old, old_key = self._model, self._model_key
        if key == old_key and old is not None:
            return old
        p = self.project
        t0 = time.perf_counter()
        if old is not None and old_key[0] == mesh_key:
            mesh, boundary = old.mesh, old.boundary
        else:
            self.log("Meshing…")
            mesh = meshing.generate(p, ev)
            boundary = selection.Boundary.of(mesh)
            self.log(f"Mesh: {mesh.info['n_elements']} elements, {mesh.n_nodes} nodes "
                     f"({mesh.n_dofs} DOFs), order {mesh.order} [{time.perf_counter() - t0:.1f} s]")
        t1 = time.perf_counter()
        if old is not None and old_key[:2] == key[:2]:
            K, M, cel, K_loss = old.K, old.M, old.cel, old.K_loss
        else:
            C_list, rho_list = [], []
            for name in mesh.material_names:
                C, rho = mat.evaluate(p.material(name).props, ev)
                C_list.append(C)
                rho_list.append(rho)
            K, M = assembly.assemble(mesh.points, mesh.cells, mesh.cell_domain, C_list, rho_list, mesh.order)
            cel = np.array(C_list)[mesh.cell_domain]
            K_loss = None
            if p.solid.study.harmonic:
                eta = self._loss_weights(mesh, ev)
                if np.any(eta != 0):
                    K_loss = assembly.assemble_weighted(mesh.points, mesh.cells, mesh.cell_domain, C_list, eta,
                                                        mesh.order)
        from ... import lattice as lat
        periodic = (lat.translations(p.lattice, ev) if p.lattice.vectors     # any lattice
                    else {ax: ev(v) for ax, v in p.lattice.periodic.items()})  # box cell (strict checks)
        held, _ = self._prescribed(mesh, boundary, ev, values=False)
        cons = bloch.build_constraints(mesh.points, periodic, (), p.solid.symmetry, p.solid.antisymmetry,
                                       constrained_dofs=held)
        self.log(f"Assembly: {cons.n_free} free DOFs [{time.perf_counter() - t1:.1f} s]")
        self._model, self._model_key = Model(mesh, K, M, cons, boundary, cel, K_loss), key
        return self._model

    def _selected(self, mesh, boundary, c, ev) -> np.ndarray:
        idx = selection.select(mesh, boundary, c, ev)
        if len(idx) == 0:
            self.log(f"Warning: condition '{c.name or c.type}' selects no boundary faces")
        return idx

    def _prescribed(self, mesh, boundary, ev, values: bool = True):
        """(N, 3) mask of the components held by fixed constraints and prescribed displacements, and (with
        values=True) their prescribed values (m). Where selections overlap, the later condition wins."""
        held = np.zeros((mesh.n_nodes, 3), dtype=bool)
        vals = np.zeros((mesh.n_nodes, 3))
        for c in self.project.solid.conditions:
            if c.type == "load":
                continue
            nodes = np.unique(boundary.tris[self._selected(mesh, boundary, c, ev)])
            for comp, free in enumerate(c.free_components()):
                if free:
                    continue
                held[nodes, comp] = True
                if values:
                    vals[nodes, comp] = ev(c.displacement[comp]) if c.type == "displacement" else 0.0
        return held, vals

    def load_vector(self, m: Model, ev: Evaluator) -> np.ndarray:
        """(N, 3) nodal forces of the boundary loads (N)."""
        mesh, b = m.mesh, m.boundary
        f = np.zeros((mesh.n_nodes, 3))
        for c in self.project.solid.conditions:
            if c.type != "load":
                continue
            idx = self._selected(mesh, b, c, ev)
            tris = b.tris[idx]
            if c.load_type == "pressure":
                f -= ev(c.pressure) * surface.normal_terms(mesh.points, tris, mesh.cells[b.owner[idx]], mesh.order,
                                                           mesh.n_nodes)
                continue
            _, w = surface.surface_terms(mesh.points, tris, mesh.order, load=1.0, n_nodes=mesh.n_nodes)
            vec = ev.vector(c.load)
            if c.load_type == "total":   # spread uniformly over the selected area
                area = w.sum()
                if area <= 0:
                    continue
                vec = vec / area
            f += w[:, None] * vec[None, :]
        return f

    def solve_static(self, ev: Evaluator) -> np.ndarray:
        """Stationary displacement (N, 3) in m: K u = f with the constraints and prescribed displacements;
        periodic models are solved for a periodic displacement (k = 0)."""
        m = self.model(ev)
        cons, n = m.constraints, m.mesh.n_nodes
        held, vals = self._prescribed(m.mesh, m.boundary, ev)
        root_vals = np.zeros((n, 3))   # a value on any periodic image applies to the shared root
        rows, comps = np.nonzero(held)
        root_vals[cons.root[rows], comps] = vals[rows, comps]
        g = np.where(cons.dof_index[cons.root] < 0, root_vals[cons.root], 0.0).ravel()
        f = self.load_vector(m, ev).ravel()
        T = cons.T().real.tocsr()
        Kr = (T.T @ m.K @ T).tocsc()
        rhs = T.T @ (f - m.K @ g)
        if cons.n_free == 0:
            return g.reshape(n, 3)
        if 3 * len(np.unique(cons.root)) == cons.n_free:   # no component of any node is constrained
            raise ValueError("the model is not held in place: add a fixed constraint, prescribed displacement or "
                             "symmetry plane (a free body under load has no stationary solution)")
        try:
            ur = eigen._factorize(Kr, self.project.study.solver)(rhs)
        except Exception as exc:
            raise ValueError(f"stationary solve failed ({exc}): is the model held against rigid-body motion?") \
                from None
        resid = np.linalg.norm(Kr @ ur - rhs)
        scale = max(np.linalg.norm(rhs), 1e-300)
        if not np.all(np.isfinite(ur)) or resid > 1e-6 * scale:
            raise ValueError("stationary solve failed: the model can still move as a rigid body (translate or "
                             "rotate). Hold it with fixed constraints, prescribed displacements or symmetry planes")
        return (T @ ur + g).reshape(n, 3)

    def solve_harmonic(self, ev: Evaluator) -> np.ndarray:
        """Complex displacement amplitude (N, 3) in m of the harmonic response at the study's frequency:
        (K (1 + i eta) - omega^2 M) u = f, the loads and prescribed displacements being the amplitudes of the
        drive (u(t) = Re(u exp(i omega t)))."""
        m = self.model(ev)
        cons, n = m.constraints, m.mesh.n_nodes
        held, vals = self._prescribed(m.mesh, m.boundary, ev)
        root_vals = np.zeros((n, 3))
        rows, comps = np.nonzero(held)
        root_vals[cons.root[rows], comps] = vals[rows, comps]
        g = np.where(cons.dof_index[cons.root] < 0, root_vals[cons.root], 0.0).ravel()
        f = self.load_vector(m, ev).ravel()
        omega = 2 * np.pi * ev(self.project.solid.study.frequency)
        A = m.K.astype(complex) - omega ** 2 * m.M
        if m.K_loss is not None:
            A = A + 1j * m.K_loss
        if cons.n_free == 0:
            return g.reshape(n, 3).astype(complex)
        T = cons.T().real.tocsr()
        Ar = (T.T @ A @ T).tocsc()
        rhs = T.T @ (f - A @ g)
        ur = eigen._factorize(Ar, self.project.study.solver)(rhs)
        if not np.all(np.isfinite(ur)):
            raise ValueError("harmonic solve failed: the drive frequency is (numerically) an eigenfrequency of a "
                             "lossless model; add a loss factor (solid.study.loss or a material's eta)")
        return (T @ ur + g).reshape(n, 3)

    # ---- solving ---------------------------------------------------------
    def _sigma(self, ev: Evaluator):
        f = ev(self.project.solid.study.search_freq)
        return (2 * np.pi * f) ** 2 if f > 0 else None

    def solve_point(self, ev: Evaluator, k: np.ndarray, n: int | None = None, modes: bool = False):
        """Eigenfrequencies (Hz) at one k; with modes=True also (n, N, 3) displacements."""
        m = self.model(ev)
        T = m.constraints.T(k)
        TH = T.conj().T
        Kr, Mr = (TH @ m.K @ T).tocsr(), (TH @ m.M @ T).tocsr()
        st = self.project.solid.study
        lam, vec = eigen.solve(Kr, Mr, n or st.num_eigs, self._sigma(ev), backend_name=self.project.study.solver,
                               tol=st.tol, maxiter=st.maxiter)
        freqs = eigen.frequencies(lam)
        if not modes:
            return freqs
        u = (T @ vec).T.reshape(len(lam), -1, 3)
        return freqs, u

    def k_of(self, ev: Evaluator) -> np.ndarray:
        return ev.vector(self.project.solid.study.k)

    def run(self, quantities=(), store_modes: bool = False):
        """Run the project's study: eigenfrequency (modal.ModalResults; quantities are evaluated for every mode,
        store_modes keeps mode shapes in memory for later evaluation) or stationary (static.StaticResults)."""
        if self.project.solid.study.field_study:
            from . import static
            return static.run(self, quantities)
        from . import modal
        return modal.run(self, quantities, store_modes)
