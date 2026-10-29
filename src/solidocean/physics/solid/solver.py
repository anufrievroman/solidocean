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
from ..base import Cancelled, LogFn, ProgressFn  # noqa: F401  (Cancelled: raised by the problems)
from .model import NOT_CONSTRAINTS


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
    """Runs problems for a project, caching the discretised model between steps."""

    def __init__(self, project, log: LogFn = print,
                 progress: ProgressFn | None = None, cancel: threading.Event | None = None):
        self.project = project
        self.log = log
        self.progress = progress or (lambda i, n: None)
        self.cancel = cancel or threading.Event()
        # the caller's event, or None when nobody can set one: meshing only pays for a separate
        # process when there is a cancel button behind it
        self.mesh_cancel = cancel
        self._model_key = None
        self._model: Model | None = None

    # ---- model -----------------------------------------------------------
    def _geometry_key(self, ev: Evaluator):
        p = self.project
        feats = evaluate_features(p, ev)
        vals = [(f.feature.type, f.feature.op, f.feature.material, f.feature.on,
                 tuple((k, np.round(np.atleast_1d(v), 15).tobytes()) for k, v in sorted(f.values.items())))
                for f in feats]
        mesh = p.mesh.key(ev)
        from ... import lattice as lat
        periods = tuple((n, tuple(np.round(v, 18))) for n, v in lat.translations(p.lattice, ev))
        return repr((vals, mesh, periods))

    def _constraint_key(self, ev: Evaluator):
        """Changes when the constrained DOFs change (not their prescribed values)."""
        ph = self.project.solid
        held = [(c.type, selection.key(c, ev), c.free_components()) for c in ph.active_conditions
                if c.type not in NOT_CONSTRAINTS]
        return repr((tuple(ph.symmetry), tuple(ph.antisymmetry), held))

    def _material_key(self, ev: Evaluator):
        st = self.project.solid.problem
        loss = (ev(st.loss), tuple(mat.loss_factor(m.props) for m in self.project.materials)) if st.harmonic else ()
        return repr((tuple((m.name, mat.evaluate(m.props, ev)[0].tobytes(), mat.evaluate(m.props, ev)[1])
                           for m in self.project.materials), loss))

    def _loss_weights(self, mesh, ev: Evaluator) -> np.ndarray:
        """Loss factor of every element (at its centroid): the problem's plus its material's eta."""
        from ... import fields
        p = self.project
        eta = np.full(len(mesh.cells), ev(p.solid.problem.loss))
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
            mesh = meshing.generate(p, ev, cancel=self.mesh_cancel)
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
            if p.solid.problem.harmonic:
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
        ph = self.project.solid
        idx = selection.select(mesh, boundary, c, ev, exclude=ph.symmetry + ph.antisymmetry)  # cut planes
        if len(idx) == 0:
            self.log(f"Warning: condition '{c.name or c.type}' selects no boundary faces")
        return idx

    def _prescribed(self, mesh, boundary, ev, values: bool = True):
        """(N, 3) mask of the components held by fixed constraints and prescribed displacements, and (with
        values=True) their prescribed values (m). Where selections overlap, the later condition wins."""
        held = np.zeros((mesh.n_nodes, 3), dtype=bool)
        vals = np.zeros((mesh.n_nodes, 3))
        for c in self.project.solid.active_conditions:
            if c.type in NOT_CONSTRAINTS:
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
        for c in self.project.solid.active_conditions:
            if c.type == "edge_load":
                f += self._edge_load(m, c, ev)
                continue
            if c.type == "point_load":
                f += self._point_load(m, c, ev)
                continue
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

    def _edge_load(self, m: Model, c, ev: Evaluator) -> np.ndarray:
        """(N, 3) nodal forces of an edge load: the boundary-triangle edges lying on the segment start -> end
        carry the force per length (N/m), or the total force (N) spread over their length."""
        mesh, b = m.mesh, m.boundary
        f = np.zeros((mesh.n_nodes, 3))
        edges = surface.boundary_edges(b.tris, mesh.order)
        on = surface.edges_on_segment(mesh.points, edges, ev.vector(c.start), ev.vector(c.end),
                                      selection.tolerance(mesh))
        if not on.any():
            self.log(f"Warning: edge load '{c.name or c.type}' lies on no boundary edges of the mesh")
            return f
        w = surface.edge_terms(mesh.points, edges[on], mesh.order, mesh.n_nodes)
        vec = ev.vector(c.load)
        if c.load_type == "total":
            vec = vec / w.sum()
        return w[:, None] * vec[None, :]

    def _point_load(self, m: Model, c, ev: Evaluator) -> np.ndarray:
        """(N, 3) nodal forces of a point load: the force acts on the mesh node nearest to the point."""
        mesh = m.mesh
        f = np.zeros((mesh.n_nodes, 3))
        x = ev.vector(c.point)
        dist = np.linalg.norm(mesh.points - x, axis=1)
        node = int(np.argmin(dist))
        lo, hi = mesh.bounds()
        if dist[node] > 1e-3 * np.linalg.norm(hi - lo):
            self.log(f"Warning: point load '{c.name or c.type}' is {dist[node]:.3g} m from the nearest mesh node; "
                     f"it acts there (place the point on a vertex of the geometry)")
        f[node] = ev.vector(c.load)
        return f

    def impedance_matrix(self, m: Model, ev: Evaluator):
        """Real matrix D of the absorbing boundaries: the harmonic system gets + i omega D, with
        D = sum over faces of int Z N_a N_b dA (times the identity on the components) and Z = rho (c_p + c_s) / 2
        of the material behind each face. The speeds follow from the Christoffel matrix n.C.n of the element's
        stiffness: c_p is the fastest wave along the face normal n, c_s the mean of the other two."""
        import scipy.sparse as sp
        mesh, b = m.mesh, m.boundary
        n = mesh.n_nodes
        D = sp.csr_matrix((n, n))
        rho = np.array([mat.evaluate(self.project.material(name).props, ev)[1] for name in mesh.material_names])
        for c in self.project.solid.active_conditions:
            if c.type != "impedance":
                continue
            idx = self._selected(mesh, b, c, ev)
            if len(idx) == 0:
                continue
            tris, owner = b.tris[idx], b.owner[idx]
            nrm = surface.face_normals(mesh.points, tris)
            C = m.cel[owner]                                    # (F, 6, 6) Voigt
            # Christoffel matrix G_ik = n_j C_ijkl n_l, via the Voigt index of each pair (i, j)
            voigt = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
            G = np.einsum("fj,fl,fijkl->fik", nrm, nrm,
                          C[:, voigt[:, :, None, None], voigt[None, None, :, :]])
            r = rho[mesh.cell_domain[owner]]
            speeds = np.sqrt(np.maximum(np.linalg.eigvalsh(G), 0.0) / r[:, None])    # ascending
            z = r * (speeds[:, 2] + speeds[:, :2].mean(axis=1)) / 2
            coeff = z[:, None] * surface.surface_at_points(mesh.points, tris, mesh.order, np.ones(n))
            D = D + surface.surface_terms(mesh.points, tris, mesh.order, coeff=coeff, n_nodes=n)[0]
        return sp.kron(D, sp.identity(3), format="csr")

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
            ur = eigen._factorize(Kr, self.project.problem.solver)(rhs)
        except Exception as exc:
            raise ValueError(f"stationary solve failed ({exc}): is the model held against rigid-body motion?") \
                from None
        resid = np.linalg.norm(Kr @ ur - rhs)
        scale = max(np.linalg.norm(rhs), 1e-300)
        if not np.all(np.isfinite(ur)) or resid > 1e-6 * scale:
            raise ValueError("stationary solve failed: the model can still move as a rigid body (translate or "
                             "rotate). Hold it with fixed constraints, prescribed displacements or symmetry planes")
        return (T @ ur + g).reshape(n, 3)

    def harmonic_drive(self, m: Model, ev: Evaluator):
        """(omega, alpha, beta, f, g) of a harmonic solve: the angular frequency, the Rayleigh coefficients, the
        nodal forces (3N,) and the prescribed displacements of the nodes the constraints hold (3N,)."""
        cons, n = m.constraints, m.mesh.n_nodes
        held, vals = self._prescribed(m.mesh, m.boundary, ev)
        root_vals = np.zeros((n, 3))
        rows, comps = np.nonzero(held)
        root_vals[cons.root[rows], comps] = vals[rows, comps]
        g = np.where(cons.dof_index[cons.root] < 0, root_vals[cons.root], 0.0).ravel()
        f = self.load_vector(m, ev).ravel()
        st = self.project.solid.problem
        return 2 * np.pi * ev(st.frequency), ev(st.rayleigh_alpha), ev(st.rayleigh_beta), f, g

    def harmonic_matrix(self, m: Model, D, omega: float, alpha: float, beta: float):
        """System matrix K (1 + i omega beta) + i K_loss - (omega^2 - i omega alpha) M + i omega D (full space)."""
        A = m.K * (1 + 1j * omega * beta) - (omega ** 2 - 1j * omega * alpha) * m.M
        if m.K_loss is not None:
            A = A + 1j * m.K_loss
        if D.nnz:
            A = A + 1j * omega * D
        return A

    def solve_harmonic(self, ev: Evaluator) -> np.ndarray:
        """Complex displacement amplitude (N, 3) in m of the harmonic response at the problem's frequency:
        (K (1 + i eta) + i omega (alpha M + beta K + D) - omega^2 M) u = f, with the loss factors eta, the
        Rayleigh damping alpha M + beta K and the impedance of absorbing boundaries D, the loads and prescribed
        displacements being the amplitudes of the drive (u(t) = Re(u exp(i omega t)))."""
        m = self.model(ev)
        cons, n = m.constraints, m.mesh.n_nodes
        omega, alpha, beta, f, g = self.harmonic_drive(m, ev)
        A = self.harmonic_matrix(m, self.impedance_matrix(m, ev), omega, alpha, beta)
        if cons.n_free == 0:
            return g.reshape(n, 3).astype(complex)
        # Bloch vector of a periodic model, as the eigenfrequency path does: T carries the phase
        # e^{i k.R} and the reduction is Hermitian, T^H A T. At k = 0 the phases are 1, T is real
        # and this is exactly the old T^T A T. A non-zero k makes the solve the response of the
        # infinite periodic structure to a drive modulated by e^{i k.x} — one wavevector at a time.
        T = cons.T(self.k_of(ev)).tocsr()
        TH = T.conj().T.tocsr()
        Ar = (TH @ A @ T).tocsc()
        rhs = TH @ (f - A @ g)
        ur = eigen._factorize(Ar, self.project.problem.solver)(rhs)
        if not np.all(np.isfinite(ur)):
            raise ValueError("harmonic solve failed: the drive frequency is (numerically) an eigenfrequency of a "
                             "lossless model; add a loss factor (solid.problem.loss or a material's eta) or Rayleigh damping")
        return (T @ ur + g).reshape(n, 3)

    def sweep_key(self, ev: Evaluator):
        """What fixes the matrices of a harmonic solve: equal for two values of a sweep when only the frequency,
        the Rayleigh coefficients and the drive (loads, prescribed displacements) differ between them."""
        ph = self.project.solid
        absorbing = [selection.key(c, ev) for c in ph.active_conditions if c.type == "impedance"]
        return (self._geometry_key(ev), self._material_key(ev), self._constraint_key(ev), repr(absorbing))

    # ---- solving ---------------------------------------------------------
    def _sigma(self, ev: Evaluator):
        f = ev(self.project.solid.problem.search_freq)
        return (2 * np.pi * f) ** 2 if f > 0 else None

    def solve_point(self, ev: Evaluator, k: np.ndarray, n: int | None = None, modes: bool = False):
        """Eigenfrequencies (Hz) at one k; with modes=True also (n, N, 3) displacements."""
        m = self.model(ev)
        T = m.constraints.T(k)
        TH = T.conj().T
        Kr, Mr = (TH @ m.K @ T).tocsr(), (TH @ m.M @ T).tocsr()
        st = self.project.solid.problem
        lam, vec = eigen.solve(Kr, Mr, n or st.num_eigs, self._sigma(ev), backend_name=self.project.problem.solver,
                               tol=st.tol, maxiter=st.maxiter)
        freqs = eigen.frequencies(lam)
        if not modes:
            return freqs
        u = (T @ vec).T.reshape(len(lam), -1, 3)
        return freqs, u

    def k_of(self, ev: Evaluator) -> np.ndarray:
        return ev.vector(self.project.solid.problem.k)

    def run(self, quantities=(), store_modes: bool = False):
        """Run the project's problem: eigenfrequency (modal.ModalResults; quantities are evaluated for every mode,
        store_modes keeps mode shapes in memory for later evaluation) or stationary (static.StaticResults)."""
        if self.project.solid.problem.field_problem:
            from . import static
            return static.run(self, quantities)
        from . import modal
        return modal.run(self, quantities, store_modes)
