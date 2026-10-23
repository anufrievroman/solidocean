"""Hydrodynamic heat transport (Guyer-Krumhansl, steady state): boundary conditions and solution.

The finite-element system is in fem/hydro.py. The heat-transfer conditions mean here:

- temperature: a heat bath, the natural condition of the weak form (T prescribed, q free);
- flux: an inward heat flux through the face, i.e. q = -flux n at its nodes (with the face's spatial profile),
  and no slip along it;
- faces without a condition, and the edges where a bath meets them: insulated walls with no slip, q = 0;
- symmetry planes (heat.symmetry): the normal component of q is zero, the tangential ones are free (no
  stress). A slab closed by symmetry planes is therefore exactly a 2D model.

Convection and radiation, thermal contacts and temperature-dependent properties are not supported.
"""

from __future__ import annotations

import numpy as np

from ... import selection
from ...fem import heat, hydro, surface

HYDRO_CONDITIONS = ("temperature", "flux")
TRI = "tag"


def conductivities(m) -> np.ndarray:
    """The (isotropic) conductivity of every domain."""
    out = []
    for name, k in zip(m.mesh.material_names, m.props.constant()[0]):
        if not np.allclose(k, k[0, 0] * np.eye(3), rtol=1e-9, atol=0):
            raise ValueError(f"material '{name}': hydrodynamic heat transport needs an isotropic conductivity")
        out.append(float(k[0, 0]))
    return np.array(out)


def _select(solver, m, c, ev) -> np.ndarray:
    """Indices into m.tris of a condition's triangles."""
    return selection.select(m.mesh, selection.Boundary(m.tris, m.owner), c, ev,
                            exclude=solver.project.heat.symmetry)


def face_node_normals(points, tris, owner_cells):
    """(nodes, outward unit normal at each (len(nodes), 3)) of the nodes of boundary triangles: the mean of the
    normals of the triangles around the node."""
    nrm = surface.face_normals(points, tris)
    inward = points[owner_cells[:, :4]].mean(axis=1) - points[tris[:, :3]].mean(axis=1)
    nrm = nrm * np.where(np.einsum("fi,fi->f", nrm, inward) > 0, -1.0, 1.0)[:, None]
    nodes, inv = np.unique(tris, return_inverse=True)
    total = np.zeros((len(nodes), 3))
    np.add.at(total, inv.reshape(tris.shape), nrm[:, None, :].repeat(tris.shape[1], axis=1))
    return nodes, total / np.linalg.norm(total, axis=1, keepdims=True)


def solve(solver, ev, m):
    """(T (N,) nodal temperature, q (N, 3) nodal heat flux) of the steady hydrodynamic problem."""
    p, mesh = solver.project, m.mesh
    n, pts = mesh.n_nodes, mesh.points
    if mesh.order != 2:
        raise ValueError("hydrodynamic heat transport needs quadratic elements (Mesh → order 2)")
    ell = ev(p.heat.problem.viscous_length)
    k = conductivities(m)
    A, G = hydro.assemble(pts, mesh.cells, mesh.cell_domain, k, ell ** 2 / k)
    vert, pos = hydro.vertices(mesh.cells)
    f_q = np.zeros(3 * n)

    # what every boundary triangle is: 0 wall, 1 bath, 2 flux, 3 symmetry plane (the latter wins, see select)
    tag = np.zeros(len(m.tris), dtype=np.int8)
    tol = selection.tolerance(mesh)
    sym_axis = np.full(len(m.tris), -1)
    for face in p.heat.symmetry:
        on = np.all(np.abs(pts[m.tris[:, :3], "xyz".index(face[0])] -
                           (pts[:, "xyz".index(face[0])].min() if face.endswith("min")
                            else pts[:, "xyz".index(face[0])].max())) < tol, axis=1)
        tag[on], sym_axis[on] = 3, "xyz".index(face[0])
    baths, fluxes = [], []
    for c in p.heat.active_conditions:
        idx = _select(solver, m, c, ev)
        if c.type == "temperature":
            baths.append((idx, solver._profile(ev, c.value, transient=False)))
            tag[idx] = np.where(tag[idx] == 2, 2, 1)
        elif c.type == "flux":
            fluxes.append((idx, c))
            tag[idx] = 2

    fixed_q: dict[int, float] = {}
    # symmetry planes: normal component zero
    for axis in range(3):
        for node in np.unique(m.tris[sym_axis == axis]):
            fixed_q[axis * n + int(node)] = 0.0
    # no slip on walls, and where a wall meets a bath
    for node in np.unique(m.tris[tag == 0]):
        for comp in range(3):
            fixed_q[comp * n + int(node)] = 0.0
    # baths: the prescribed temperature acts on q through the surface term -int T_D w.n
    for idx, T_D in baths:
        nt = surface.normal_terms(pts, m.tris[idx], mesh.cells[m.owner[idx]], 2, n)
        f_q -= T_D * nt.T.ravel()
    # fluxes: q = -flux n at the nodes of the faces (the inward heat flux is against the outward normal)
    inflow = 0.0
    for idx, c in fluxes:
        tris = m.tris[idx]
        if len(tris) == 0:
            continue
        nodes, nhat = face_node_normals(pts, tris, mesh.cells[m.owner[idx]])
        value = solver._profile(ev, c.value, transient=False)
        factor = c.profile.factor(ev, pts[nodes]) if c.profile.shape else 1.0
        qn = value * factor
        for comp in range(3):
            for node, v in zip(nodes, -qn * nhat[:, comp] * np.ones(len(nodes))):
                fixed_q[comp * n + int(node)] = float(v)
        load = 1.0
        if c.profile.shape:
            xq = np.stack([heat.surface_at_points(pts, tris, 2, pts[:, i]) for i in range(3)], axis=-1)
            load = c.profile.factor(ev, xq)
        inflow += value * float(heat.surface_terms(pts, tris, 2, load=load, n_nodes=n)[1].sum())

    f_T = np.zeros(len(vert))
    for q in p.heat.active_sources:
        unit = [1.0 if name == q.material else 0.0 for name in mesh.material_names]
        if not any(unit):
            continue
        value = solver._profile(ev, q.value, transient=False)
        f_T += value * hydro.source_vector(pts, mesh.cells, mesh.cell_domain, unit,
                                           profile=(lambda X, q=q: q.profile.factor(ev, X)) if q.profile.shape else None)
    # div q = Q: the right-hand side of the second block is -F (G^T = -B)
    f_T = -f_T

    fixed = np.array(sorted(fixed_q), dtype=np.int64)
    values = np.array([fixed_q[i] for i in fixed])
    if not baths:       # no heat bath: T is defined up to a constant (set at one vertex), and heat in = heat out
        source = -f_T.sum()
        if abs(inflow + source) > 1e-6 * max(abs(inflow), abs(source), 1e-300):
            raise ValueError(f"no steady state: no heat bath (fixed temperature) and a net heat input of "
                             f"{inflow + source:.4g} W. Add a heat bath, or balance the flux and the sources.")
        fixed = np.append(fixed, 3 * n)
        values = np.append(values, float(solver._T0(ev, m)[vert[0]]))
    x = hydro.solve(A, G, f_q, f_T, fixed, values, p.problem.solver)
    q = x[:3 * n].reshape(3, n).T
    T = hydro.embed_temperature(mesh.cells, n, x[3 * n:])
    return T, q
