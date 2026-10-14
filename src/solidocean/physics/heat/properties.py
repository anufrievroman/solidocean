"""Thermal properties of the mesh domains: conductivity tensor k and volumetric heat capacity rho*cp, constant
or functions of the temperature T (material properties may use `T`, in K, e.g. k = "148[W/(m*K)]*(300[K]/T)^1.3").

Temperature-dependent properties are evaluated pointwise (at quadrature points, sample points or nodes); the
conductivity tensor is built in the crystal frame and rotated with the material's `rotation_deg`.
"""

from __future__ import annotations

import numpy as np

from ... import fields
from ... import materials as mat
from ...expressions import Evaluator
from ...materials import REFERENCE_T, TEMPERATURE, depends_on_T

CHECK_T = (REFERENCE_T, 50.0, 300.0, 600.0, 1200.0)   # K: temperatures that identify a property for caching


def nonlinear(project) -> bool:
    """True if the heat model needs iterations: radiation, temperature-dependent material properties or
    thermal contacts."""
    from .interfaces import varies
    if any(c.type == "radiation" for c in project.heat.conditions) or varies(project):
        return True
    used = {f.material for f in project.objects if f.enabled and f.op == "add"}
    for m in project.materials:
        if m.name in used:
            _, p = mat.resolve_properties(m.props)
            if any(depends_on_T(p.get(key, "")) for key in (*mat.THERMAL_KEYS, "rho")):
                return True
    return False


class _Value:
    """One property: a constant, or an expression of T evaluated pointwise."""

    def __init__(self, expr, ev: Evaluator, where: str):
        self.expr, self.ev, self.where = expr, ev, where
        self.varies = depends_on_T(expr)
        self.const = None if self.varies else ev(expr)
        if self.varies:
            self(np.array([REFERENCE_T]))     # reports a broken expression now, not mid-solve

    def __call__(self, T: np.ndarray) -> np.ndarray:
        if not self.varies:
            return np.full(np.shape(T), self.const)
        try:
            value = fields.evaluate_values(str(self.expr), self.ev.values, {TEMPERATURE: np.asarray(T, float)})
        except Exception as exc:
            raise ValueError(f"{self.where}: {exc}") from exc
        return np.broadcast_to(value, np.shape(T)).astype(float)


class Material:
    def __init__(self, name: str, props: dict, ev: Evaluator):
        _, p = mat.resolve_properties(props)
        missing = [key for key in ("rho", "cp") if key not in p]
        if "k" not in p and not all(key in p for key in ("kxx", "kyy", "kzz")):
            missing.append("k (or kxx, kyy, kzz)")
        if missing:
            raise ValueError(f"material '{name}': thermal properties missing: {', '.join(missing)}")
        where = f"material '{name}'"
        keys = ("k", "k", "k") if "k" in p else ("kxx", "kyy", "kzz")
        self.k = [_Value(p[key], ev, f"{where}, {key}") for key in keys]
        self.R = None
        if "k" not in p and p.get(mat.ROTATION_KEY) is not None:
            self.R = mat.rotation_matrix(ev.vector(p[mat.ROTATION_KEY]))
        self.rho = _Value(p["rho"], ev, f"{where}, rho")
        self.cp = _Value(p["cp"], ev, f"{where}, cp")
        self.name = name

    @property
    def varies(self) -> bool:
        return self.k_varies or self.c_varies

    @property
    def k_varies(self) -> bool:
        return any(v.varies for v in self.k)

    @property
    def c_varies(self) -> bool:
        return self.rho.varies or self.cp.varies

    def conductivity(self, T: np.ndarray) -> np.ndarray:
        """(..., 3, 3) for temperatures T (...)."""
        diag = np.stack([k(T) for k in self.k], axis=-1)
        if np.any(diag <= 0):
            bad = np.asarray(T)[np.any(diag <= 0, axis=-1)]
            raise ValueError(f"material '{self.name}': conductivity not positive at T = {bad.flat[0]:.6g} K")
        K = diag[..., :, None] * np.eye(3)
        return K if self.R is None else self.R @ K @ self.R.T

    def capacity(self, T: np.ndarray) -> np.ndarray:
        """rho*cp (...) for temperatures T (...)."""
        rc = self.rho(T) * self.cp(T)
        if np.any(rc <= 0):
            bad = np.asarray(T)[rc <= 0]
            raise ValueError(f"material '{self.name}': rho*cp not positive at T = {bad.flat[0]:.6g} K")
        return rc


class Properties:
    """Thermal properties of every mesh domain (in mesh.material_names order)."""

    def __init__(self, project, ev: Evaluator, names: list[str]):
        self.materials = [Material(n, project.material(n).props, ev) for n in names]

    @property
    def varies(self) -> bool:
        return any(m.varies for m in self.materials)

    @property
    def k_varies(self) -> bool:
        return any(m.k_varies for m in self.materials)

    @property
    def c_varies(self) -> bool:
        return any(m.c_varies for m in self.materials)

    def key(self) -> str:
        """Identifies the properties (for re-assembly): their values at a few temperatures."""
        T = np.array(CHECK_T)
        return repr([(m.name, [np.round(v(T), 12).tobytes() for v in (*m.k, m.rho, m.cp)],
                      None if m.R is None else m.R.round(12).tobytes()) for m in self.materials])

    def _per_domain(self, fn, domain: np.ndarray, T: np.ndarray, shape) -> np.ndarray:
        out = np.zeros(np.shape(domain) + shape)
        for d, m in enumerate(self.materials):
            sel = domain == d
            if np.any(sel):
                out[sel] = fn(m, T[sel])
        return out

    def conductivity(self, domain: np.ndarray, T) -> np.ndarray:
        """Conductivity tensors (..., 3, 3) at points in mesh domains `domain` (...) at temperatures T."""
        T = np.broadcast_to(np.asarray(T, float), np.shape(domain))
        return self._per_domain(lambda m, t: m.conductivity(t), domain, T, (3, 3))

    def capacity(self, domain: np.ndarray, T) -> np.ndarray:
        T = np.broadcast_to(np.asarray(T, float), np.shape(domain))
        return self._per_domain(lambda m, t: m.capacity(t), domain, T, ())

    def constant(self) -> tuple[list[np.ndarray], list[float]]:
        """(conductivity tensor, rho*cp) per domain; temperature-dependent ones at REFERENCE_T."""
        T = np.array(REFERENCE_T)
        return [m.conductivity(T) for m in self.materials], [float(m.capacity(T)) for m in self.materials]


class Conductivity:
    """k of every element of a mesh as a function of temperature, for the heat flux of results:
    tensor(elements, T) -> (..., 3, 3)."""

    def __init__(self, props: Properties, cell_domain: np.ndarray):
        self.props, self.cell_domain = props, cell_domain
        self.table = None if props.k_varies else np.array(props.constant()[0])

    def tensor(self, elements: np.ndarray, T) -> np.ndarray:
        domain = self.cell_domain[elements]
        if self.table is not None:
            return self.table[domain]
        return self.props.conductivity(domain, T)
