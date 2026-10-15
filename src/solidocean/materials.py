"""Elastic materials: stiffness tensors in Voigt notation and a small library.

Voigt order: xx, yy, zz, yz, xz, xy (engineering shear strains).

Elastic constants are given in the crystal frame. An optional ``rotation_deg``
property ([about x, about y, about z], applied in that order) rotates the
crystal relative to the global axes; the stiffness is transformed accordingly.

Thermal properties: conductivity ``k`` (isotropic) or ``kxx``, ``kyy``, ``kzz``
(principal values in the crystal frame, rotated like the stiffness) and the
specific heat capacity ``cp``. Values are for bulk crystals at room temperature. For heat transfer, the
thermal properties and the density may depend on the temperature ``T`` (K), e.g.
``k = "148[W/(m*K)]*(300[K]/T)^1.3"``.
"""

from __future__ import annotations

import re

import numpy as np

from .expressions import Evaluator

# Property names required by each material model.
MODELS: dict[str, list[str]] = {
    "isotropic": ["rho", "E", "nu"],
    "cubic": ["rho", "C11", "C12", "C44"],
    "hexagonal": ["rho", "C11", "C12", "C13", "C33", "C44"],  # c-axis along z
    "anisotropic": ["rho"] + [f"C{i}{j}" for i in range(1, 7) for j in range(i, 7)],
}

LIBRARY: dict[str, dict[str, str]] = {
    "Si": {"model": "cubic", "rho": "2329[kg/m^3]", "C11": "165.7[GPa]", "C12": "63.9[GPa]", "C44": "79.6[GPa]",
           "k": "148[W/(m*K)]", "cp": "712[J/(kg*K)]"},
    # polycrystalline (deposited) aluminium: isotropic
    "Al": {"model": "isotropic", "rho": "2700[kg/m^3]", "E": "70[GPa]", "nu": "0.35",
           "k": "237[W/(m*K)]", "cp": "897[J/(kg*K)]"},
    # gold, polycrystalline; bulk k (evaporated films of ~100 nm conduct less, roughly 150-250 W/(m K))
    "Au": {"model": "isotropic", "rho": "19300[kg/m^3]", "E": "79[GPa]", "nu": "0.42",
           "k": "317[W/(m*K)]", "cp": "129[J/(kg*K)]"},
    # fused silica (amorphous SiO2; thermal oxide and deposited films are close, E ~ 66-75 GPa)
    "SiO2": {"model": "isotropic", "rho": "2203[kg/m^3]", "E": "73[GPa]", "nu": "0.17",
             "k": "1.38[W/(m*K)]", "cp": "740[J/(kg*K)]"},
    # amorphous silicon nitride (low-stress LPCVD films: membranes, cantilevers). Elastic values from
    # Babaei Gavan et al. 2009 (size-dependent Young's modulus of SiN cantilevers), as used for the band
    # structure of perforated SiN membranes; k is that of the film (far below bulk Si3N4 ceramics)
    "SiN": {"model": "isotropic", "rho": "3100[kg/m^3]", "E": "250[GPa]", "nu": "0.23",
            "k": "3[W/(m*K)]", "cp": "700[J/(kg*K)]"},
    # chromium, polycrystalline (adhesion layers); bulk values, sputtered films are often softer
    "Cr": {"model": "isotropic", "rho": "7190[kg/m^3]", "E": "279[GPa]", "nu": "0.21",
           "k": "93.9[W/(m*K)]", "cp": "449[J/(kg*K)]"},
    # palladium, polycrystalline; bulk values
    "Pd": {"model": "isotropic", "rho": "12023[kg/m^3]", "E": "121[GPa]", "nu": "0.39",
           "k": "71.8[W/(m*K)]", "cp": "244[J/(kg*K)]"},
    "GaAs": {"model": "cubic", "rho": "5317[kg/m^3]", "C11": "118.8[GPa]", "C12": "53.8[GPa]", "C44": "59.4[GPa]",
             "k": "55[W/(m*K)]", "cp": "330[J/(kg*K)]"},
    "Diamond": {"model": "cubic", "rho": "3515[kg/m^3]", "C11": "1076[GPa]", "C12": "125[GPa]",
                "C44": "577[GPa]", "k": "2200[W/(m*K)]", "cp": "509[J/(kg*K)]"},
    "Ge": {"model": "cubic", "rho": "5323[kg/m^3]", "C11": "128.5[GPa]", "C12": "48.3[GPa]", "C44": "66.8[GPa]",
           "k": "60[W/(m*K)]", "cp": "320[J/(kg*K)]"},
    # silicon carbide: cubic 3C (thin films on Si) and hexagonal 4H (wafers, c-axis along z)
    "3C-SiC": {"model": "cubic", "rho": "3210[kg/m^3]", "C11": "390[GPa]", "C12": "142[GPa]", "C44": "256[GPa]",
               "k": "320[W/(m*K)]", "cp": "690[J/(kg*K)]"},
    "4H-SiC": {"model": "hexagonal", "rho": "3210[kg/m^3]", "C11": "507[GPa]", "C12": "108[GPa]",
               "C13": "52[GPa]", "C33": "547[GPa]", "C44": "159[GPa]",
               "k": "370[W/(m*K)]", "cp": "690[J/(kg*K)]"},
    # lithium niobate, congruent, trigonal 3m: c-axis along z, x along a crystal a-axis (IEEE convention).
    # Stiffness at constant electric field (Kovacs et al. 1990; piezoelectric stiffening not included)
    "LiNbO3": {"model": "anisotropic", "rho": "4647[kg/m^3]",
               "C11": "198.39[GPa]", "C12": "54.72[GPa]", "C13": "65.13[GPa]", "C14": "7.88[GPa]",
               "C22": "198.39[GPa]", "C23": "65.13[GPa]", "C24": "-7.88[GPa]", "C33": "227.9[GPa]",
               "C44": "59.65[GPa]", "C55": "59.65[GPa]", "C56": "7.88[GPa]", "C66": "71.835[GPa]",
               "kxx": "4.6[W/(m*K)]", "kyy": "4.6[W/(m*K)]", "kzz": "4.2[W/(m*K)]", "cp": "628[J/(kg*K)]"},
}

THERMAL_KEYS = ("k", "kxx", "kyy", "kzz", "cp")
LOSS_KEY = "eta"            # mechanical loss factor (harmonic response: stiffness C (1 + i eta)); may use x, y, z
TEMPERATURE = "T"           # thermal properties may depend on the temperature T (K) ...
REFERENCE_T = 293.15        # ... and are shown (and checked) at this temperature


def depends_on_T(expr) -> bool:
    return re.search(rf"\b{TEMPERATURE}\b", str(expr)) is not None

# display colours of materials (GUI): the automatic ones, in the order materials are added, are also the
# choices offered for picking one. The first two are from the icon's palette (sea foam, sand); the first is
# also the interface's accent colour (gui/theme.py).
DEFAULT_COLORS = ["#9adcd2", "#e9d8a6", "#8fa8c8", "#d9a066", "#c97f9f", "#b0a3d4", "#d4cf7a"]
COLOR_CHOICES = {"Default": DEFAULT_COLORS}


def is_color(text: str) -> bool:
    """A colour as #rrggbb."""
    import re
    return bool(re.fullmatch(r"#[0-9a-fA-F]{6}", text))


def stiffness(model: str, p: dict[str, float]) -> np.ndarray:
    """6x6 Voigt stiffness matrix from evaluated properties (SI)."""
    C = np.zeros((6, 6))
    if model == "isotropic":
        E, nu = p["E"], p["nu"]
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        C[:3, :3] = lam
        C[[0, 1, 2], [0, 1, 2]] = lam + 2 * mu
        C[[3, 4, 5], [3, 4, 5]] = mu
    elif model == "cubic":
        C[:3, :3] = p["C12"]
        C[[0, 1, 2], [0, 1, 2]] = p["C11"]
        C[[3, 4, 5], [3, 4, 5]] = p["C44"]
    elif model == "hexagonal":
        C[0, 0] = C[1, 1] = p["C11"]
        C[0, 1] = C[1, 0] = p["C12"]
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = p["C13"]
        C[2, 2] = p["C33"]
        C[3, 3] = C[4, 4] = p["C44"]
        C[5, 5] = 0.5 * (p["C11"] - p["C12"])
    elif model == "anisotropic":
        for i in range(6):
            for j in range(i, 6):
                C[i, j] = C[j, i] = p.get(f"C{i + 1}{j + 1}", 0.0)
    else:
        raise ValueError(f"unknown material model '{model}'")
    return C


ROTATION_KEY = "rotation_deg"
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def rotation_matrix(angles_deg) -> np.ndarray:
    """R = Rz @ Ry @ Rx: rotate about global x, then y, then z (degrees)."""
    ax, ay, az = np.radians(np.asarray(angles_deg, dtype=float))
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def voigt_to_tensor(C: np.ndarray) -> np.ndarray:
    T = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(_VOIGT_PAIRS):
        for J, (k, l) in enumerate(_VOIGT_PAIRS):
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    T[a, b, c, d] = C[I, J]
    return T


def tensor_to_voigt(T: np.ndarray) -> np.ndarray:
    C = np.empty((6, 6))
    for I, (i, j) in enumerate(_VOIGT_PAIRS):
        for J, (k, l) in enumerate(_VOIGT_PAIRS):
            C[I, J] = T[i, j, k, l]
    return C


def rotate_stiffness(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Stiffness of a crystal rotated by R, expressed in the global frame."""
    T = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, voigt_to_tensor(C), optimize=True)
    out = tensor_to_voigt(T)
    out[np.abs(out) < 1e-12 * np.abs(out).max()] = 0.0  # clean round-off
    return 0.5 * (out + out.T)


def resolve_properties(props: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Merge library defaults (``library = "Si"``) with explicit properties."""
    props = dict(props)
    lib = props.pop("library", None)
    if lib:
        if lib not in LIBRARY:
            raise ValueError(f"unknown library material '{lib}'")
        merged = dict(LIBRARY[lib])
        merged.update(props)
        props = merged
    model = props.pop("model", "isotropic")
    if model not in MODELS:
        raise ValueError(f"unknown material model '{model}'")
    return model, props


def thermal(props: dict[str, str], ev: Evaluator) -> tuple[np.ndarray, float, float]:
    """(conductivity tensor 3x3 in the global frame, rho, cp) for a material property table."""
    _, props = resolve_properties(props)
    missing = [key for key in ("rho", "cp") if key not in props]
    if "k" not in props and not all(key in props for key in ("kxx", "kyy", "kzz")):
        missing.append("k (or kxx, kyy, kzz)")
    if missing:
        raise ValueError(f"thermal properties missing: {', '.join(missing)}")
    if "k" in props:
        K = np.eye(3) * ev(props["k"])
    else:
        K = np.diag([ev(props[key]) for key in ("kxx", "kyy", "kzz")])
        rotation = props.get(ROTATION_KEY)
        if rotation is not None:
            R = rotation_matrix(ev.vector(rotation))
            K = R @ K @ R.T
    return K, ev(props["rho"]), ev(props["cp"])


def loss_factor(props: dict[str, str]) -> str:
    """The material's loss factor expression ("" = lossless)."""
    _, p = resolve_properties(props)
    return str(p.get(LOSS_KEY, "")).strip()


def evaluate(props: dict[str, str], ev: Evaluator) -> tuple[np.ndarray, float]:
    """Return (C 6x6 in the global frame, rho) for a material property table."""
    model, props = resolve_properties(props)
    rotation = props.pop(ROTATION_KEY, None)
    for key in THERMAL_KEYS + (LOSS_KEY,):  # not needed for the stiffness
        props.pop(key, None)
    required = MODELS[model] if model != "anisotropic" else ["rho"]
    missing = [k for k in required if k not in props]
    if missing:
        raise ValueError(f"{model} material missing properties: {', '.join(missing)}")
    values = {k: ev(v) for k, v in props.items()}
    C = stiffness(model, values)
    if rotation is not None:
        angles = ev.vector(rotation)
        if np.any(angles != 0):
            C = rotate_stiffness(C, rotation_matrix(angles))
    return C, values["rho"]
