"""Shared pieces of the validation cases: the report of a case, convergence orders, plot style, model files.

A case module defines TITLE, ABOUT (what is checked, markdown) and run(quick) -> Report. Every case builds
ordinary SolidOcean models (saved to validation/models/ so they can be opened in the GUI) and solves them with
the same solvers as the command line.

The figures follow one style: Arial, ticks inside, no grid, legends without a frame, the SolidOcean results in
the shore colours of the app icon (solid circles) against grey reference solutions, axes labelled
"Quantity, symbol (units)" with the symbol in math italics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = HERE / "models"
DOCS = ROOT.parent / "solidocean-docs" / "docs"      # the documentation repository, cloned next to this one
FIGURES = DOCS / "images" / "validation"             # set by run.py --docs
FIGURE_LINK = "../images/validation"                 # figures relative to docs/validation/report.md

DPI = 300
FONT = "Arial"
SHORE = ["#0b2a5b", "#13608f", "#1b9aaa", "#5cc8c0", "#bfe8dc"]   # the app icon's palette, open sea -> beach
REF_COLOR = "0.35"                   # reference solutions are grey, so the shore colours are SolidOcean's own
FEM_COLOR = SHORE[1]
COLORS = SHORE

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",    # Arial has no subscript digits: symbols are set as math text
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.grid": False,
    "axes.axisbelow": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.markersize": 4,
    "savefig.dpi": DPI,
})


def quiet(*_):
    pass


@dataclass
class Check:
    label: str
    value: float
    limit: float
    kind: str = "max"            # max: value <= limit; min: value >= limit
    fmt: str = ".2e"

    @property
    def ok(self) -> bool:
        return self.value <= self.limit if self.kind == "max" else self.value >= self.limit

    def text(self) -> str:
        op = "≤" if self.kind == "max" else "≥"
        return f"{self.label}: {format(self.value, self.fmt)} ({op} {format(self.limit, self.fmt)})"


@dataclass
class Report:
    name: str
    title: str
    about: str
    figures: list = field(default_factory=list)      # (file name, caption)
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def figure(self, fig, file: str, caption: str):
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / file, dpi=DPI)
        plt.close(fig)
        self.figures.append((file, caption))

    def check(self, label, value, limit, kind="max", fmt=".2e"):
        self.checks.append(Check(label, float(value), float(limit), kind, fmt))

    def markdown(self) -> str:
        out = [f"## {self.title}", "", self.about.strip(), ""]
        for file, caption in self.figures:
            out += [f"![{caption}]({FIGURE_LINK}/{file})", f"*{caption}*", ""]
        if self.notes:
            out += [*self.notes, ""]
        out.append("**Checks:** " + "; ".join(("✅ " if c.ok else "❌ ") + c.text() for c in self.checks)
                   + f". Run time {self.seconds:.0f} s.")
        out.append("")
        return "\n".join(out)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def mean_edge(mesh) -> float:
    """Mean edge length of the (corner) tetrahedra: the element size h of convergence studies, measured
    because gmsh's sizing (from curvature and model size) makes coarse meshes finer than max_size asks."""
    c = mesh.points[mesh.cells[:, :4]]
    return float(np.mean([np.linalg.norm(c[:, i] - c[:, j], axis=1).mean() for i in range(4) for j in range(i)]))


def order(h, err) -> float:
    """Observed convergence order: least-squares slope of log(err) against log(h)."""
    h, err = np.asarray(h, float), np.asarray(err, float)
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])


def convergence_axes(ax, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def slope_guide(ax, h, err, p: int, label=True):
    """Dashed line of slope p through the finest point."""
    h = np.asarray(h, float)
    x = np.array([h.min(), h.max()])
    y = err[np.argmin(h)] * (x / h.min()) ** p
    ax.plot(x, y, "--", color="0.6", lw=1, label=f"Order {p}" if label else None)


def thin(index, n: int = 22):
    """At most n of the given (sorted) sample indices, evenly spaced: solid markers would otherwise run into
    each other and hide the reference curve they are meant to be compared with."""
    index = np.asarray(index)
    return index if len(index) <= n else index[np.linspace(0, len(index) - 1, n).round().astype(int)]


def figure(w=6.0, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    return fig, ax


def figures(n=2, w=9.0, h=3.8):
    """A row of n panels sharing the style."""
    fig, axes = plt.subplots(1, n, figsize=(w, h), layout="constrained")
    return fig, axes


def save_model(project, name: str) -> Path:
    MODELS.mkdir(exist_ok=True)
    return project.save(MODELS / f"{name}.toml")
