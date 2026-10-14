"""Shared pieces of the validation cases: the report of a case, convergence orders, plot style, model files.

A case module defines TITLE, ABOUT (what is checked, markdown) and run(quick) -> Report. Every case builds
ordinary SolidOcean models (saved to validation/models/ so they can be opened in the GUI) and solves them with
the same solvers as the command line.
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

REF_COLOR, FEM_COLOR = "0.35", "C3"
COLORS = ["C0", "C3", "C2", "C1", "C4"]


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
    tables: list = field(default_factory=list)       # (caption, headers, rows)
    figures: list = field(default_factory=list)      # (file name, caption)
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def table(self, caption, headers, rows):
        self.tables.append((caption, headers, rows))

    def figure(self, fig, file: str, caption: str):
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / file, dpi=130)
        plt.close(fig)
        self.figures.append((file, caption))

    def check(self, label, value, limit, kind="max", fmt=".2e"):
        self.checks.append(Check(label, float(value), float(limit), kind, fmt))

    def markdown(self) -> str:
        out = [f"## {self.title}", "", self.about.strip(), ""]
        for caption, headers, rows in self.tables:
            if caption:
                out += [f"*{caption}*", ""]
            out.append("| " + " | ".join(headers) + " |")
            out.append("|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|")
            out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
            out.append("")
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


def convergence_axes(ax, xlabel="Element size h", ylabel="Relative error"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)


def fraction_ticks(ax, fracs, symbol: str):
    """Ticks of element sizes given as fractions of a length, labelled e.g. 't/12'."""
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(list(fracs))
    ax.set_xticklabels([f"{symbol}/{1 / f:.0f}" for f in fracs])


def plain_ticks(ax, values, axis="x"):
    """Log axis labelled only at the given values, without exponents."""
    from matplotlib.ticker import NullLocator
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_minor_locator(NullLocator())
    (ax.set_xticks if axis == "x" else ax.set_yticks)(list(values))
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([f"{v:g}" for v in values])


def slope_guide(ax, h, err, p: int, label=True):
    """Dashed line of slope p through the finest point."""
    h = np.asarray(h, float)
    x = np.array([h.min(), h.max()])
    y = err[np.argmin(h)] * (x / h.min()) ** p
    ax.plot(x, y, "--", color="0.6", lw=1, label=f"order {p}" if label else None)


def figure(w=6.0, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
    return fig, ax


def save_model(project, name: str) -> Path:
    MODELS.mkdir(exist_ok=True)
    return project.save(MODELS / f"{name}.toml")


def fmt_err(e: float) -> str:
    return f"{e:.2e}"
