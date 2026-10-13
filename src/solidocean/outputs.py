"""Where a run's files go and what goes with them.

`[output]` names the results root (`dir`) and, inside it, one folder per run built from a template
(`folder`): "{name} a={a}" becomes "Slab a=600nm". Placeholders are the model's variables and a few
built-ins; a run folder also keeps the model file and a record of the run (`contents` "model"), so the
parameters of a saved run can always be read back.

Works on a Project but never imports it, so project.py can use this module.
"""

from __future__ import annotations

import datetime
import itertools
import json
import re
from pathlib import Path

from . import units

# What a run can write; the values of [output] contents.
CONTENTS = ("tables", "data", "model")
CONTENT_HELP = {
    "tables": "Tables (CSV): dispersion.csv / quantities.csv and a table per 1D plot.",
    "data": "Full data (.npz): fields and quantities in SI, reloadable with 'Load results…'.",
    "model": "Model file (model.toml) and a record of the run (run.json).",
}
# What to do when the run folder already exists.
ON_EXISTS = ("increment", "timestamp", "overwrite")
ON_EXISTS_HELP = {
    "increment": "Add a number: 'Slab a=600nm (2)'.",
    "timestamp": "Always add the date and time, so every run keeps its own folder.",
    "overwrite": "Write into the same folder again, replacing the files of the earlier run.",
}
# Template placeholders that are not model variables.
BUILTINS = ("name", "physics", "sweep", "date", "time", "datetime", "n")
BUILTIN_HELP = {
    "name": "the project name",
    "physics": "the physics solved",
    "sweep": "the variable sweep, e.g. a_0.6-0.9um",
    "date": "2026-10-13",
    "time": "14-42-07",
    "datetime": "2026-10-13_144207",
    "n": "the lowest free run number, 001, 002, …",
}

_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')
# a plain number with an optional unit: "600", "600[nm]", "-1.5e3[um]"
_LITERAL = re.compile(r"^[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?(?:\s*\[[^\]]*\])?$")


def placeholders(template: str) -> list[str]:
    """The placeholder names used by a folder template, in order."""
    return [m.group(1).strip() for m in _PLACEHOLDER.finditer(template or "")]


def _compact(text: str) -> str:
    """A formatted value as a file-name part: '0.6 µm' -> '0.6um'."""
    return text.replace("µ", "u").replace(" ", "")


def _clean(name: str) -> str:
    """A folder name without the characters a file system objects to (spaces are kept)."""
    name = _ILLEGAL.sub("_", name.replace("[", "").replace("]", ""))
    name = re.sub(r"_{2,}", "_", name).strip().strip(".")
    return name


def variable_text(project, name: str, ev) -> str:
    """A variable as a file-name part: its own expression when that is a plain value ('600[nm]' ->
    '600nm'), otherwise the value in the preferred units ('a/2' -> '0.3um')."""
    expr = str(project.variables[name])
    if _LITERAL.match(expr.strip()):
        return _compact(expr)
    return _compact(units.format_value(ev(expr), units.dimension(expr, ev), project.display_units))


def sweep_text(project, ev) -> str:
    """The variable sweep as a file-name part: 'a_0.6-0.9um' (empty if nothing is swept)."""
    sw = project.study.active_sweep
    if not sw:
        return ""
    try:
        values = ev.array(sw.values)
        dim = units.dimension(str(project.variables.get(sw.variable, "0")), ev)
        lo, hi = (_compact(units.format_value(v, dim, project.display_units)) for v in (values[0], values[-1]))
    except Exception:
        return sw.variable
    unit = re.sub(r"^[-+0-9.eE]+", "", hi)          # "0.9um" -> "um", written once
    if unit and lo.endswith(unit):
        lo = lo[: -len(unit)]
    return f"{sw.variable}_{lo}-{hi}"


def expand(project, template: str | None = None, now: datetime.datetime | None = None, n: int | None = None) -> str:
    """The folder name of one run: the template with its placeholders replaced."""
    template = project.output.folder if template is None else template
    now = now or datetime.datetime.now()
    try:
        ev = project.evaluator()
    except Exception:
        ev = None
    built = {"name": project.name, "physics": project.physics_kind, "date": f"{now:%Y-%m-%d}",
             "time": f"{now:%H-%M-%S}", "datetime": f"{now:%Y-%m-%d_%H%M%S}",
             "sweep": sweep_text(project, ev) if ev else "", "n": f"{n or 1:03d}"}

    def value(match) -> str:
        key = match.group(1).strip()
        if key in built:
            return str(built[key])
        if ev is not None and key in project.variables:
            try:
                return variable_text(project, key, ev)
            except Exception:
                return key
        return match.group(0)      # unknown: left as it is, validate() reports it

    return _clean(_PLACEHOLDER.sub(value, template))


def problems(project) -> list[str]:
    """Human-readable problems of the [output] settings (empty if OK)."""
    out, errors = project.output, []
    for what in out.contents:
        if what not in CONTENTS:
            errors.append(f"output.contents: unknown '{what}' (one of {', '.join(CONTENTS)})")
    if out.on_exists not in ON_EXISTS:
        errors.append(f"output.on_exists must be one of {', '.join(ON_EXISTS)}")
    if Path(out.dir).is_absolute():
        errors.append("output.dir must be relative to the model file")
    swept = project.study.active_sweep.variable if project.study.active_sweep else None
    for key in placeholders(out.folder):
        if key == swept:
            errors.append(f"output.folder: '{{{key}}}' is the swept variable, which has no single value in a "
                          "run — use {sweep}")
        elif key not in BUILTINS and key not in project.variables:
            errors.append(f"output.folder: unknown placeholder '{{{key}}}' (a variable, or one of "
                          f"{', '.join('{' + b + '}' for b in BUILTINS)})")
    if out.folder.strip() and not expand(project).strip():
        errors.append("output.folder: expands to an empty folder name")
    return errors


def resolve(project, root: str | Path | None = None, now: datetime.datetime | None = None) -> Path:
    """The folder this run writes into: the results root plus the expanded template, made unique
    according to [output] on_exists. Creates nothing."""
    root = Path(root) if root is not None else project.results_root()
    template = project.output.folder.strip()
    if not template:
        return root
    now = now or datetime.datetime.now()
    if "n" in placeholders(template):                 # explicit numbering: the lowest free number
        for i in itertools.count(1):
            path = root / expand(project, now=now, n=i)
            if not path.exists():
                return path
    name = expand(project, now=now)
    if project.output.on_exists == "timestamp":
        name = _clean(f"{name} {now:%Y-%m-%d_%H%M%S}")
    path = root / name
    if not path.exists() or project.output.on_exists == "overwrite":
        return path
    for i in itertools.count(2):
        path = root / _clean(f"{name} ({i})")
        if not path.exists():
            return path


def write_record(project, directory: Path, results=None) -> list[Path]:
    """The provenance of a run ([output] contents 'model'): the model file it was run from, and run.json
    with what the model file cannot say — when it ran, with which version, what its expressions evaluated
    to, and what the solver reported. Everything else is in model.toml beside it."""
    from . import __version__
    directory = Path(directory)
    model = directory / "model.toml"
    model.write_text(project.dumps())
    record = {"solidocean": __version__, "written": datetime.datetime.now().isoformat(timespec="seconds")}
    try:    # the variables as numbers (SI); their expressions are in model.toml
        ev = project.evaluator()
        record["values"] = {k: float(ev(v)) for k, v in project.variables.items()}
    except Exception:
        pass
    if results is not None and getattr(results, "info", None):
        record["run"] = results.info
    path = directory / "run.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    return [model, path]
