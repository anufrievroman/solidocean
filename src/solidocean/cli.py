"""Command-line entry point.

    solidocean model.toml            run the problem headless, write results ([output])
    solidocean                       open the GUI
    solidocean --gui model.toml      open a model in the GUI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _parse_set(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects name=expression, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _run(project, out_dir, log, progress):
    """Run the project's problem headless and write its results; None on error."""
    from . import physics
    spec = physics.of(project)
    plots = [pl for pl in project.plots if pl.kind != "field" and not spec.plot_unavailable(project, pl)]
    try:     # 3D field views are GUI only
        results = spec.solver(project, log=log, progress=progress).run([q for pl in plots for q in pl.expressions()])
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None
    return results.write_outputs(project, out_dir, plots)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="solidocean", description="SolidOcean: FEM solver for elastic eigenfrequencies / "
                                 "phonon dispersion, stationary solid mechanics and heat transfer")
    ap.add_argument("input", nargs="?", help="model file (.toml)")
    ap.add_argument("--gui", action="store_true", help="open the GUI (default when no input is given)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=EXPR",
                    help="override a variable, e.g. --set 'a=600[nm]' (repeatable)")
    ap.add_argument("--check", action="store_true", help="validate the model file and exit")
    ap.add_argument("--mesh-only", action="store_true", help="build geometry and mesh, write mesh.msh, and exit")
    ap.add_argument("-o", "--output", metavar="DIR",
                    help="write the results into DIR (exactly that folder, ignoring the [output] template)")
    from .outputs import CONTENTS
    ap.add_argument("--contents", metavar="LIST",
                    help="what to write, overriding [output] contents: " + ",".join(CONTENTS))
    ap.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    from . import physics
    from .templates import by_physics
    ap.add_argument("--template", metavar="NAME",
                    help="write a new model from a template to INPUT and exit. " + "; ".join(
                        f"{spec.title}: {', '.join(by_physics(kind))}" for kind, spec in physics.REGISTRY.items()))
    ap.add_argument("--install-app", nargs="?", const="~/Applications", metavar="DIR",
                    help="macOS: create 'SolidOcean.app' in DIR (default ~/Applications) and exit")
    ap.add_argument("--version", action="version", version=f"solidocean {__version__}")
    args = ap.parse_args(argv)

    if args.install_app:
        from .macapp import install_app
        try:
            print(f"Created {install_app(args.install_app)}")
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.template:
        from .templates import create
        if not args.input:
            print("error: --template needs an output file, e.g. solidocean --template hexagonal model.toml",
                  file=sys.stderr)
            return 2
        try:
            path = create(args.template).save(args.input)
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Created {path}")
        return 0

    if args.gui or not args.input:
        from .gui.app import run as run_gui
        return run_gui(args.input)

    from .project import Project
    
    log = (lambda msg: None) if args.quiet else print
    try:
        project = Project.load(args.input)
    except Exception as exc:
        print(f"error: cannot read {args.input}: {exc}", file=sys.stderr)
        return 2
    project.variables.update(_parse_set(args.set))
    if args.contents is not None:
        project.output.contents = [c.strip() for c in args.contents.split(",") if c.strip()]
    errors = project.validate_geometry() if args.mesh_only else project.validate()
    if errors:
        print("error: invalid model:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2
    if args.check:
        log("OK")
        return 0

    out_dir = Path(args.output) if args.output else project.run_dir()
    if args.mesh_only:
        from . import meshing
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh = meshing.generate(project, project.evaluator(), save_msh=out_dir / "mesh.msh")
        log(f"Mesh: {mesh.info['n_elements']} elements, {mesh.n_nodes} nodes -> {out_dir / 'mesh.msh'}")
        return 0

    tty = sys.stdout.isatty()

    def progress(i, n):
        if args.quiet:
            return
        if tty:
            print(f"\r[{i}/{n}]", end="\n" if i == n else "", flush=True)
        elif i == n or i % max(n // 10, 1) == 0:  # logs and pipes: about ten lines
            print(f"[{i}/{n}]", flush=True)

    log(f"SolidOcean {__version__}: {project.name} -> {out_dir}")
    written = _run(project, out_dir, log, progress)
    if written is None:
        return 1
    for path in written:
        log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
