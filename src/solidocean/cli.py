"""Command-line entry point.

    solidocean model.toml            run the problem headless, write results ([output])
    solidocean                       open the GUI
    solidocean --gui model.toml      open a model in the GUI
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__

BAR_WIDTH = 24


def _duration(seconds: float) -> str:
    """A rough human-readable duration: 42s, 3m 07s, 1h 12m."""
    seconds = int(round(max(seconds, 0)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


class _Reporter:
    """Where a headless run's messages go: the solver's log lines, and how far it has got — the percentage
    done with the elapsed and the estimated remaining time (from the average pace so far, so it settles
    down only after the first few steps).

    On a terminal the progress is a bar on one line that rewrites itself; into a pipe or a log file — what
    a script or an agent reads — it is about ten plain lines instead, so the progress survives in captured
    output. A step is a sweep value, or a time step of a transient problem; a single solve reports only
    once it is done, so a long one shows where it is through its log lines instead."""

    def __init__(self, quiet: bool):
        self.quiet = quiet
        self.tty = sys.stdout.isatty()
        utf8 = (sys.stdout.encoding or "").lower().replace("-", "").startswith("utf8")
        self.blocks = "█░" if utf8 else "#-"   # a plain bar where the console cannot print block characters
        self.start = time.perf_counter()
        self.live = False       # a bar is sitting on the current terminal row, waiting to be overwritten
        self.last_draw = 0.0
        self.last_decile = -1   # -1, so the first step reports: on a long run that is the first estimate

    def _end_bar(self):
        """Take the cursor off the bar's row so the next line does not land on top of it."""
        if self.live:
            print("\r\x1b[K", end="", flush=True)
            self.live = False

    def log(self, msg: str):
        if self.quiet:
            return
        self._end_bar()
        print(msg, flush=True)

    def error(self, msg: str):
        self._end_bar()
        print(msg, file=sys.stderr, flush=True)

    def progress(self, i: int, n: int):
        if self.quiet or n <= 0:
            return
        done = min(max(i, 0), n)
        elapsed = time.perf_counter() - self.start
        frac = done / n
        eta = ""
        if 0 < done < n and elapsed > 0.5:
            eta = f", ~{_duration(elapsed * (n - done) / done)} left"
        text = f"{frac * 100:3.0f}% ({done}/{n}), {_duration(elapsed)} elapsed{eta}"
        if self.tty:
            now = time.perf_counter()
            if done < n and now - self.last_draw < 0.2:    # at most ~5 redraws a second
                return
            self.last_draw = now
            filled = int(round(frac * BAR_WIDTH))
            bar = self.blocks[0] * filled + self.blocks[1] * (BAR_WIDTH - filled)
            print(f"\r  {bar} {text}\x1b[K", end="\n" if done == n else "", flush=True)
            self.live = done < n
        else:
            decile = int(frac * 10)
            if done == n or decile > self.last_decile:     # the first step, each 10%, and the end
                self.last_decile = decile
                print(f"  {text}", flush=True)


def _parse_set(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects name=expression, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _run(project, out_dir, rep: _Reporter):
    """Run the project's problem headless and write its results; None on error."""
    from . import physics
    spec = physics.of(project)
    plots = [pl for pl in project.plots if pl.kind != "field" and not spec.plot_unavailable(project, pl)]
    solver = spec.solver(project, log=rep.log, progress=rep.progress)
    try:     # 3D field views are GUI only
        results = solver.run([q for pl in plots for q in pl.expressions()])
    except Exception as exc:
        rep.error(f"error: {exc}")
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
    
    rep = _Reporter(args.quiet)
    log = rep.log
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

    log(f"SolidOcean {__version__}: {project.name} -> {out_dir}")
    written = _run(project, out_dir, rep)
    if written is None:
        return 1
    for path in written:
        log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
