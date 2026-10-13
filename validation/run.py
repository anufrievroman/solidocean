"""Run the validation cases and write the report docs/validation.md (figures in docs/validation/).

    python validation/run.py                 # every case
    python validation/run.py lamb_plate      # some cases (the others keep their last results)
    python validation/run.py --quick         # coarser studies, for a quick check

Each case compares SolidOcean with an analytic solution (or another solver) and studies the convergence;
its checks set the exit code. The sections of the report are kept in docs/validation/sections.json, so a
partial run updates only its own cases.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import platform
import sys
import traceback

import common

CASES = ["lamb_plate", "free_sphere", "cantilever", "driven_bar", "lame_cylinder", "semi_infinite", "spreading_disk",
         "contact_stack", "three_omega", "fdtr"]
GROUPS = [("Elastic waves and vibrations", ["lamb_plate", "free_sphere", "cantilever", "driven_bar"]),
          ("Statics", ["lame_cylinder"]),
          ("Heat transfer", ["semi_infinite", "spreading_disk", "contact_stack", "three_omega", "fdtr"])]
SECTIONS = common.FIGURES / "sections.json"

INTRO = """# Validation

SolidOcean is checked against exact solutions of elasticity and heat conduction. Every case builds ordinary
models (written to `validation/models/`, they open in the GUI) and solves them with the same solvers as the
command line; most also refine the mesh or the time step to show that the error falls at the rate theory
predicts for the elements (for quadratic elements: eigenfrequencies $h^4$, displacements and temperatures
$h^3$, stresses and fluxes $h^2$). Element sizes $h$ are measured mean edge lengths of the meshes.

Regenerate with `python validation/run.py` (about three minutes; `--quick` for coarser studies). The fast code
tests are separate: `pytest`.
"""


def load_sections() -> dict:
    try:
        return json.loads(SECTIONS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_report(sections: dict) -> None:
    from solidocean import __version__
    out = [INTRO]
    out += ["| case | result | checks | run time |", "|---|---|---:|---:|"]
    for name in CASES:
        s = sections.get(name)
        if s:
            anchor = s["title"].lower().replace(":", "").replace(",", "").replace(" ", "-")
            out.append(f"| [{s['title']}](#{anchor}) | {'✅ passed' if s['ok'] else '❌ failed'} | "
                       f"{s['n_ok']}/{s['n_checks']} | {s['seconds']:.0f} s |")
    out.append("")
    for group, names in GROUPS:
        present = [n for n in names if n in sections]
        if present:
            out.append(f"# {group}\n")
            out += [sections[n]["markdown"] for n in present]
    stamps = sorted({s["date"] for s in sections.values()})
    out.append(f"---\n*SolidOcean {__version__}; generated {', '.join(stamps)} on {platform.system()} "
               f"{platform.machine()}, Python {platform.python_version()}.*\n")
    (common.ROOT / "docs" / "validation.md").write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*", default=CASES, help=f"cases: {', '.join(CASES)}")
    ap.add_argument("--quick", action="store_true", help="coarser convergence studies")
    args = ap.parse_args()
    unknown = [c for c in args.cases if c not in CASES]
    if unknown:
        ap.error(f"unknown case(s): {', '.join(unknown)}")
    sections = load_sections()
    failed = []
    for name in args.cases:
        print(f"{name} …", flush=True)
        try:
            report = importlib.import_module(name).run(quick=args.quick)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            continue
        for c in report.checks:
            print(f"  {'ok  ' if c.ok else 'FAIL'} {c.text()}")
        print(f"  {report.seconds:.0f} s")
        if not report.ok:
            failed.append(name)
        sections[name] = {"title": report.title, "markdown": report.markdown(), "ok": report.ok,
                          "n_checks": len(report.checks), "n_ok": sum(c.ok for c in report.checks),
                          "seconds": report.seconds, "date": datetime.date.today().isoformat()}
        common.FIGURES.mkdir(parents=True, exist_ok=True)
        SECTIONS.write_text(json.dumps(sections, indent=1, ensure_ascii=False), encoding="utf-8")
        write_report(sections)
    print("failed: " + ", ".join(failed) if failed else "all passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
