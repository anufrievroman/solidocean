"""[output]: the run folder template, what a run writes, and where it ends up."""
import datetime
import json

import numpy as np
import pytest

from solidocean import outputs, templates
from solidocean.cli import main
from solidocean.physics.solid.modal import ModalResults
from solidocean.project import Output, Project

NOW = datetime.datetime(2026, 10, 13, 14, 42, 7)


@pytest.fixture
def MODEL(tmp_path):
    """A saved model with variables a = 400 nm, t = 200 nm, r = 0.43 a."""
    p = templates.create("square")
    p.name = "Square slab"
    p.save(tmp_path / "m.toml")
    return p


def results() -> ModalResults:
    return ModalResults(np.array([[[1e9, 2e9]]]), np.zeros((1, 1, 3)), np.zeros(1), "k")


def test_defaults_and_roundtrip(tmp_path, MODEL):
    assert MODEL.output.contents == ["tables", "data", "model"] and MODEL.output.on_exists == "increment"
    assert "output" not in MODEL.to_dict()                     # defaults are left out of the file
    MODEL.output = Output(dir="runs", folder="{name} a={a}", contents=["tables"], on_exists="timestamp",
                          autosave=True)
    q = Project.load(MODEL.save(tmp_path / "m.toml"))
    assert q.to_dict()["output"] == {"dir": "runs", "folder": "{name} a={a}", "contents": ["tables"],
                                     "on_exists": "timestamp", "autosave": True}
    assert q.validate() == [] and q.results_root() == tmp_path / "runs"


def test_old_plot_key_is_accepted(MODEL):
    """0.1 models wrote `plot = false` (no PNGs); runs write no plots at all now."""
    assert Output.from_dict({"dir": "out", "plot": False}) == Output(dir="out")


def test_template_expansion(MODEL):
    MODEL.length_unit = "nm"
    assert outputs.expand(MODEL, "{name} a={a} t={t}") == "Square slab a=400nm t=200nm"
    assert outputs.expand(MODEL, "r={r}") == "r=172nm"          # derived: the value in the preferred unit
    assert outputs.expand(MODEL, "{physics}") == "solid"
    assert outputs.expand(MODEL, "{date}_{time}", now=NOW) == "2026-10-13_14-42-07"
    assert outputs.expand(MODEL, "a/{a}:x", now=NOW) == "a_400nm_x"   # file-system characters are replaced
    assert outputs.expand(MODEL, "run {n}", n=7) == "run 007"


def test_sweep_placeholder(MODEL):
    from solidocean.project import VariableSweep
    MODEL.problem.sweep = VariableSweep("a", "linspace(600[nm], 900[nm], 4)")
    assert outputs.expand(MODEL, "{sweep}") == "a_0.6-0.9um"
    MODEL.output.folder = "{a}"                                # swept: no single value in the run
    assert any("swept variable" in e for e in MODEL.validate())
    MODEL.output.folder = "{sweep}"
    assert MODEL.validate() == []


def test_bad_settings_are_reported(MODEL):
    MODEL.output.folder = "{nope}"
    assert any("unknown placeholder" in e for e in MODEL.validate())
    MODEL.output.folder = ""
    MODEL.output.contents = ["tables", "pictures"]
    MODEL.output.on_exists = "clobber"
    MODEL.output.dir = "/absolute"
    problems = MODEL.validate()
    assert any("contents" in e and "pictures" in e for e in problems)
    assert any("on_exists" in e for e in problems)
    assert any("relative to the model file" in e for e in problems)


def test_run_folders_never_overwrite(tmp_path, MODEL):
    MODEL.output.folder = "{name}"
    first = MODEL.run_dir()
    assert first == tmp_path / "m" / "Square slab"
    first.mkdir(parents=True)
    assert MODEL.run_dir() == tmp_path / "m" / "Square slab (2)"
    (tmp_path / "m" / "Square slab (2)").mkdir()
    assert MODEL.run_dir() == tmp_path / "m" / "Square slab (3)"
    MODEL.output.on_exists = "overwrite"
    assert MODEL.run_dir() == first
    MODEL.output.on_exists = "timestamp"
    assert MODEL.run_dir(now=NOW) == tmp_path / "m" / "Square slab 2026-10-13_144207"
    MODEL.output.folder = "run {n}"                            # explicit numbering: the lowest free number
    assert MODEL.run_dir() == tmp_path / "m" / "run 001"
    (tmp_path / "m" / "run 001").mkdir()
    assert MODEL.run_dir() == tmp_path / "m" / "run 002"
    MODEL.output.folder = ""                                   # no template: everything in the results root
    assert MODEL.run_dir() == tmp_path / "m"


def test_contents_decide_which_files_are_written(tmp_path, MODEL):
    r, out = results(), tmp_path / "run"
    MODEL.output.contents = ["tables"]
    assert {p.name for p in r.write_outputs(MODEL, out, [])} == {"dispersion.csv"}
    MODEL.output.contents = ["data"]
    assert {p.name for p in r.write_outputs(MODEL, out, [])} == {"results.npz"}
    MODEL.output.contents = ["model"]
    assert {p.name for p in r.write_outputs(MODEL, out, [])} == {"model.toml", "run.json"}
    MODEL.output.contents = []
    assert r.write_outputs(MODEL, tmp_path / "empty", []) == [] and (tmp_path / "empty").is_dir()


def test_the_run_record_holds_what_the_model_file_cannot_say(tmp_path, MODEL):
    """model.toml has the expressions; run.json has their values, the version, the time and the run."""
    MODEL.variables["a"] = "600[nm]"
    out = tmp_path / "run"
    MODEL.output.contents = ["model"]
    r = results()
    r.info = {"elapsed_s": 1.5}
    r.write_outputs(MODEL, out, [])
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert set(record) == {"solidocean", "written", "values", "run"}
    assert record["values"]["a"] == pytest.approx(6e-7) and record["run"] == {"elapsed_s": 1.5}
    assert Project.load(out / "model.toml").variables["a"] == "600[nm]"   # runnable as it stands
    assert MODEL.path == tmp_path / "m.toml"                              # the model's own file is untouched


def test_cli_writes_one_folder_per_run(tmp_path):
    p = templates.create("square")
    p.name = "Slab"
    p.mesh.max_size = "a/3"
    p.solid.problem.kpath.n = 2
    p.solid.problem.num_eigs = 4
    p.output.folder = "{name} a={a}"
    p.output.contents = ["tables"]
    path = p.save(tmp_path / "m.toml")
    assert main([str(path), "-q"]) == 0
    assert main([str(path), "-q", "--set", "a=600[nm]"]) == 0
    assert main([str(path), "-q"]) == 0                        # again: the first run is kept
    assert {d.name for d in (tmp_path / "m").iterdir()} == {
        "Slab a=400nm", "Slab a=600nm", "Slab a=400nm (2)"}
    assert main([str(path), "-q", "-o", str(tmp_path / "here"), "--contents", "tables,model"]) == 0
    assert {f.name for f in (tmp_path / "here").iterdir()} == {   # a table per plot, too
        "dispersion.csv", "dispersion_colored.csv", "model.toml", "run.json"}
    assert main([str(path), "-q", "--contents", "tables,nope"]) == 2      # validated like any setting


def test_quantity_columns_are_named_after_the_plots(tmp_path):
    """quantities.csv: a column per plot under the plot's name, whatever its expression (commas, quotes),
    and quantities.json: name -> expression in the order of the columns."""
    import csv
    from solidocean.plots import ResultPlot
    p = templates.create("heat_bar")
    p.output.contents = ["tables"]
    p.plots = [ResultPlot("Probe", 'integral(T*exp(-x/1[um]), "all")', kind="quantity"),
               ResultPlot("Mean", "mean(T)", kind="quantity"),
               ResultPlot("Peak", "max(T)", kind="quantity")]
    path = p.save(tmp_path / "m.toml")
    assert main([str(path), "-q", "-o", str(tmp_path / "run")]) == 0
    with open(tmp_path / "run" / "quantities.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0])[-3:] == ["Probe", "Mean", "Peak"]
    assert float(rows[0]["Peak"]) >= float(rows[0]["Mean"])
    index = json.loads((tmp_path / "run" / "quantities.json").read_text(encoding="utf-8"))
    assert index == {"Probe": 'integral(T*exp(-x/1[um]), "all")', "Mean": "mean(T)", "Peak": "max(T)"}
