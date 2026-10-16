"""Progress of a headless run: the percentage, the elapsed and the remaining time on the way through a
sweep, and that every problem type actually reports its steps."""
import io
import sys

from solidocean import templates
from solidocean.cli import _Reporter, _duration
from solidocean.project import VariableSweep


def test_duration_reads_as_a_rough_time():
    assert _duration(0) == "0s" and _duration(42.4) == "42s"
    assert _duration(67) == "1m 07s" and _duration(3599) == "59m 59s"
    assert _duration(3600) == "1h 00m" and _duration(5000) == "1h 23m"
    assert _duration(-1) == "0s"


def pipe_report(steps, quiet=False):
    """The lines a script or an agent reads: `progress(i, steps)` for every i, with stdout a pipe."""
    out = io.StringIO()          # a StringIO is not a tty, which is the branch agents and logs get
    stdout, sys.stdout = sys.stdout, out
    try:
        rep = _Reporter(quiet)
        rep.start -= 60.0        # pretend a minute has passed, so the estimate is reported
        for i in range(1, steps + 1):
            rep.progress(i, steps)
    finally:
        sys.stdout = stdout
    return out.getvalue().splitlines()


def test_a_piped_run_reports_about_ten_lines_with_the_percentage_and_the_time_left():
    lines = pipe_report(200)
    assert len(lines) == 11                              # the first step, nine more deciles, and 100%
    assert lines[0].strip().startswith("0% (1/200)")     # the first step reports: the earliest estimate
    assert "1m 00s elapsed" in lines[0] and "~3h 19m left" in lines[0]
    assert lines[1].strip().startswith("10% (20/200)")
    assert lines[-1].strip().startswith("100% (200/200)")
    assert "left" not in lines[-1]                       # nothing left to wait for


def test_every_step_is_a_line_when_there_are_only_a_few():
    lines = pipe_report(3)
    assert [l.split("%")[0].strip() for l in lines] == ["33", "67", "100"]


def test_the_estimate_comes_from_the_pace_so_far():
    lines = pipe_report(4)                                # all four "take" the same frozen minute
    assert "~3m 00s left" in lines[0]                     # 1 of 4 done in 1m -> 1m a step -> 3m left
    assert "~20s left" in lines[2]                        # 3 of 4 done in 1m -> 20s a step -> 20s left


def test_quiet_reports_nothing():
    assert pipe_report(10, quiet=True) == []


def test_a_terminal_rewrites_one_line_and_a_log_line_clears_it(monkeypatch):
    """On a terminal the bar sits on one row; a log line must take that row over, not land on the bar."""
    out = io.StringIO()
    monkeypatch.setattr(out, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys, "stdout", out)
    rep = _Reporter(False)
    rep.progress(1, 4)
    assert rep.live and out.getvalue().startswith("\r") and "\n" not in out.getvalue()
    rep.log("T_hot = 310")
    assert not rep.live
    assert "\r\x1b[KT_hot = 310\n" in out.getvalue()     # the bar's row is cleared first
    rep.progress(4, 4)
    assert not rep.live and out.getvalue().endswith("\n")  # the last one stays on screen


def steps_of(project):
    """The (i, n) pairs a run reports."""
    from solidocean import physics
    seen = []
    physics.of(project).solver(project, log=lambda m: None,
                               progress=lambda i, n: seen.append((i, n))).run()
    return seen


def test_a_sweep_reports_its_steps_whatever_the_problem_type():
    """One step per sweep value, ending at 100%. A frequency-domain heat run used to report nothing,
    because only stationary ones called back."""
    for name, values, mesh in (("heat_bar", "[305, 310, 315] * 1[K]", "40[um]"),      # heat, stationary
                               ("heat_3omega", "[10, 20, 30] * 1[kHz]", "40[um]"),   # heat, frequency domain
                               ("tensile", "[0, 30, 60]", "w"),                      # solid, stationary
                               ("square", "[400, 500, 600] * 1[nm]", "a/3")):        # solid, eigenfrequency
        p = templates.create(name)
        p.mesh.max_size = mesh                     # coarse: this is about the reporting, not the physics
        if name == "square":                       # no sweep of its own: sweep the lattice constant
            p.problem.sweep = VariableSweep(variable="a")
            p.solid.problem.kpath.n, p.solid.problem.num_eigs = 2, 3
        p.problem.active_sweep.values = values
        steps = steps_of(p)
        assert steps, f"{name}: no progress reported at all"
        assert steps[-1] == (steps[-1][1], steps[-1][1]), f"{name}: ends at {steps[-1]}, not at 100%"
        assert [i for i, _ in steps] == sorted(i for i, _ in steps), f"{name}: steps go backwards"
        assert len({n for _, n in steps}) == 1, f"{name}: the scale changes mid-run: {steps}"
        if name != "square":                       # eigenfrequency counts k-points within each value
            assert steps == [(1, 3), (2, 3), (3, 3)], name


def test_a_transient_reports_its_time_steps():
    """A transient has many more steps than sweep values: every saved time step of every value counts."""
    p = templates.create("heat_bar")
    p.mesh.max_size = "40[um]"
    p.problem.active_sweep.values = "[305, 310] * 1[K]"
    p.heat.problem.type = "transient"
    p.heat.problem.duration, p.heat.problem.step = "4[us]", "1[us]"
    seen = steps_of(p)
    assert seen[0][1] == seen[-1][1] and seen[-1][1] > 2        # one scale over the whole run
    assert seen[-1][0] == seen[-1][1]                           # and it ends at 100%
    assert [i for i, _ in seen] == sorted(i for i, _ in seen)   # never going backwards
