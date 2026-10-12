"""Heat transfer templates. Naming inside a model: first word capitalised."""

from __future__ import annotations

from ...project import Feature, MeshRegion, MeshSettings, VariableSweep, Project, Study
from ...plots import PlotStyle, ResultPlot
from ...templates import Template, library_material
from .model import HeatCondition, HeatInterface, HeatPhysics, HeatSource, HeatStudy, Profile, Pulse
from .plots import field_plot, time_plot


def _si(**overrides):
    return library_material("Si", **overrides)


def _lib(name: str, **overrides):
    return library_material(name, **overrides)


# ---- heat transfer -----------------------------------------------------------------
def heat_empty() -> Project:
    return Project(
        name="Untitled", physics_kind="heat", variables={"a": "1[um]", "T0": "293.15[K]"}, materials=[_si()],
        geometry=[Feature("block", "Block", material="Si", params={"center": ["0", "0", "0"], "size": ["a", "a", "a"]})],
        mesh=MeshSettings(max_size="a/5"),
        heat=HeatPhysics(initial="T0", conditions=[
            HeatCondition("temperature", ["zmin"], "T0", name="Heat sink"),
            HeatCondition("flux", ["zmax"], "1e6[W/m^2]", name="Heating")]))


def heat_bar() -> Project:
    return Project(
        name="Simple heated bar", physics_kind="heat",
        description="Steady heat conduction along a Si bar between a hot and a cold end; the sides are insulated",
        variables={"L": "2[um]", "w": "0.5[um]", "t": "220[nm]", "T_hot": "310[K]", "T_cold": "300[K]"},
        materials=[_si()],
        geometry=[Feature("block", "Bar", material="Si", params={"center": ["0", "0", "0"], "size": ["L", "w", "t"]})],
        mesh=MeshSettings(max_size="t/2"),
        heat=HeatPhysics(initial="T_cold", conditions=[
            HeatCondition("temperature", ["xmin"], "T_hot", name="Hot end"),
            HeatCondition("temperature", ["xmax"], "T_cold", name="Cold end")]),
        study=Study(sweep=VariableSweep("T_hot", "range(305, 5, 330) * 1[K]")),
        plots=[field_plot(),
               ResultPlot("Temperature along the bar", "T", kind="line", start=["-L/2", "0", "0"],
                          end=["L/2", "0", "0"], style=PlotStyle(markers=False)),
               ResultPlot("Mean temperature", "mean(T)", kind="quantity", y_label="Mean temperature (K)"),
               ResultPlot("Effective conductivity", "integral(qx) / ((T_hot - T_cold) * w * t)", kind="quantity",
                          y_label="k_eff (W/(m·K))")])


def heat_convection() -> Project:
    return Project(
        name="Heated plate cooled by convection", physics_kind="heat", length_unit="mm",
        description="Si plate (a chip) heated with power P over a D x D area of its top and cooled by convection "
                    "(heat-transfer coefficient h, ambient T_ext) on its whole outer surface. Temperature rise vs. "
                    "h, from natural (~5-25 W/(m^2 K)) to forced air convection, compared with the lumped estimate "
                    "P / (h A), A the plate's surface area. The heater zone is a block inside the plate, so the "
                    "mesh follows its edges and the heated area is exactly D^2",
        variables={"L": "10[mm]", "t": "0.5[mm]", "D": "2[mm]", "P": "100[mW]", "h": "20[W/(m^2*K)]",
                    "T_ext": "293.15[K]", "A": "2*L^2 + 4*L*t"},
        materials=[_si()],
        geometry=[Feature("block", "Plate", material="Si", params={"center": ["0", "0", "0"],
                                                                  "size": ["L", "L", "t"]}),
                  Feature("block", "Heater", material="Si", params={"center": ["0", "0", "t/2 - t/10"],
                                                                   "size": ["D", "D", "t/5"]})],
        mesh=MeshSettings(max_size="L/20"),
        heat=HeatPhysics(initial="T_ext", conditions=[
            HeatCondition("convection", ["all"], name="Convection", h="h", T_ext="T_ext"),
            HeatCondition("flux", ["zmax"], "P / D^2", name="Heating", on="Heater")]),
        study=Study(sweep=VariableSweep("h", "[5, 10, 20, 50, 100, 200] * 1[W/(m^2*K)]")),
        plots=[field_plot(),
               ResultPlot("Temperature rise", "mean(T) - T_ext", kind="quantity", y_label="Mean temperature rise (K)"),
               ResultPlot("Ratio to lumped estimate", "(mean(T) - T_ext) / (P / (h*A))", kind="quantity",
                          y_label="FEM / lumped estimate"),
               ResultPlot("Temperature across the plate", "T - T_ext", kind="line", start=["-L/2", "0", "t/2"],
                          end=["L/2", "0", "t/2"], y_label="Temperature rise (K)", style=PlotStyle(markers=False))])


def heat_pad() -> Project:
    def block(name, material, center, size):
        return Feature("block", name, material=material, params={"center": center, "size": size})

    return Project(
        name="Laser pulse on the ribbon with pad", physics_kind="heat", time_unit="us",
        description="TDTR experiment with Si ribbon held at T0 at both ends and a gold pad on top, heated by a "
                    "laser pulse: a Gaussian beam (FWHM w, centred at x0 on the ribbon axis) of absorbed power P, "
                    "which falls on the pad and, around it, on the ribbon; average pad temperature vs. time",
        variables={"L": "30[um]", "W": "5[um]", "t": "200[nm]", "D": "4[um]", "h": "100[nm]", "T0": "300[K]",
                    "P": "1[mW]", "w": "3[um]", "x0": "0[um]", "q0": "4*log(2)*P/(pi*w^2)",
                    "t_pulse": "200[ns]", "k_si": "148[W/(m*K)]", "S": "15[um]", "t_sio2": "1[um]"},
        materials=[_si(k="k_si"), _lib("Au"), _lib("SiO2")],
        geometry=[block("Ribbon", "Si", ["0", "0", "0"], ["L", "W", "t"]),
                  # heat sinks: wide Si pads at both ends of the ribbon, on a SiO2 layer
                  block("Sink1", "Si", ["L/2+S/2", "0", "0"], ["S", "S*2", "t"]),
                  block("Sink2", "Si", ["-L/2-S/2", "0", "0"], ["S", "S*2", "t"]),
                  block("SiO2", "SiO2", ["L/2+S/2", "0", "-t_sio2/2-t/2"], ["S", "S*2", "t_sio2"]),
                  block("SiO1", "SiO2", ["-L/2-S/2", "0", "-t_sio2/2-t/2"], ["S", "S*2", "t_sio2"]),
                  Feature("cylinder", "Pad", material="Au",
                          params={"center": ["0", "0", "t/2"], "axis": ["0", "0", "1"], "radius": "D/2",
                                  "height": "h"})],
        # fine where the heat flows (ribbon, pad), coarse in the wide heat sinks
        mesh=MeshSettings(max_size="S/6", regions=[MeshRegion(on="Ribbon", max_size="W/10"),
                                                   MeshRegion(on="Pad", max_size="W/10")]),
        heat=HeatPhysics(initial="T0", conditions=[
            HeatCondition("temperature", ["zmin", "ymin", "xmin", "ymax", "xmax"], "T0", name="Ribbon ends"),
            HeatCondition("flux", ["zmax"], "q0", name="Laser on pad", on="Pad",
                          pulse=Pulse("rect", duration="t_pulse"), profile=Profile("gauss", ["x0", "0"], "w")),
            HeatCondition("flux", ["zmax"], "q0", name="Laser on ribbon", on="Ribbon",
                          pulse=Pulse("rect", duration="t_pulse"), profile=Profile("gauss", ["x0", "0"], "w"))],
            study=HeatStudy("transient", t_start="0[ns]", t_end="10[us]", steps=310)),
        study=Study(sweep=VariableSweep("k_si", "[30, 60, 100, 148] * 1[W/(m*K)]")),
        plots=[field_plot(),
               time_plot("Pad temperature", 'mean(T, "Pad")', "Average pad temperature (K)"),
               time_plot("Peak temperature", "max(T)", "Maximum temperature (K)"),
               ResultPlot("Temperature along the ribbon", "T", kind="line", start=["-L/2", "0", "0"],
                          end=["L/2", "0", "0"], at_time=50, style=PlotStyle(markers=False))])


def heat_radiation() -> Project:
    return Project(
        name="Radiative cooling", physics_kind="heat", length_unit="mm",
        description="A hot Si sphere (radius r, initially T_hot) in vacuum cools only by thermal radiation to "
                    "surroundings at T_amb: q = eps sigma (T_amb^4 - T^4) on its whole surface. Mean temperature "
                    "vs. time for several emissivities eps; the temperature spread inside stays small because "
                    "conduction is much faster than radiation (the sphere cools almost uniformly, time constant "
                    "~ rho cp r / (12 eps sigma T^3), a few seconds at 1000 K)",
        variables={"r": "0.5[mm]", "T_hot": "1000[K]", "T_amb": "293.15[K]", "eps": "0.6"},
        materials=[_si()],
        geometry=[Feature("sphere", "Sphere", material="Si", params={"center": ["0", "0", "0"], "radius": "r"})],
        mesh=MeshSettings(max_size="r/3"),
        heat=HeatPhysics(initial="T_hot", conditions=[
            HeatCondition("radiation", ["all"], name="Radiation", emissivity="eps", T_ext="T_amb")],
            study=HeatStudy("transient", t_end="60[s]", steps=240)),
        study=Study(sweep=VariableSweep("eps", "[0.2, 0.5, 0.9]")),
        plots=[field_plot(),
               time_plot("Mean temperature", "mean(T)", "Mean temperature (K)"),
               time_plot("Temperature spread inside", "max(T) - min(T)", "max(T) − min(T) (K)")])


def heat_diode() -> Project:
    return Project(
        name="Thermal diode", physics_kind="heat",
        description="Thermal rectification by temperature-dependent conductivities: two blocks in contact, "
                    "crystalline Si (k falls with T, length L_si) and amorphous SiO2 (k rises with T, length "
                    "L_ox = L_si/3). One end is held at "
                    "T0 + |dT|, the other at T0: dT > 0 heats the Si end, dT < 0 the SiO2 end. With the SiO2 end hot "
                    "both materials sit where they conduct better, so the same temperature difference drives a "
                    "little more heat that way (about 9 % at |dT| = 400 K; the SiO2 dominates the resistance, which "
                    "limits the effect). Compare the points at +dT and -dT; a shorter SiO2 block rectifies more",
        variables={"T0": "300[K]", "dT": "300[K]", "L_si": "3[um]", "L_ox": "L_si/3", "w": "1[um]",
                    "T_left": "T0 + if(dT > 0, dT, 0)", "T_right": "T0 + if(dT < 0, -dT, 0)"},
        materials=[_si(k="148[W/(m*K)]*(T0/T)^1.3"), _lib("SiO2", k="1.38[W/(m*K)]*(T/T0)^0.45")],
        geometry=[Feature("block", "Si", material="Si", params={"center": ["-L_si/2", "0", "0"],
                                                               "size": ["L_si", "w", "w"]}),
                  Feature("block", "SiO2", material="SiO2", params={"center": ["L_ox/2", "0", "0"],
                                                                   "size": ["L_ox", "w", "w"]})],
        mesh=MeshSettings(max_size="w/4"),
        heat=HeatPhysics(initial="T0", conditions=[
            HeatCondition("temperature", ["xmin"], "T_left", name="Si end"),
            HeatCondition("temperature", ["xmax"], "T_right", name="SiO2 end")]),
        study=Study(sweep=VariableSweep(
            "dT", "[-400, -300, -200, -100, -50, 50, 100, 200, 300, 400] * 1[K]")),
        plots=[field_plot(),
               ResultPlot("Heat flow", "abs(integral(qx)) / (L_si + L_ox) / 1e-6[W]", kind="quantity",
                          y_label="Heat flow (µW)", x_label="dT (K): > 0 Si end hot, < 0 SiO2 end hot"),
               ResultPlot("Thermal conductance", "abs(integral(qx)) / (L_si + L_ox) / abs(dT) / 1e-6[W]",
                          kind="quantity", y_label="Heat flow / |dT| (µW/K)",
                          x_label="dT (K): > 0 Si end hot, < 0 SiO2 end hot"),
               ResultPlot("Temperature along the blocks", "T", kind="line", start=["-L_si", "0", "0"],
                          end=["L_ox", "0", "0"], style=PlotStyle(markers=False))])


def heat_contact() -> Project:
    return Project(
        name="Thermal boundary resistance", physics_kind="heat",
        description="A gold heater pad (power P dissipated in it) on a Si substrate held at T0 at its bottom. The "
                    "Au | Si interface has a thermal boundary (Kapitza) resistance R, so the temperature jumps across "
                    "it by about P R / A (A the pad area). Pad temperature rise vs. R: the substrate's spreading "
                    "resistance at small R, the interface at large R (log-log). Switch the interface to a thin layer "
                    "to model e.g. a Cr adhesion layer of a few nm instead",
        variables={"a": "1[um]", "h": "200[nm]", "S": "4[um]", "H": "2[um]", "T0": "300[K]", "P": "100[uW]",
                    "R": "1e-8[m^2*K/W]"},
        materials=[_si(), _lib("Au"), _lib("Cr")],
        geometry=[Feature("block", "Substrate", material="Si", params={"center": ["0", "0", "-H/2"],
                                                                       "size": ["S", "S", "H"]}),
                  Feature("block", "Pad", material="Au", params={"center": ["0", "0", "h/2"], "size": ["a", "a", "h"]})],
        mesh=MeshSettings(max_size="S/6", regions=[MeshRegion(on="Pad", max_size="h/2")]),   # fine in the thin pad
        heat=HeatPhysics(initial="T0", conditions=[
            HeatCondition("temperature", ["zmin"], "T0", name="Heat sink")],
            sources=[HeatSource("Au", "P / (a^2 * h)", name="Heater")],
            interfaces=[HeatInterface("resistance", ["Pad", "Substrate"], "R", name="Au | Si")]),
        study=Study(sweep=VariableSweep("R", "[0.1, 0.3, 1, 3, 10, 30, 100] * 1e-9[m^2*K/W]")),
        plots=[field_plot(),
               ResultPlot("Pad temperature rise", 'mean(T, "Pad") - T0', kind="quantity",
                          y_label="Pad temperature rise (K)", x_label="Boundary resistance R (m²K/W)",
                          style=PlotStyle(x_log=True, y_log=True)),
               ResultPlot("Interface share", 'P * R / a^2 / (mean(T, "Pad") - T0)', kind="quantity",
                          y_label="Interface jump P R / A ÷ pad temperature rise",
                          x_label="Boundary resistance R (m²K/W)", style=PlotStyle(x_log=True)),
               ResultPlot("Temperature through the pad", "T", kind="line", start=["0", "0", "-H"],
                          end=["0", "0", "h"], style=PlotStyle(markers=False))])


def heat_3omega() -> Project:
    def heater(q):
        return ResultPlot(q[0], q[1], kind="quantity", y_label=q[2], style=PlotStyle(x_log=True))
    return Project(
        name="3ω method: heater line on glass", physics_kind="heat", frequency_unit="kHz",
        description="A gold heater line (width 2b) on a glass substrate, driven by an AC current: its Joule heating "
                    "oscillates at twice the current's frequency with amplitude P per length, and the line's "
                    "temperature oscillation (read as the 3ω voltage) is computed in the frequency domain, swept over "
                    "frequency f. The in-phase part falls linearly with ln f, with slope -P/(2 pi k): the 3ω slope "
                    "method for the substrate's conductivity k; the out-of-phase part stays near -P/(4k). A slice of "
                    "the line (insulated cut faces: an infinite line), half of it by symmetry; far faces at T0",
        variables={"b": "1[um]", "h": "100[nm]", "D": "20[um]", "Lx": "0.5[um]", "P": "1[W/m]",
                    "T0": "300[K]", "f": "10[kHz]"},
        materials=[_lib("SiO2"), _lib("Au")],
        geometry=[Feature("block", "Substrate", material="SiO2", params={"center": ["0", "D/2", "-D/2"],
                                                                         "size": ["Lx", "D", "D"]}),
                  Feature("block", "Heater", material="Au", params={"center": ["0", "b/2", "h/2"],
                                                                    "size": ["Lx", "b", "h"]})],
        mesh=MeshSettings(max_size="b/4"),
        heat=HeatPhysics(initial="T0", conditions=[
            HeatCondition("temperature", ["ymax", "zmin"], "T0", name="Far field")],
            sources=[HeatSource("Au", "P / (2*b*h)", name="Joule heating (2ω)")],
            study=HeatStudy(type="frequency", frequency="f")),
        study=Study(sweep=VariableSweep("f", "logspace(4, 6, 9) * 1[Hz]")),
        plots=[field_plot("Temperature amplitude", "Tamp"),
               heater(("In-phase temperature", 'mean(Tre, "Heater")', "In-phase heater temperature (K)")),
               heater(("Out-of-phase temperature", 'mean(Tim, "Heater")', "Out-of-phase heater temperature (K)")),
               heater(("Heater phase", 'atan2(mean(Tim, "Heater"), mean(Tre, "Heater")) * 180/pi',
                       "Phase of the heater temperature (°)"))])


def heat_fdtr() -> Project:
    probe = "exp(-4*log(2)*(x^2 + y^2)/w^2)"
    weighted = f'integral(Tim*{probe}, "Film"), integral(Tre*{probe}, "Film")'
    return Project(
        name="FDTR: gold transducer on silicon", physics_kind="heat", frequency_unit="MHz",
        description="Frequency-domain thermoreflectance: a pump laser modulated at f heats a gold transducer film on "
                    "silicon through a Gaussian spot (FWHM w, absorbed power amplitude P); the probe, a Gaussian of "
                    "the same size, reads the film's temperature. The phase of the probed temperature against f is "
                    "what FDTR fits for the substrate's conductivity and the Au | Si boundary conductance G. A quarter "
                    "of the spot (symmetry planes x = 0 and y = 0); far faces at T0",
        variables={"w": "2[um]", "d": "100[nm]", "R": "10[um]", "P": "1[mW]", "G": "150[MW/(m^2*K)]",
                    "T0": "300[K]", "f": "1[MHz]", "q0": "4*log(2)*P/(pi*w^2)"},
        materials=[_si(), _lib("Au")],
        geometry=[Feature("block", "Substrate", material="Si", params={"center": ["R/2", "R/2", "-R/2"],
                                                                       "size": ["R", "R", "R"]}),
                  Feature("block", "Film", material="Au", params={"center": ["R/2", "R/2", "d/2"],
                                                                  "size": ["R", "R", "d"]})],
        mesh=MeshSettings(max_size="w/5"),
        heat=HeatPhysics(initial="T0", conditions=[
            HeatCondition("flux", ["zmax"], "q0", name="Pump", profile=Profile("gauss", ["0", "0"], "w")),
            HeatCondition("temperature", ["xmax", "ymax", "zmin"], "T0", name="Far field")],
            interfaces=[HeatInterface("resistance", ["Film", "Substrate"], "1/G", name="Au | Si")],
            study=HeatStudy(type="frequency", frequency="f")),
        study=Study(sweep=VariableSweep("f", "logspace(6, 7.3, 9) * 1[Hz]")),
        plots=[field_plot("Temperature amplitude", "Tamp"),
               ResultPlot("Probe phase", f"atan2({weighted}) * 180/pi", kind="quantity",
                          y_label="Phase of the probed temperature (°)",
                          style=PlotStyle(x_log=True)),
               ResultPlot("Probe amplitude", f'sqrt(integral(Tre*{probe}, "Film")^2 + integral(Tim*{probe}, "Film")^2)'
                          f' / integral({probe}, "Film")', kind="quantity",
                          y_label="Amplitude of the probed temperature (K)", style=PlotStyle(x_log=True, y_log=True))])


TEMPLATES = {  # in display order
    # heat transfer
    "heat_empty": Template("Empty model", heat_empty, "heat"),
    "heat_bar": Template("Simple heated bar", heat_bar, "heat"),
    "heat_convection": Template("Heated plate cooled by convection", heat_convection, "heat"),
    "heat_pad": Template("Laser pulse on the ribbon with pad", heat_pad, "heat"),
    "heat_radiation": Template("Radiative cooling", heat_radiation, "heat"),
    "heat_diode": Template("Thermal diode", heat_diode, "heat"),
    "heat_contact": Template("Thermal boundary resistance", heat_contact, "heat"),
    "heat_3omega": Template("3ω method: heater line on glass", heat_3omega, "heat"),
    "heat_fdtr": Template("FDTR: gold transducer on silicon", heat_fdtr, "heat"),
}
