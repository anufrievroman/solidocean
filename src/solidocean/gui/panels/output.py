"""Output panel: where a run's files go ([output]) and which files are written, plus saving the run in
memory. Headless runs write a folder every time; in the GUI that happens on autosave or on demand."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit

from ... import outputs
from .. import theme
from .common import Panel, muted_label


class OutputPanel(Panel):
    title = "Output"
    help = ("Where the results of a run are written and which files they are. A headless run "
            "(solidocean model.toml) always writes its folder; in the GUI, autosave writes one after every "
            "Run, and the buttons below save the results in memory.")

    def build(self):
        out = self.p.output
        root = QLineEdit(out.dir)
        root.setPlaceholderText(self.p.path.stem if self.p.path else "<model name>")
        root.setToolTip("Results root, relative to the model file. Empty: a folder named after the model "
                        "file, next to it.")
        root.textEdited.connect(lambda text: (setattr(out, "dir", text.strip()), self.win.mark_dirty(),
                                              self._preview()))
        self.field("Results in", root)

        folder = QLineEdit(out.folder)
        folder.setPlaceholderText("all runs in the folder above")
        folder.setToolTip("One folder per run inside it, built from this template. Empty: every run writes "
                          "into the results root, overwriting the files of the previous one.")
        folder.textEdited.connect(lambda text: (setattr(out, "folder", text), self.win.mark_dirty(),
                                                self._preview()))
        self.field("Run folder", folder)
        self.shown = muted_label("")
        self.outer.addWidget(self.shown)

        exists = QComboBox()
        for key in outputs.ON_EXISTS:
            exists.addItem(key, key)
            exists.setItemData(exists.count() - 1, outputs.ON_EXISTS_HELP[key], Qt.ItemDataRole.ToolTipRole)
        exists.setCurrentIndex(max(0, exists.findData(out.on_exists)))
        exists.setToolTip("What to do when that folder exists already — rerunning with the same parameters "
                          "must not destroy the earlier run.")
        exists.currentIndexChanged.connect(lambda i: (setattr(out, "on_exists", exists.itemData(i)),
                                                      self.win.mark_dirty(), self._preview()))
        self.field_row(("If it exists", exists), stretch=False)
        self._placeholder_help()

        g = self.form("Files of a run")
        for key in outputs.CONTENTS:
            label, _, tip = outputs.CONTENT_HELP[key].partition(": ")
            box = QCheckBox(label)
            box.setChecked(out.wants(key))
            box.setToolTip(tip)
            box.toggled.connect(lambda on, key=key: self._set_content(key, on))
            g.addRow(box)

        auto = QCheckBox("Write a run folder after every Run")
        auto.setChecked(out.autosave)
        auto.setToolTip("The model must be saved to a file first: run folders are written next to it.")
        auto.toggled.connect(lambda on: (setattr(out, "autosave", on), self.win.mark_dirty()))
        h = self.form("Autosave")
        h.addRow(auto)

        if self.win.results is None:
            self.outer.addWidget(muted_label("No results in memory yet. Configure the Study and press Run."))
        else:
            self.button_row(("Save this run", lambda: self.win.save_run()),
                            ("Save to folder…", self.win.export_results))
        self._preview()

    def _placeholder_help(self):
        """What may be written in the run folder template, collapsed away until it is wanted."""
        lines = [f"<b>{{{name}}}:</b> {what}" for name, what in outputs.BUILTIN_HELP.items()]
        if self.p.variables:
            lines.append("<b>" + ", ".join(f"{{{v}}}" for v in self.p.variables) + ":</b> the model's "
                         "variables, as the value each one has in the run. A swept variable has no single "
                         "value; use {sweep} for it.")
        self.spoiler("Available placeholders", "<br>".join(lines), "show_placeholder_help")

    def _set_content(self, key: str, on: bool):
        """Keep [output] contents in the order of outputs.CONTENTS, so the file reads the same every time."""
        out = self.p.output
        chosen = (set(out.contents) | {key}) if on else (set(out.contents) - {key})
        out.contents = [c for c in outputs.CONTENTS if c in chosen]
        self.win.mark_dirty()

    def _preview(self):
        """Show the folder the next run would write into, or the first problem with the settings."""
        problems = outputs.problems(self.p)
        if problems:
            self.shown.setText(problems[0])
            self.shown.setStyleSheet(f"color: {theme.error()};")
            return
        self.shown.setStyleSheet(f"color: {theme.muted()};")
        if self.p.path is None:
            self.shown.setText("The model has no file yet; save it, and runs are written next to it.")
        else:
            self.shown.setText(f"Next run: {self.p.run_dir()}")
