"""Run long tasks (meshing, solving) off the GUI thread."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, Signal

from ..physics.base import Cancelled


class TaskSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)


class Task(QRunnable):
    """Runs ``fn(signals)`` in a thread pool; signals are delivered on the GUI thread."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = TaskSignals()

    def run(self):
        try:
            try:
                result = self.fn(self.signals)
            except Cancelled:
                self.signals.failed.emit("Cancelled")
            except Exception as exc:  # report everything to the message log
                traceback.print_exc()
                self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            else:
                self.signals.done.emit(result)
        except RuntimeError:
            pass  # the window was closed while the task ran: nobody is listening any more
