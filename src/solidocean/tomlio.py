"""Writing model files: compact, readable TOML.

Settings are written without the values that equal their defaults (loading fills them in again), and short
arrays stay on one line: `center = ["0", "0", "t/2"]` rather than one element per line.
"""

from __future__ import annotations

import dataclasses
import difflib
import re

import tomli_w

LINE = 100   # longest array kept on one line (characters, including the key)


def check_keys(d: dict, known, where: str) -> None:
    """Refuse the keys of a table that are not among `known`: a misspelt key would otherwise be dropped, and
    the model silently be a different one."""
    known = list(known)
    unknown = [k for k in d if k not in known]
    if not unknown:
        return
    hints = []
    for k in unknown:
        close = difflib.get_close_matches(k, known, n=1)
        hints.append(f"'{k}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
    raise ValueError(f"{where}: unknown key{'s' if len(unknown) > 1 else ''} {', '.join(hints)}; "
                     f"known: {', '.join(sorted(known))}")


def table(cls, d: dict, where: str):
    """cls(**d) for a table of the model file, refusing keys that are not settings of the dataclass cls."""
    check_keys(d, [f.name for f in dataclasses.fields(cls) if f.init], where)
    return cls(**d)


def label(d: dict, default: str = "") -> str:
    """How an entry of a list of tables is named in messages: its name, else its type."""
    return f"'{d.get('name') or d.get('type') or default}'"


def without_defaults(obj, d: dict, keep: tuple[str, ...] = ()) -> dict:
    """The table `d` of a dataclass instance `obj` without the keys whose value equals the field's default
    (keys in `keep` always stay)."""
    defaults = {}
    for f in dataclasses.fields(obj):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()
    out = {}
    for key, value in d.items():
        default = defaults.get(key, dataclasses.MISSING)
        if key in keep or default is dataclasses.MISSING or not _same(value, default):
            out[key] = value
    return out


def _same(value, default) -> bool:
    if isinstance(default, (list, tuple)):
        return isinstance(value, (list, tuple)) and list(value) == list(default)
    if isinstance(default, bool) or isinstance(value, bool):
        return value is default
    return not dataclasses.is_dataclass(default) and value == default


_ITEM = r'[ \t]+(?:"(?:[^"\\\n]|\\.)*"|[^\n\[\]{}"]+),\n'      # one scalar per line: a string or a number
_ARRAY = re.compile(r"^(?P<head>[^\n=]+ = )\[\n(?P<body>(?:" + _ITEM + r")+)\]$", re.M)
_INNER = re.compile(r"^(?P<indent>[ \t]+)\[\n(?P<body>(?:" + _ITEM + r")+)[ \t]+\],?$", re.M)


def _join(body: str) -> str:
    return ", ".join(item.strip().rstrip(",") for item in body.strip("\n").split("\n"))


def dumps(d: dict) -> str:
    """TOML text of a model table, with short arrays (and short rows of nested arrays) on one line."""
    text = tomli_w.dumps(d)
    # rows of nested arrays (e.g. lattice vectors, polygon points) first, then whole arrays
    text = _INNER.sub(lambda m: f"{m.group('indent')}[{_join(m.group('body'))}],", text)

    def one_line(m):
        line = f"{m.group('head')}[{_join(m.group('body'))}]"
        return line if len(line) <= LINE else m.group(0)
    text = _ARRAY.sub(one_line, text)
    # an outer array whose rows are now single lines: one line too, if short
    nested = re.compile(r"^(?P<head>[^\n=]+ = )\[\n(?P<body>(?:[ \t]+\[[^\n]*\],\n)+)\]$", re.M)
    text = nested.sub(lambda m: (f"{m.group('head')}[{_join(m.group('body'))}]"
                                 if len(m.group('head')) + len(m.group('body')) <= LINE else m.group(0)), text)
    return text
