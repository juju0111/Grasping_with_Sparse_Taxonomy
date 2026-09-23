"""Headless-safe GUI import guard.

``sim_core.utility.utils`` does a module-level ``import pyautogui``
(only used for an interactive screen-size helper, ``pyautogui.size()``, inside a
function that is never called during training/eval). On a headless server with
no ``DISPLAY`` that import chain crashes::

    import pyautogui → import mouseinfo →
    _display = Display(os.environ['DISPLAY'])   # KeyError: 'DISPLAY'

Training and evaluation never touch the GUI, so importing this module (or calling
:func:`ensure_headless_gui_stubs`) **before the first heavy ``grit.*`` import**
registers harmless no-op stand-ins for ``pyautogui`` / ``mouseinfo`` when there
is no display — letting the transitive ``import pyautogui`` succeed without X.

Env knobs:
  * ``GRIT_FORCE_GUI=1``       → never stub (a real display is present / wanted).
  * ``GRIT_FORCE_HEADLESS=1``  → stub even if ``DISPLAY`` is set (e.g. a stale
                                 DISPLAY pointing at an unreachable X server).

Pure stdlib so it is import-safe from the earliest bootstrap (this package's
``__init__`` files are empty, so importing it pulls in nothing heavy).
"""
from __future__ import annotations

import os
import sys
import types

# GUI-automation modules that import Xlib at module load and need a display.
_GUI_MODULES = ("pyautogui", "mouseinfo")


class _HeadlessGUIStub(types.ModuleType):
    """No-op stand-in module: any attribute access returns a do-nothing callable.

    Only ``import pyautogui`` executes at import time in the dependency chain;
    the real GUI calls live inside functions that are never invoked headless, so
    a permissive no-op is sufficient and never touches X.
    """

    _is_headless_stub = True

    # GUI helpers whose return value is unpacked by callers (e.g.
    # ``w, h = pyautogui.size()`` inside ``sim_core`` 's
    # ``get_monitor_size()``). A bare no-op returning ``None`` would make those
    # callers raise ``TypeError: cannot unpack non-iterable NoneType``. Return a
    # plausible (width, height) / (x, y) tuple instead — the value only feeds
    # viewer/render paths that never execute headless, so any sane default works.
    _TUPLE_RETURNING = {"size": (1920, 1080), "position": (0, 0)}

    def __getattr__(self, name):
        # Dunders (``__file__``, ``__path__``, ``__spec__`` …) must raise so the
        # import / ``inspect`` machinery treats this like a normal module with no
        # such attribute. Returning a callable here breaks ``inspect.getsourcefile``
        # (it would do ``<function>.endswith(...)``) during e.g. torch's fake-op
        # registration, which walks ``sys.modules``. Only real GUI attribute
        # accesses (``size``, ``click`` …) get the no-op.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        if name in self._TUPLE_RETURNING:
            _ret = self._TUPLE_RETURNING[name]
            def _tuple_noop(*_args, **_kwargs):
                return _ret
            return _tuple_noop

        def _noop(*_args, **_kwargs):
            return None
        return _noop


def ensure_headless_gui_stubs() -> bool:
    """Register GUI stubs when running without a usable display.

    Idempotent and safe to call multiple times. Returns ``True`` if (any) stub
    was installed, ``False`` otherwise. Does nothing if the real module is
    already imported (a display was present) or if ``GRIT_FORCE_GUI`` is set.
    """
    if os.environ.get("GRIT_FORCE_GUI"):
        return False
    headless = (not os.environ.get("DISPLAY")) or bool(os.environ.get("GRIT_FORCE_HEADLESS"))
    if not headless:
        return False

    installed = False
    for name in _GUI_MODULES:
        if name not in sys.modules:                  # don't clobber a real import
            sys.modules[name] = _HeadlessGUIStub(name)
            installed = True
    return installed


# Run on import so a bare ``import grit.util.headless_guard`` is enough.
ensure_headless_gui_stubs()
