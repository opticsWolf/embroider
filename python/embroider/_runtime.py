"""Find the ONNX Runtime library, the one way every consumer finds it.

embroider, bobine and okfgraph load ONNX Runtime dynamically through
``ort`` (``ORT_DYLIB_PATH``). Before 0.15 bobine and okfgraph each found
the library by importing the Python ``onnxruntime`` module, which loads a
second copy of the runtime into the process.

The search is embroider's Rust locator (``_native.locate_runtime``), the
same one every model open runs. This layer hands it the folder
:func:`importlib.util.find_spec` resolves for ``onnxruntime`` (the copy
Python itself would import: user site, ``PYTHONPATH`` and editable
installs included) and sets ``ORT_DYLIB_PATH`` through :data:`os.environ`,
so Python code and child processes read the same value. It never imports
the module.
"""
from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Optional

from embroider import _native


def _native_os(os_name: Optional[str], sys_platform: Optional[str]) -> str:
    if (os.name if os_name is None else os_name) == "nt":
        return "windows"
    if (sys.platform if sys_platform is None else sys_platform) == "darwin":
        return "macos"
    return "linux"


def library_in_package(package_dir: Path, os_name: Optional[str] = None,
                       sys_platform: Optional[str] = None) -> Optional[Path]:
    """The ONNX Runtime library inside an installed package, or ``None``.

    Linux prefers the newest versioned ``libonnxruntime.so.*`` over a bare
    ``libonnxruntime.so``; macOS the bare ``libonnxruntime.dylib`` over a
    versioned one. *os_name* / *sys_platform* default to this machine's
    and are parameters so every branch is testable anywhere.
    """
    found = _native.ort_library_in_package(
        str(package_dir), _native_os(os_name, sys_platform))
    return Path(found) if found else None


def _package_dir() -> Optional[Path]:
    """The ``onnxruntime`` package folder, found without importing it."""
    try:
        spec = find_spec("onnxruntime")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    if spec.origin:
        return Path(spec.origin).parent
    return None


def locate_runtime(*, set_env: bool = True) -> Dict[str, Any]:
    """Find the ONNX Runtime library and, by default, point ``ort`` at it.

    Returns ``{"dylib", "source", "version"}``:

    1. ``ORT_DYLIB_PATH`` already set: it wins, ``source="env"``.
    2. A pip-installed ``onnxruntime`` / ``onnxruntime-gpu`` where Python
       would import it from: ``source`` names the distribution
       (``pip:onnxruntime-gpu``) and ``version`` its version, both read
       from its dist-info. With *set_env* the path becomes
       ``ORT_DYLIB_PATH`` and, on Windows, its folder a DLL search
       directory (the GPU build's CUDA provider sits beside it).
    3. Nothing: ``dylib=None``, ``source="os"``. ``ort`` would then try
       the OS loader path, which can hold a stale system copy; consumers
       should refuse to open a session instead.

    Never imports ``onnxruntime`` and never raises.
    """
    package = _package_dir()
    hints = [str(package.parent)] if package is not None else []
    found = dict(_native.locate_runtime(hints, search_prefix=False, set_env=False))
    if set_env and found["source"] != "env" and found["dylib"]:
        os.environ["ORT_DYLIB_PATH"] = found["dylib"]
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if os.name == "nt" and callable(add_dll_directory):
            try:
                add_dll_directory(str(Path(found["dylib"]).parent))
            except OSError:
                pass
    return found
