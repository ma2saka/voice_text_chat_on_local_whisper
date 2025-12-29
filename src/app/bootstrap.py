from __future__ import annotations

import ctypes
import os
from pathlib import Path

from dotenv import load_dotenv


def bootstrap() -> None:
    load_dotenv()
    prepare_cuda_runtime()


def prepare_cuda_runtime() -> None:
    _append_cuda_lib_paths()
    _preload_cuda_libs()


def _append_cuda_lib_paths() -> None:
    candidates = _cuda_lib_dirs()
    if not candidates:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in existing.split(":") if p]
    for path in candidates:
        path_str = str(path)
        if path_str not in parts:
            parts.append(path_str)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _cuda_lib_dirs() -> list[Path]:
    base = Path(".venv/lib")
    if not base.exists():
        base = Path("~/.venv/lib").expanduser()
    candidates: list[Path] = []
    for root in base.glob("python*/site-packages/nvidia"):
        for lib_dir in root.glob("*/lib*"):
            if lib_dir.is_dir():
                candidates.append(lib_dir)
    return candidates


def _preload_cuda_libs() -> None:
    lib_dirs = _cuda_lib_dirs()
    if not lib_dirs:
        return
    lib_names = [
        "libcublas.so",
        "libcublasLt.so",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
    ]
    for lib_dir in lib_dirs:
        for name in lib_names:
            for path in sorted(lib_dir.glob(f"{name}*")):
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
