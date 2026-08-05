#!/usr/bin/env python3
"""Kaggle-safe entry point for the shared-memory Real-ESRGAN runner.

Large native-scale output arrays may exceed Docker's POSIX shared-memory quota.
Use POSIX shared memory when there is ample capacity and transparently fall back
to a persistent file-backed mmap otherwise. Both paths avoid Queue pickling of
frame/tile arrays.

The repository has a top-level ``realesrgan.py`` runner. Load that adjacent
file explicitly as the ``realesrgan`` module before importing
``realesrgan_fast``. The project vendors SRVGGNetCompact locally, so no
separate installable ``realesrgan`` package is required or expected.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parent


def _load_local_realesrgan() -> ModuleType:
    """Load the adjacent CLI/video runner without relying on import search order."""
    module_path = _REPOSITORY_ROOT / "realesrgan.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Local Real-ESRGAN runner not found: {module_path}")

    spec = importlib.util.spec_from_file_location("realesrgan", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses and spawned workers can resolve
    # the defining module while the file is being imported.
    sys.modules["realesrgan"] = module
    spec.loader.exec_module(module)

    required = (
        "build_parser",
        "process_video",
        "PersistentWorkers",
        "TileProcessor",
        "WorkerConfig",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"Local runner {module_path} is missing required symbols: {missing}"
        )
    return module


_LOCAL_BASE = _load_local_realesrgan()

import realesrgan_fast as fast  # noqa: E402  (requires local-module bootstrap)
from enhance.playback_writer import install_playback_writer  # noqa: E402

# Be explicit even though realesrgan_fast imported the same sys.modules entry.
fast.base = _LOCAL_BASE
install_playback_writer(_LOCAL_BASE)

_original_shared_array_init = fast.SharedArray.__init__


def _guarded_shared_array_init(
    self: fast.SharedArray,
    shape: tuple[int, ...],
    dtype: np.dtype,
    prefix: str,
) -> None:
    resolved_shape = tuple(int(value) for value in shape)
    resolved_dtype = np.dtype(dtype)
    required = int(np.prod(resolved_shape, dtype=np.int64)) * resolved_dtype.itemsize
    shm_path = Path("/dev/shm")
    use_posix_shm = False
    if shm_path.is_dir():
        try:
            free = shutil.disk_usage(shm_path).free
            use_posix_shm = required <= int(free * 0.70)
        except OSError:
            use_posix_shm = False

    if use_posix_shm:
        _original_shared_array_init(self, resolved_shape, resolved_dtype, prefix)
        return

    self.shape = resolved_shape
    self.dtype = resolved_dtype
    self.size = required
    self.shm = None
    root = Path("/kaggle/working")
    if not root.is_dir():
        root = Path(tempfile.gettempdir())
    fd, path = tempfile.mkstemp(prefix=f"{prefix}-", suffix=".mmap", dir=root)
    os.close(fd)
    with open(path, "wb") as handle:
        handle.truncate(required)
    self.path = path
    self.array = np.memmap(path, mode="r+", dtype=resolved_dtype, shape=resolved_shape)
    self.spec = fast.SharedArraySpec("mmap", path, resolved_shape, resolved_dtype.str)


fast.SharedArray.__init__ = _guarded_shared_array_init


if __name__ == "__main__":
    fast.mp.freeze_support()
    fast.main()
