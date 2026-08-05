#!/usr/bin/env python3
"""AutoDL RTX 4090 single-GPU entry point for the AutoDL branch."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_RUNTIME_ROOT = Path("/root/autodl-tmp/realesrgan-runtime")
_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(_RUNTIME_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from realesrgan_fast_entry import _LOCAL_BASE, fast  # noqa: E402
from realesrgan_autodl_runtime import install  # noqa: E402

install(_LOCAL_BASE, fast)


def _preflight() -> None:
    import torch
    import torchvision
    from torchvision.ops import deform_conv2d

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"AutoDL branch requires exactly one visible GPU; found {torch.cuda.device_count()}"
        )
    name = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if "4090" not in name:
        raise RuntimeError(f"Expected RTX 4090/4090D, found {name}")
    if memory_gib < 20.0:
        raise RuntimeError(f"Expected about 24 GiB VRAM, found {memory_gib:.1f} GiB")

    sample = torch.randn(1, 1, 8, 8, device="cuda:0")
    offset = torch.zeros(1, 18, 8, 8, device="cuda:0")
    weight = torch.randn(1, 1, 3, 3, device="cuda:0")
    output = deform_conv2d(sample, offset, weight, padding=1)
    if output.shape != sample.shape or not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("torchvision deform_conv2d CUDA preflight failed")
    del sample, offset, weight, output
    torch.cuda.synchronize(0)
    torch.cuda.empty_cache()

    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise FileNotFoundError(
                f"{binary} was not found; install it with apt-get install -y ffmpeg"
            )
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or "hevc_nvenc" not in result.stdout + result.stderr:
        raise RuntimeError("FFmpeg does not provide hevc_nvenc")

    print(
        f"[autodl] gpu={name}, vram={memory_gib:.1f}GiB, "
        f"python={sys.version.split()[0]}, torch={torch.__version__}, "
        f"torchvision={torchvision.__version__}, torch_cuda={torch.version.cuda}, "
        f"cudnn={torch.backends.cudnn.version()}",
        flush=True,
    )
    print("[autodl] deform_conv2d and HEVC NVENC preflight OK", flush=True)


if __name__ == "__main__":
    fast.mp.freeze_support()
    _preflight()
    fast.main()
