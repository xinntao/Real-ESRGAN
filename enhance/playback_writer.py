"""Playback-compatible FFmpeg writer for enhanced video output.

The enhancement pipeline remains float-first and unchanged.  This writer only
controls final video timestamps and encoder compatibility:

* raw frames are declared with an exact rational input framerate;
* output is explicitly constant-frame-rate with an exact rational rate;
* the MP4 track timescale is chosen to represent one frame exactly;
* HEVC profile is made explicit for the selected 8/10-bit pixel format.
"""

from __future__ import annotations

import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Sequence

import numpy as np


def _track_timescale(rate_text: str) -> int:
    """Choose an MP4 timescale with an integral duration for every frame."""
    rate = Fraction(rate_text)
    # For 24000/1001 and 30000/1001 this produces 24000/30000, so each frame
    # lasts exactly 1001 ticks.  Small integer rates use 1000 ticks per frame.
    numerator = int(rate.numerator)
    if numerator < 1000:
        numerator *= 1000
    return max(1000, min(numerator, 2_000_000_000))


def install_playback_writer(base: object) -> type:
    """Install and return a timestamp-stable replacement for RawVideoWriter."""

    class PlaybackSafeRawVideoWriter:
        def __init__(
            self,
            path: Path,
            ffmpeg_bin: str,
            width: int,
            height: int,
            input_fps_rate: str,
            output_fps_rate: str,
            codec: str,
            crf: int,
            preset: str,
            cq: int,
            nvenc_preset: str,
            encode_gpu: int,
            output_pix_fmt: str,
            video_filters: Sequence[str],
            color_spec: object,
        ) -> None:
            input_rate = str(Fraction(input_fps_rate))
            output_rate = str(Fraction(output_fps_rate))
            command = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb48le",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                input_rate,
                "-i",
                "pipe:0",
                "-an",
            ]

            filters = list(video_filters)
            if output_rate != input_rate:
                filters.append(f"fps=fps={output_rate}:round=near")
            if filters:
                command += ["-vf", ",".join(filters)]

            command += [
                "-r",
                output_rate,
                "-fps_mode",
                "cfr",
                "-c:v",
                codec,
            ]
            if codec in {"libx264", "libx265"}:
                command += ["-preset", preset, "-crf", str(crf)]
            else:
                command += [
                    "-gpu",
                    str(encode_gpu),
                    "-preset",
                    nvenc_preset,
                    "-tune",
                    "hq",
                    "-rc",
                    "vbr",
                    "-cq",
                    str(cq),
                    "-b:v",
                    "0",
                    "-multipass",
                    "fullres",
                    "-spatial_aq",
                    "1",
                    "-temporal_aq",
                    "1",
                    "-rc-lookahead",
                    "32",
                    "-bf",
                    "3",
                ]

            if codec == "hevc_nvenc":
                profile = "main10" if output_pix_fmt in {"p010le", "yuv420p10le"} else "main"
                command += ["-profile:v", profile]
            elif codec == "h264_nvenc":
                command += ["-profile:v", "high"]

            command += [
                "-pix_fmt",
                output_pix_fmt,
                *base.color_output_args(color_spec),
                "-video_track_timescale",
                str(_track_timescale(output_rate)),
                "-movflags",
                "+faststart",
            ]
            if codec in {"libx265", "hevc_nvenc"}:
                command += ["-tag:v", "hvc1"]
            command.append(str(path))

            print(
                f"[playback] cfr={output_rate}, pix_fmt={output_pix_fmt}, "
                f"track_timescale={_track_timescale(output_rate)}, codec={codec}",
                flush=True,
            )
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        def write(self, frame: np.ndarray) -> None:
            assert self.process.stdin is not None
            if frame.dtype != np.float32 or frame.ndim != 3 or frame.shape[2] != 3:
                raise TypeError("RawVideoWriter expects float32 HWC RGB")
            if not np.isfinite(frame).all():
                raise ValueError("Non-finite value found before rgb48le encoding")
            frame16 = np.rint(np.clip(frame, 0.0, 1.0) * 65535.0).astype("<u2")
            try:
                self.process.stdin.write(
                    memoryview(np.ascontiguousarray(frame16)).cast("B")
                )
            except BrokenPipeError as error:
                detail = (
                    self.process.stderr.read().decode(errors="replace")
                    if self.process.stderr
                    else ""
                )
                raise RuntimeError(
                    f"ffmpeg encoder closed its input early:\n{detail}"
                ) from error

        def close(self) -> None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            stderr = self.process.stderr.read() if self.process.stderr is not None else b""
            if self.process.stderr is not None:
                self.process.stderr.close()
            return_code = self.process.wait()
            if return_code != 0:
                raise RuntimeError(
                    f"ffmpeg encode failed (exit {return_code}):\n"
                    f"{stderr.decode(errors='replace')}"
                )

    base.RawVideoWriter = PlaybackSafeRawVideoWriter
    return PlaybackSafeRawVideoWriter


__all__ = ["install_playback_writer"]
