"""AutoDL single-GPU runtime for quality-preserving BasicVSR++ inference.

The model, checkpoint, spatial cosine weights, temporal weights, tile padding and
FP16 policy are inherited unchanged from the guarded stage-two runtime.  The
only behavioral change is the single-GPU reduction path: overlap-add and weight
normalization stay on CUDA and only the final normalized clip is copied to CPU.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch

from . import basicvsrpp_stage2_guard as _guard


for _name, _value in vars(_guard).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

_quality = _guard._stage2._quality


class BasicVSRPPPreprocessor(_guard.BasicVSRPPPreprocessor):
    """Guarded stage-two BasicVSR++ with a CUDA-resident single-GPU reduction."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._autodl_single_gpu = (
            len(self._runners) == 1 and self._runners[0].device.type == "cuda"
        )
        if self._autodl_single_gpu:
            print(
                "[basicvsrpp] AutoDL single-GPU GPU-normalize path enabled",
                flush=True,
            )

    def _enhance_tensor(self, clip: torch.Tensor) -> torch.Tensor:
        if not self._autodl_single_gpu:
            return super()._enhance_tensor(clip)

        _n, _t, _c, height, width = clip.shape
        runner = self._runners[0]
        device = runner.device

        if self.config.tile_size == 0:
            wall_started = time.monotonic()
            result = runner._run_model(clip)
            torch.cuda.synchronize(device)
            self.parallel_wall_seconds += time.monotonic() - wall_started
            transfer_started = time.monotonic()
            result_cpu = result.float().cpu()
            self.d2h_seconds += time.monotonic() - transfer_started
            return result_cpu

        tile_size = int(self.config.tile_size)
        context = int(self.config.tile_pad)
        blend_overlap = min(max(context * 2, 32), max(1, tile_size // 4))
        x_starts = _quality._axis_starts(width, tile_size, blend_overlap)
        y_starts = _quality._axis_starts(height, tile_size, blend_overlap)
        x_ends = [min(start + tile_size, width) for start in x_starts]
        y_ends = [min(start + tile_size, height) for start in y_starts]
        x_weights = [
            _quality._axis_weight(start, end, index, x_starts, x_ends)
            for index, (start, end) in enumerate(zip(x_starts, x_ends))
        ]
        y_weights = [
            _quality._axis_weight(start, end, index, y_starts, y_ends)
            for index, (start, end) in enumerate(zip(y_starts, y_ends))
        ]

        jobs: list[tuple[object, ...]] = []
        tile_index = 0
        for y_index, (y0, y1) in enumerate(zip(y_starts, y_ends)):
            context_y0 = max(0, y0 - context)
            context_y1 = min(height, y1 + context)
            for x_index, (x0, x1) in enumerate(zip(x_starts, x_ends)):
                context_x0 = max(0, x0 - context)
                context_x1 = min(width, x1 + context)
                spatial_weight = np.multiply.outer(
                    y_weights[y_index], x_weights[x_index]
                )
                jobs.append(
                    (
                        tile_index,
                        y_index,
                        x_index,
                        y0,
                        y1,
                        x0,
                        x1,
                        context_y0,
                        context_y1,
                        context_x0,
                        context_x1,
                        spatial_weight,
                    )
                )
                tile_index += 1
        self.tiles += len(jobs)

        wall_started = time.monotonic()
        result = torch.zeros(clip.shape, dtype=torch.float32, device=device)
        weight_sum = torch.zeros((height, width), dtype=torch.float32, device=device)
        model_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        lane_start = torch.cuda.Event(enable_timing=True)
        lane_end = torch.cuda.Event(enable_timing=True)

        with torch.cuda.device(device), torch.inference_mode():
            lane_start.record()
            for job in jobs:
                (
                    tile_index,
                    y_index,
                    x_index,
                    y0,
                    y1,
                    x0,
                    x1,
                    context_y0,
                    context_y1,
                    context_x0,
                    context_x1,
                    spatial_weight,
                ) = job
                patch = clip[..., context_y0:context_y1, context_x0:context_x1]
                model_start = torch.cuda.Event(enable_timing=True)
                model_end = torch.cuda.Event(enable_timing=True)
                model_start.record()
                enhanced = runner._run_model(patch)
                model_end.record()
                model_events.append((model_start, model_end))

                crop_y0 = y0 - context_y0
                crop_x0 = x0 - context_x0
                core_height = y1 - y0
                core_width = x1 - x0
                core = enhanced[
                    ...,
                    crop_y0 : crop_y0 + core_height,
                    crop_x0 : crop_x0 + core_width,
                ]
                weight = self._device_weight(
                    runner,
                    (height, width, tile_index, y_index, x_index),
                    spatial_weight,
                )
                result[..., y0:y1, x0:x1].add_(core.float() * weight)
                weight_sum[y0:y1, x0:x1].add_(weight[0, 0, 0])

            if not bool(torch.all(weight_sum > 0).item()):
                raise RuntimeError(
                    "BasicVSR++ tile blending left uncovered output pixels"
                )
            result.div_(weight_sum.view(1, 1, 1, height, width))
            lane_end.record()
            torch.cuda.synchronize(device)

        lane_gpu_seconds = lane_start.elapsed_time(lane_end) / 1000.0
        model_seconds = sum(
            start.elapsed_time(end) for start, end in model_events
        ) / 1000.0
        self.model_gpu_seconds += model_seconds
        self.gpu_blend_seconds += max(0.0, lane_gpu_seconds - model_seconds)
        self.parallel_wall_seconds += time.monotonic() - wall_started

        transfer_started = time.monotonic()
        result_cpu = result.cpu()
        self.d2h_seconds += time.monotonic() - transfer_started
        return result_cpu


class BasicVSRPPStreamReader(_guard.BasicVSRPPStreamReader):
    pass


__all__ = sorted(
    {
        name
        for name in globals()
        if not name.startswith("_")
        and name not in {"np", "torch", "time", "Sequence"}
    }
)
