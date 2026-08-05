"""Single-GPU pipelined Real-ESRGAN runtime for the AutoDL branch.

This runtime keeps the same model, checkpoints, direct-write tile geometry,
FP16 model execution and final Lanczos resize as the 3.6 baseline.  It replaces
the single-GPU multiprocessing round trip with:

* two CUDA frame slots;
* persistent pinned host staging buffers;
* independent H2D/compute/D2H CUDA streams;
* one ordered CPU post-processing thread, overlapping frame N+1 inference with
  frame N Lanczos/RGB48 conversion/FFmpeg writing.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

import realesrgan as base
import realesrgan_fast as fast


@dataclass
class _StageSlot:
    pinned: torch.Tensor
    gpu: torch.Tensor
    h2d_start: torch.cuda.Event
    h2d_end: torch.cuda.Event
    compute_done: torch.cuda.Event
    initialized: bool = False


@dataclass
class _FrameSlot:
    index: int
    gpu_output: torch.Tensor
    cpu_output: torch.Tensor
    compute_done: torch.cuda.Event
    copy_start: torch.cuda.Event
    copy_end: torch.cuda.Event
    copy_done: torch.cuda.Event
    h2d_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    model_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list)
    future: Optional[Future] = None
    frame_id: int = -1


class AutoDLSingleGPUEngine:
    """In-process double-buffered inference engine for one CUDA GPU."""

    def __init__(
        self,
        config: base.WorkerConfig,
        tile_pad: int,
        requested_tile: int,
        requested_batch: int,
        auto_tile: bool,
        max_tile: int,
        auto_batch: bool,
        max_batch: int,
    ) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "The AutoDL runtime requires exactly one visible CUDA GPU. "
                "Set CUDA_VISIBLE_DEVICES=0."
            )
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.config = config
        self.tile_pad = int(tile_pad)
        self.requested_tile = max(fast._MIN_TILE_SIZE, int(requested_tile))
        self.requested_batch = max(1, int(requested_batch))
        self.auto_tile = bool(auto_tile)
        self.max_tile = max(fast._MIN_TILE_SIZE, int(max_tile))
        self.auto_batch = bool(auto_batch)
        self.max_batch = max(1, int(max_batch))

        self.model, self.native_scale = base.load_worker_model(config, self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.h2d_stream = torch.cuda.Stream(device=self.device)
        self.d2h_stream = torch.cuda.Stream(device=self.device)

        self.frame_shape: Optional[tuple[int, int, int]] = None
        self.regions: list[fast.FastTileRegion] = []
        self.grouped: dict[tuple[int, int, int], list[fast.FastTileRegion]] = {}
        self.selected_tile = self.requested_tile
        self.selected_batch = self.requested_batch
        self.frame_slots: list[_FrameSlot] = []
        self.stage_cache: dict[tuple[int, int, int], list[_StageSlot]] = {}
        self.stage_cursor: dict[tuple[int, int, int], int] = defaultdict(int)
        self.autotune_seconds = 0.0
        self._closed = False

    @property
    def output_dtype(self) -> torch.dtype:
        return torch.float16 if self.config.fp16 else torch.float32

    def _tensor_from_numpy(self, batch: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(
            self.device,
            dtype=self.output_dtype,
        )
        tensor.clamp_(0.0, 1.0)
        if self.config.channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _probe(self, frame: np.ndarray, tile: int, batch_size: int) -> tuple[bool, float]:
        patch_h = min(frame.shape[0], tile + 2 * self.tile_pad)
        patch_w = min(frame.shape[1], tile + 2 * self.tile_pad)
        y0 = max(0, (frame.shape[0] - patch_h) // 2)
        x0 = max(0, (frame.shape[1] - patch_w) // 2)
        patch = np.ascontiguousarray(frame[y0 : y0 + patch_h, x0 : x0 + patch_w])
        batch = np.repeat(patch[None, ...], batch_size, axis=0)
        started = time.monotonic()
        tensor = None
        reserve_input = None
        output = None
        try:
            tensor = self._tensor_from_numpy(batch)
            reserve_input = torch.empty_like(tensor)
            with torch.inference_mode():
                output = self.model(tensor)
                output.clamp_(0.0, 1.0)
            torch.cuda.synchronize(self.device)
            elapsed = time.monotonic() - started
            del output, reserve_input, tensor, batch
            torch.cuda.empty_cache()
            return True, elapsed
        except torch.cuda.OutOfMemoryError:
            del output, reserve_input, tensor, batch
            torch.cuda.empty_cache()
            return False, 0.0

    def _autotune(self, frame: np.ndarray) -> None:
        started = time.monotonic()
        height, width = frame.shape[:2]
        candidates = (
            [
                candidate
                for candidate in fast._tile_candidates(width, height, self.max_tile)
                if candidate >= self.requested_tile
            ]
            if self.auto_tile
            else [self.requested_tile]
        )
        if not candidates:
            candidates = [self.requested_tile]
        selected: Optional[int] = None
        best_estimate: Optional[tuple[float, int]] = None
        for candidate in candidates:
            ok, seconds = self._probe(frame, candidate, 1)
            if not ok:
                print(f"[autodl auto-tile] rejected={candidate} OOM", flush=True)
                continue
            region_count = len(fast._regions(width, height, candidate, self.tile_pad))
            estimate = seconds * region_count
            print(
                f"[autodl auto-tile] candidate={candidate}, tiles={region_count}, "
                f"probe={seconds:.3f}s, estimated_model={estimate:.3f}s",
                flush=True,
            )
            if best_estimate is None or estimate < best_estimate[0]:
                best_estimate = (estimate, candidate)
                selected = candidate

        if selected is None:
            raise RuntimeError(
                f"No Real-ESRGAN tile >= {fast._MIN_TILE_SIZE} fits the resident 4090 state"
            )
        self.selected_tile = selected
        self.regions = fast._regions(width, height, selected, self.tile_pad)
        grouped: dict[tuple[int, int, int], list[fast.FastTileRegion]] = defaultdict(list)
        for region in self.regions:
            grouped[region.patch_shape].append(region)
        self.grouped = dict(grouped)
        max_group = max(len(group) for group in self.grouped.values())

        if self.auto_batch:
            best = 1
            for candidate in fast._batch_candidates(min(self.max_batch, max_group)):
                ok, seconds = self._probe(frame, selected, candidate)
                if not ok:
                    break
                best = candidate
                print(
                    f"[autodl auto-batch] candidate={candidate}, probe={seconds:.3f}s",
                    flush=True,
                )
            self.selected_batch = best
        else:
            self.selected_batch = min(self.requested_batch, max_group)

        self.autotune_seconds += time.monotonic() - started
        print(
            f"[autodl] selected tile={self.selected_tile}, batch={self.selected_batch}, "
            f"tiles={len(self.regions)}, shapes={len(self.grouped)}, "
            f"autotune={self.autotune_seconds:.1f}s",
            flush=True,
        )

    def _make_frame_slot(self, index: int, height: int, width: int) -> _FrameSlot:
        output_shape = (
            height * self.native_scale,
            width * self.native_scale,
            3,
        )
        gpu_output = torch.empty(
            output_shape,
            dtype=self.output_dtype,
            device=self.device,
        )
        cpu_output = torch.empty(
            output_shape,
            dtype=self.output_dtype,
            device="cpu",
            pin_memory=True,
        )
        return _FrameSlot(
            index=index,
            gpu_output=gpu_output,
            cpu_output=cpu_output,
            compute_done=torch.cuda.Event(),
            copy_start=torch.cuda.Event(enable_timing=True),
            copy_end=torch.cuda.Event(enable_timing=True),
            copy_done=torch.cuda.Event(),
        )

    def initialize(self, frame: np.ndarray) -> None:
        if self.frame_shape is not None:
            if tuple(frame.shape) != self.frame_shape:
                raise RuntimeError(
                    f"Frame shape changed during inference: {frame.shape} != {self.frame_shape}"
                )
            return
        if frame.dtype != np.float32:
            raise TypeError(f"AutoDL engine expects float32 RGB frames, got {frame.dtype}")
        self.frame_shape = tuple(int(value) for value in frame.shape)
        height, width = frame.shape[:2]
        self.frame_slots = [
            self._make_frame_slot(0, height, width),
            self._make_frame_slot(1, height, width),
        ]
        self._autotune(frame)
        for shape, group in self.grouped.items():
            capacity = min(self.selected_batch, len(group))
            self._stage_slots(shape, capacity)
        print(
            "[autodl] double-buffered CUDA engine ready: "
            "pinned H2D + compute stream + pinned D2H + ordered CPU postprocess",
            flush=True,
        )

    def _stage_slots(self, shape: tuple[int, int, int], capacity: int) -> list[_StageSlot]:
        key = (shape[0], shape[1], capacity)
        cached = self.stage_cache.get(key)
        if cached is not None:
            return cached
        slots: list[_StageSlot] = []
        for _ in range(2):
            pinned = torch.empty(
                (capacity, shape[0], shape[1], shape[2]),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            memory_format = (
                torch.channels_last
                if self.config.channels_last
                else torch.contiguous_format
            )
            gpu = torch.empty(
                (capacity, shape[2], shape[0], shape[1]),
                dtype=self.output_dtype,
                device=self.device,
                memory_format=memory_format,
            )
            slots.append(
                _StageSlot(
                    pinned=pinned,
                    gpu=gpu,
                    h2d_start=torch.cuda.Event(enable_timing=True),
                    h2d_end=torch.cuda.Event(enable_timing=True),
                    compute_done=torch.cuda.Event(),
                )
            )
        self.stage_cache[key] = slots
        return slots

    def submit(self, frame_id: int, frame: np.ndarray, slot: _FrameSlot) -> None:
        self.initialize(frame)
        if slot.future is not None:
            raise RuntimeError("Attempted to reuse an AutoDL frame slot before CPU completion")
        slot.frame_id = int(frame_id)
        slot.h2d_events.clear()
        slot.model_events.clear()

        for shape in sorted(self.grouped):
            group = self.grouped[shape]
            capacity = min(self.selected_batch, len(group))
            stages = self._stage_slots(shape, capacity)
            key = (shape[0], shape[1], capacity)
            for offset in range(0, len(group), capacity):
                chunk = group[offset : offset + capacity]
                stage_index = self.stage_cursor[key] % len(stages)
                self.stage_cursor[key] += 1
                stage = stages[stage_index]
                if stage.initialized:
                    stage.compute_done.synchronize()

                pinned_view = stage.pinned.numpy()
                for batch_index, region in enumerate(chunk):
                    patch = frame[
                        region.context_y0 : region.context_y1,
                        region.context_x0 : region.context_x1,
                    ]
                    np.copyto(pinned_view[batch_index], patch, casting="no")

                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self.h2d_stream):
                    h2d_start.record(self.h2d_stream)
                    source = stage.pinned[: len(chunk)].permute(0, 3, 1, 2)
                    stage.gpu[: len(chunk)].copy_(source, non_blocking=True)
                    h2d_end.record(self.h2d_stream)
                slot.h2d_events.append((h2d_start, h2d_end))

                model_start = torch.cuda.Event(enable_timing=True)
                model_end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self.compute_stream), torch.inference_mode():
                    self.compute_stream.wait_event(h2d_end)
                    model_start.record(self.compute_stream)
                    native = self.model(stage.gpu[: len(chunk)])
                    native.clamp_(0.0, 1.0)
                    model_end.record(self.compute_stream)
                    slot.model_events.append((model_start, model_end))

                    for batch_index, region in enumerate(chunk):
                        crop_x0 = (
                            region.x0 - region.context_x0
                        ) * self.native_scale
                        crop_y0 = (
                            region.y0 - region.context_y0
                        ) * self.native_scale
                        core_w = (region.x1 - region.x0) * self.native_scale
                        core_h = (region.y1 - region.y0) * self.native_scale
                        core = native[
                            batch_index,
                            :,
                            crop_y0 : crop_y0 + core_h,
                            crop_x0 : crop_x0 + core_w,
                        ].permute(1, 2, 0)
                        oy0 = region.y0 * self.native_scale
                        ox0 = region.x0 * self.native_scale
                        slot.gpu_output[
                            oy0 : oy0 + core_h,
                            ox0 : ox0 + core_w,
                        ].copy_(core)
                    stage.compute_done.record(self.compute_stream)
                    stage.initialized = True

        with torch.cuda.stream(self.compute_stream):
            slot.compute_done.record(self.compute_stream)
        with torch.cuda.stream(self.d2h_stream):
            self.d2h_stream.wait_event(slot.compute_done)
            slot.copy_start.record(self.d2h_stream)
            slot.cpu_output.copy_(slot.gpu_output, non_blocking=True)
            slot.copy_end.record(self.d2h_stream)
            slot.copy_done.record(self.d2h_stream)

    def slot_metrics(self, slot: _FrameSlot) -> dict[str, float]:
        slot.copy_done.synchronize()
        return {
            "realesrgan_h2d_gpu": sum(
                start.elapsed_time(end) for start, end in slot.h2d_events
            )
            / 1000.0,
            "realesrgan_model_gpu": sum(
                start.elapsed_time(end) for start, end in slot.model_events
            )
            / 1000.0,
            "realesrgan_d2h_gpu": slot.copy_start.elapsed_time(slot.copy_end)
            / 1000.0,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize(self.device)
        self.stage_cache.clear()
        self.frame_slots.clear()
        del self.model
        torch.cuda.empty_cache()


def _postprocess_and_write(
    engine: AutoDLSingleGPUEngine,
    slot: _FrameSlot,
    writer: object,
    output_width: int,
    output_height: int,
) -> dict[str, float]:
    metrics = engine.slot_metrics(slot)
    convert_started = time.monotonic()
    native = np.ascontiguousarray(slot.cpu_output.numpy().astype(np.float32, copy=True))
    metrics["fp16_to_fp32"] = time.monotonic() - convert_started

    local_timings = {"lanczos": 0.0}
    output = base.finalize_output_frame(
        native,
        output_width,
        output_height,
        local_timings,
    )
    metrics["lanczos"] = local_timings["lanczos"]
    write_started = time.monotonic()
    writer.write(output)
    metrics["float_to_rgb48"] = time.monotonic() - write_started
    return metrics


def _accumulate(target: Dict[str, float], values: Dict[str, float]) -> None:
    for name, value in values.items():
        target[name] = target.get(name, 0.0) + float(value)


def process_video(args: object) -> None:
    base.require_binary(args.ffmpeg_bin)
    base.require_binary(args.ffprobe_bin)
    base.require_encoder(args.ffmpeg_bin, args.video_codec)
    output_pix_fmt = base.resolve_output_pix_fmt(
        args.ffmpeg_bin, args.video_codec, args.output_pix_fmt
    )
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_video = output_path.with_name(
        output_path.stem + ".video_only.tmp.mp4"
    )

    info = base.probe_video(input_path, args.ffprobe_bin)
    color_spec = base.resolve_color_spec(info, args.color_policy, args.hdr_policy)
    writer_filters = base.color_filter_chain(color_spec, output_pix_fmt)
    source_rate = base.Fraction(info.fps_num, info.fps_den)
    output_rate = base.parse_output_rate(args.fps, source_rate)
    inference_rate = min(source_rate, output_rate)
    output_fps = float(output_rate)
    output_fps_rate = f"{output_rate.numerator}/{output_rate.denominator}"
    inference_fps = float(inference_rate)
    inference_fps_rate = f"{inference_rate.numerator}/{inference_rate.denominator}"
    input_width, input_height = info.width, info.height
    start, duration, expected_frames = base.resolve_range(
        info, args.start_time, args.test_seconds, inference_rate
    )
    expected_output_frames = max(1, int(round(duration * output_fps)))
    end = start + duration
    gpu_ids = base.parse_gpu_ids(args.gpu_ids)
    if gpu_ids != [0]:
        raise ValueError(
            f"AutoDL branch requires --gpu-ids 0; received {args.gpu_ids!r}"
        )
    effective_fp16 = bool(args.fp16)
    effective_channels_last = bool(args.channels_last)
    native_model_scale = 2 if args.model == "RealESRGAN_x2plus" else 4
    if args.scale > native_model_scale:
        raise ValueError(
            f"Final scale {args.scale:g} exceeds model-native scale {native_model_scale}"
        )
    output_width = int(round(input_width * args.scale))
    output_height = int(round(input_height * args.scale))
    if output_width % 2 or output_height % 2:
        raise ValueError(
            f"4:2:0 encoding needs even output dimensions, got "
            f"{output_width}x{output_height}"
        )
    base.probe_encoder_runtime(
        args.ffmpeg_bin,
        args.video_codec,
        output_pix_fmt,
        output_width,
        output_height,
        args.encode_gpu,
        writer_filters,
        color_spec,
    )
    model_paths = base.resolve_model_paths(args)
    config = base.WorkerConfig(
        model_name=args.model,
        model_paths=model_paths,
        denoise_strength=args.denoise_strength,
        native_scale=native_model_scale,
        batch_size=args.batch_size,
        fp16=effective_fp16,
        channels_last=effective_channels_last,
    )

    mode = "timed test" if args.test_seconds > 0 else "selected/full range"
    print(f"[run] wall_start={base.now_text()}", flush=True)
    print(f"[input] {input_path}", flush=True)
    print(
        f"[input] source={info.width}x{info.height}, "
        f"inference={input_width}x{input_height}, "
        f"output={output_width}x{output_height}, "
        f"source_fps={info.fps:.6f}, inference_fps={inference_fps:.6f} "
        f"({inference_fps_rate}), output_fps={output_fps:.6f} "
        f"({output_fps_rate}), audio={info.has_audio}",
        flush=True,
    )
    print(
        f"[range] mode={mode}, start={base.format_seconds(start)}, "
        f"end={base.format_seconds(end)}, duration={duration:.3f}s, "
        f"expected_inference_frames={expected_frames}, "
        f"expected_output_frames={expected_output_frames}",
        flush=True,
    )
    print(
        f"[autodl] single-GPU pipelined mode, model={args.model}, "
        f"fp16={effective_fp16}, channels_last={effective_channels_last}, "
        "frame_slots=2, ordered_cpu_workers=1",
        flush=True,
    )
    if args.basicvsrpp:
        print(
            f"[basicvsrpp] enabled, track={args.basicvsrpp_track}, "
            f"gpu=cuda:{args.basicvsrpp_gpu}, fp16={args.basicvsrpp_fp16}, "
            f"clip={args.basicvsrpp_clip_length}, "
            f"overlap={args.basicvsrpp_clip_overlap}, "
            f"tile={args.basicvsrpp_tile_size}, "
            f"tile_pad={args.basicvsrpp_tile_pad}",
            flush=True,
        )
    base.log_devices(gpu_ids, effective_fp16)

    reader: Optional[object] = None
    basicvsrpp_reader: Optional[object] = None
    writer: Optional[object] = None
    engine: Optional[AutoDLSingleGPUEngine] = None
    executor: Optional[ThreadPoolExecutor] = None
    processed = 0
    started = time.monotonic()
    timings: Dict[str, float] = {
        "model_startup": 0.0,
        "basicvsrpp_startup": 0.0,
        "decode": 0.0,
        "basicvsrpp": 0.0,
        "inference_submit": 0.0,
        "lanczos": 0.0,
        "float_to_rgb48": 0.0,
        "encode_flush": 0.0,
        "audio_mux": 0.0,
    }
    clean_video_ready = False

    try:
        stage_started = time.monotonic()
        engine = AutoDLSingleGPUEngine(
            config=config,
            tile_pad=args.tile_pad,
            requested_tile=args.tile_size,
            requested_batch=args.batch_size,
            auto_tile=fast._AUTO_TILE,
            max_tile=fast._MAX_TILE_SIZE,
            auto_batch=fast._AUTO_BATCH,
            max_batch=fast._MAX_BATCH_SIZE,
        )
        engine.initialize(
            np.zeros((input_height, input_width, 3), dtype=np.float32)
        )
        timings["model_startup"] += time.monotonic() - stage_started

        reader = base.RawVideoReader(
            input_path,
            args.ffmpeg_bin,
            input_width,
            input_height,
            inference_fps_rate,
            start,
            duration,
        )
        if args.basicvsrpp:
            stage_started = time.monotonic()
            preprocessor = base.BasicVSRPPPreprocessor(
                base.BasicVSRPPConfig(
                    track=args.basicvsrpp_track,
                    model_path=args.basicvsrpp_model_path,
                    gpu_id=args.basicvsrpp_gpu,
                    fp16=args.basicvsrpp_fp16,
                    clip_length=args.basicvsrpp_clip_length,
                    clip_overlap=args.basicvsrpp_clip_overlap,
                    tile_size=args.basicvsrpp_tile_size,
                    tile_pad=args.basicvsrpp_tile_pad,
                    strength=args.basicvsrpp_strength,
                    scene_threshold=args.basicvsrpp_scene_threshold,
                ),
                checkpoint_dir=Path(base.__file__).resolve().parent / "weights",
            )
            basicvsrpp_reader = base.BasicVSRPPStreamReader(reader, preprocessor)
            reader = basicvsrpp_reader
            timings["basicvsrpp_startup"] += time.monotonic() - stage_started

        writer = base.RawVideoWriter(
            temporary_video,
            args.ffmpeg_bin,
            output_width,
            output_height,
            inference_fps_rate,
            output_fps_rate,
            args.video_codec,
            args.crf,
            args.preset,
            args.cq,
            args.nvenc_preset,
            args.encode_gpu,
            output_pix_fmt,
            writer_filters,
            color_spec,
        )
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="autodl-postprocess",
        )
        progress = tqdm(
            total=expected_frames,
            desc="Real-ESRGAN AutoDL",
            unit="frame",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        try:
            with base.PeriodicRefresh(progress, args.progress_interval):
                while True:
                    stage_started = time.monotonic()
                    frame = reader.read()
                    read_elapsed = time.monotonic() - stage_started
                    if basicvsrpp_reader is None:
                        timings["decode"] += read_elapsed
                    if frame is None:
                        break

                    engine.initialize(frame)
                    slot = engine.frame_slots[processed % len(engine.frame_slots)]
                    if slot.future is not None:
                        _accumulate(timings, slot.future.result())
                        slot.future = None

                    submit_started = time.monotonic()
                    engine.submit(processed, frame, slot)
                    timings["inference_submit"] += (
                        time.monotonic() - submit_started
                    )
                    slot.future = executor.submit(
                        _postprocess_and_write,
                        engine,
                        slot,
                        writer,
                        output_width,
                        output_height,
                    )
                    processed += 1
                    progress.update(1)
                    elapsed = max(time.monotonic() - started, 1e-6)
                    progress.set_postfix(
                        fps=f"{processed / elapsed:.3f}",
                        refresh=False,
                    )

                for slot in engine.frame_slots:
                    if slot.future is not None:
                        _accumulate(timings, slot.future.result())
                        slot.future = None
        finally:
            progress.close()

        if basicvsrpp_reader is not None:
            timings["decode"] += basicvsrpp_reader.decode_elapsed
            timings["basicvsrpp"] += basicvsrpp_reader.preprocessor.elapsed
            print(
                f"[basicvsrpp] clips={basicvsrpp_reader.preprocessor.clips}, "
                f"tiles={basicvsrpp_reader.preprocessor.tiles}, "
                f"scene_cuts={basicvsrpp_reader.scene_cuts}",
                flush=True,
            )
        reader.close()
        reader = None
        basicvsrpp_reader = None

        executor.shutdown(wait=True, cancel_futures=False)
        executor = None
        stage_started = time.monotonic()
        writer.close()
        timings["encode_flush"] += time.monotonic() - stage_started
        writer = None
        clean_video_ready = True
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if engine is not None:
            engine.close()

    if not clean_video_ready or processed == 0:
        raise RuntimeError("No complete video was encoded.")

    actual_duration = processed / inference_fps
    output_frames = int(round(actual_duration * output_fps))
    stage_started = time.monotonic()
    base.mux_audio(
        temporary_video,
        input_path,
        output_path,
        args.ffmpeg_bin,
        start,
        actual_duration,
        info.has_audio,
        args.audio_codec,
        args.audio_bitrate,
    )
    timings["audio_mux"] += time.monotonic() - stage_started
    if temporary_video.exists():
        temporary_video.unlink()

    elapsed = time.monotonic() - started
    print(
        f"[range] actual_start={base.format_seconds(start)}, "
        f"actual_end={base.format_seconds(start + actual_duration)}, "
        f"processed_inference_frames={processed}, output_frames={output_frames}, "
        f"output_duration={actual_duration:.3f}s",
        flush=True,
    )
    print(
        f"[run] wall_end={base.now_text()}, elapsed={elapsed:.1f}s, "
        f"average={processed / max(elapsed, 1e-6):.3f} frame/s",
        flush=True,
    )
    print(
        "[timing] "
        + ", ".join(f"{name}={value:.1f}s" for name, value in timings.items()),
        flush=True,
    )
    size_mib = output_path.stat().st_size / (1024**2)
    bitrate_mbps = (
        output_path.stat().st_size
        * 8
        / max(actual_duration, 1e-6)
        / 1_000_000
    )
    print(
        f"[size] {size_mib:.2f} MiB, average_bitrate={bitrate_mbps:.2f} Mb/s",
        flush=True,
    )
    print(f"[output] {output_path}", flush=True)


def install(base_module: object, fast_module: object) -> None:
    global base, fast
    base = base_module
    fast = fast_module
    base.process_video = process_video
