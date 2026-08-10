"""Pipelined full-frame execution with shared-memory frame transport."""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import sys
import threading
import time
import traceback
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm as _tqdm

from . import runtime as base

_TIMEOUT = 300.0


class PersistentTqdm(_tqdm):
    """Keep tqdm for interactive runs and emit durable newline progress logs."""

    def __init__(self, *args, persistent_interval: float = 10.0, **kwargs):
        self._persistent_interval = max(1.0, float(persistent_interval))
        self._persistent_last_time = 0.0
        self._persistent_last_bucket = -1
        super().__init__(*args, **kwargs)

    def _persistent_log(self, postfix: str = "") -> None:
        if self.n <= 0:
            return
        now = time.monotonic()
        total = int(self.total or 0)
        percent = (100.0 * self.n / total) if total > 0 else 0.0
        bucket = int(percent // 10.0) if total > 0 else 0
        first = self.n == 1
        timed = now - self._persistent_last_time >= self._persistent_interval
        stepped = total > 0 and bucket > self._persistent_last_bucket
        final = total > 0 and self.n >= total
        if not (first or timed or stepped or final):
            return

        elapsed = max(time.time() - self.start_t, 1e-6)
        if "ETA" in postfix:
            speed_eta = postfix
        else:
            rate = self.n / elapsed
            remaining = max(0, total - self.n)
            eta = remaining / rate if rate > 1e-9 else 0.0
            speed_eta = f"{rate:.3f} frame/s | ETA {self.format_interval(eta)}"
        total_text = str(total) if total > 0 else "?"
        print(
            f"[progress] {self.n}/{total_text} | {percent:.1f}% | "
            f"{speed_eta} | elapsed {self.format_interval(elapsed)}",
            flush=True,
        )
        self._persistent_last_time = now
        self._persistent_last_bucket = bucket

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        super().set_postfix(ordered_dict=ordered_dict, refresh=refresh, **kwargs)
        self._persistent_log(str(self.postfix or ""))

    def set_postfix_str(self, s="", refresh=True):
        super().set_postfix_str(s=s, refresh=refresh)
        self._persistent_log(str(s))


def _worker(
    worker_id: int,
    gpu_id: Optional[int],
    input_queue: mp.Queue,
    result_queue: mp.Queue,
    output_slot: mp.Semaphore,
    config_dict: Dict[str, object],
    input_name: str,
    output_name: str,
    input_shape: Tuple[int, int, int],
    output_shape: Tuple[int, int, int],
    dtype_str: str,
) -> None:
    input_shm = output_shm = None
    try:
        config = base.WorkerConfig(**config_dict)  # type: ignore[arg-type]
        if gpu_id is None:
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        model, _ = base.load_worker_model(config, device)
        dtype = np.dtype(dtype_str)
        input_shm = shared_memory.SharedMemory(name=input_name)
        output_shm = shared_memory.SharedMemory(name=output_name)
        input_view = np.ndarray(input_shape, dtype=dtype, buffer=input_shm.buf)
        output_view = np.ndarray(output_shape, dtype=dtype, buffer=output_shm.buf)
        result_queue.put(("ready", worker_id, str(device)))
        while True:
            frame_id = input_queue.get()
            if frame_id is None:
                break
            started = time.monotonic()
            result = base.infer_frame(model, input_view, device)
            infer_seconds = time.monotonic() - started
            output_slot.acquire()
            np.copyto(output_view, result, casting="no")
            result_queue.put(("result", worker_id, int(frame_id), infer_seconds))
            del result
    except Exception as error:
        result_queue.put(("error", worker_id, repr(error), traceback.format_exc()))
    finally:
        if input_shm is not None:
            input_shm.close()
        if output_shm is not None:
            output_shm.close()


class SharedWorkers:
    def __init__(
        self,
        gpu_ids: Sequence[Optional[int]],
        config: base.WorkerConfig,
        input_shape: Tuple[int, int, int],
        output_shape: Tuple[int, int, int],
        dtype: np.dtype,
    ) -> None:
        self.context = mp.get_context("spawn")
        self.count = len(gpu_ids)
        self.dtype = np.dtype(dtype)
        self.input_shape = input_shape
        self.result_queue = self.context.Queue()
        self.input_queues = [self.context.Queue(maxsize=1) for _ in gpu_ids]
        self.output_slots = [self.context.Semaphore(1) for _ in gpu_ids]
        self.input_shms: List[shared_memory.SharedMemory] = []
        self.output_shms: List[shared_memory.SharedMemory] = []
        self.input_views: List[np.ndarray] = []
        self.output_views: List[np.ndarray] = []
        self.processes = []
        self.closed = False
        in_bytes = int(np.prod(input_shape, dtype=np.int64)) * self.dtype.itemsize
        out_bytes = int(np.prod(output_shape, dtype=np.int64)) * self.dtype.itemsize
        try:
            for _ in gpu_ids:
                a = shared_memory.SharedMemory(create=True, size=in_bytes)
                b = shared_memory.SharedMemory(create=True, size=out_bytes)
                self.input_shms.append(a)
                self.output_shms.append(b)
                self.input_views.append(np.ndarray(input_shape, dtype=self.dtype, buffer=a.buf))
                self.output_views.append(np.ndarray(output_shape, dtype=self.dtype, buffer=b.buf))
            for worker_id, gpu_id in enumerate(gpu_ids):
                p = self.context.Process(
                    target=_worker,
                    args=(
                        worker_id, gpu_id, self.input_queues[worker_id], self.result_queue,
                        self.output_slots[worker_id], asdict(config),
                        self.input_shms[worker_id].name, self.output_shms[worker_id].name,
                        input_shape, output_shape, self.dtype.str,
                    ),
                    daemon=True,
                )
                p.start()
                self.processes.append(p)
            self._wait_ready()
        except Exception:
            self.close()
            raise

    @property
    def memory_mib(self) -> float:
        return sum(x.size for x in self.input_shms + self.output_shms) / 2**20

    def _wait_ready(self) -> None:
        ready = 0
        deadline = time.monotonic() + _TIMEOUT
        while ready < self.count:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("Timed out loading pipelined workers.")
            try:
                msg = self.result_queue.get(timeout=left)
            except queue.Empty as error:
                raise TimeoutError("Timed out loading pipelined workers.") from error
            if msg[0] == "error":
                raise RuntimeError(f"Worker {msg[1]} failed: {msg[2]}\n{msg[3]}")
            if msg[0] == "ready":
                ready += 1

    def submit(self, worker_id: int, frame_id: int, frame: np.ndarray) -> None:
        if frame.shape != self.input_shape or frame.dtype != self.dtype:
            raise ValueError(f"Unexpected frame {frame.shape}/{frame.dtype}")
        np.copyto(self.input_views[worker_id], frame, casting="no")
        self.input_queues[worker_id].put(int(frame_id))

    def result(self, block: bool = True) -> tuple:
        if block:
            try:
                return self.result_queue.get(timeout=_TIMEOUT)
            except queue.Empty as error:
                raise TimeoutError("Timed out waiting for inference.") from error
        return self.result_queue.get_nowait()

    def output(self, worker_id: int) -> np.ndarray:
        return self.output_views[worker_id]

    def release(self, worker_id: int) -> None:
        self.output_slots[worker_id].release()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for slot in self.output_slots:
            try:
                slot.release()
            except Exception:
                pass
        for q in self.input_queues:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        for q in self.input_queues:
            try:
                q.close()
            except Exception:
                pass
        try:
            self.result_queue.close()
        except Exception:
            pass
        for shm in self.input_shms + self.output_shms:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    )


class OutputPump:
    def __init__(self, workers: SharedWorkers, writer, width: int, height: int, progress, started: float):
        self.workers = workers
        self.writer = writer
        self.width = width
        self.height = height
        self.progress = progress
        self.started = started
        self.queue: queue.Queue[Optional[Tuple[int, int]]] = queue.Queue(maxsize=max(2, workers.count * 2))
        self.processed = 0
        self.resize_seconds = 0.0
        self.write_seconds = 0.0
        self.error: Optional[BaseException] = None
        self.traceback_text = ""
        self.thread = threading.Thread(target=self._run, name="realesrgan-output", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break
                _frame_id, worker_id = item
                try:
                    t = time.monotonic()
                    frame = _resize(self.workers.output(worker_id), self.width, self.height)
                    self.resize_seconds += time.monotonic() - t
                    t = time.monotonic()
                    self.writer.write(frame)
                    self.write_seconds += time.monotonic() - t
                    self.processed += 1
                    self.progress.update(1)
                    elapsed = max(time.monotonic() - self.started, 1e-6)
                    self.progress.set_postfix(fps=f"{self.processed / elapsed:.3f}", refresh=False)
                finally:
                    self.workers.release(worker_id)
                    self.queue.task_done()
        except Exception as error:
            self.error = error
            self.traceback_text = traceback.format_exc()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Output pipeline failed: {self.error!r}\n{self.traceback_text}") from self.error

    def put(self, frame_id: int, worker_id: int) -> None:
        self.check()
        self.queue.put((frame_id, worker_id))
        self.check()

    def finish(self) -> None:
        self.check()
        self.queue.put(None)
        self.thread.join()
        self.check()

    def stop(self) -> None:
        if self.thread.is_alive():
            try:
                self.queue.put(None, timeout=30)
            except queue.Full:
                return
            self.thread.join(timeout=30)


def _result(msg: tuple) -> Tuple[int, int, float]:
    if msg[0] == "error":
        raise RuntimeError(f"Worker {msg[1]} failed: {msg[2]}\n{msg[3]}")
    if msg[0] != "result":
        raise RuntimeError(f"Unexpected worker message: {msg[0]}")
    return int(msg[1]), int(msg[2]), float(msg[3])


def process_video(args) -> None:
    if base._require_encoder is None or base._writer_type is None:
        raise RuntimeError("Encoding backend is not configured. Run through root inference.py.")
    base.require_binary(args.ffmpeg_bin)
    base.require_binary(args.ffprobe_bin)
    base._require_encoder(args.ffmpeg_bin, args.video_codec)

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_path.with_name(output_path.stem + ".video_only.tmp.mp4")

    info = base.probe_video(input_path, args.ffprobe_bin)
    source_rate = Fraction(info.fps_num, info.fps_den)
    output_rate = base.parse_output_rate(args.fps, source_rate)
    inference_rate = min(source_rate, output_rate)
    output_fps = float(output_rate)
    output_rate_text = f"{output_rate.numerator}/{output_rate.denominator}"
    inference_fps = float(inference_rate)
    inference_rate_text = f"{inference_rate.numerator}/{inference_rate.denominator}"
    in_w, in_h = info.width, info.height
    out_w, out_h = int(round(in_w * args.scale)), int(round(in_h * args.scale))
    if out_w % 2 or out_h % 2:
        raise ValueError(f"4:2:0 output needs even dimensions, got {out_w}x{out_h}.")
    start, duration, expected = base.resolve_range(info, args.start_time, args.test_seconds, inference_rate)
    expected_output = max(1, int(round(duration * output_fps)))
    end = start + duration
    gpu_ids = base.parse_gpu_ids(args.gpu_ids)
    native_scale = base._model_native_scale(args.model)
    config = base.WorkerConfig(args.model, base.resolve_model_paths(args))
    dtype = np.dtype("<u2") if info.bit_depth == 10 else np.dtype(np.uint8)
    input_shape = (in_h, in_w, 3)
    native_shape = (in_h * native_scale, in_w * native_scale, 3)

    mode = "test" if args.test_seconds > 0 else "full/selected range"
    print("=== Real-ESRGAN ===", flush=True)
    print(f"Input   : {input_path.name} | {in_w}x{in_h} | {info.fps:.3f} fps | {info.bit_depth}-bit ({info.pix_fmt})", flush=True)
    print(f"Output  : {out_w}x{out_h} | {output_fps:.3f} fps | {info.bit_depth}-bit ({base._output_pixel_format(args.video_codec, info.bit_depth)}) | {args.video_codec}", flush=True)
    print(f"Range   : {mode} | {base.format_seconds(start)} -> {base.format_seconds(end)} | {duration:.3f}s | {expected} inference / {expected_output} output frames", flush=True)
    scale_text = f"native={native_scale}x" if float(args.scale) == float(native_scale) else f"native={native_scale}x -> final={args.scale:g}x | resample=full-frame Lanczos4"
    print(f"Model   : {args.model} | {scale_text}", flush=True)
    print(f"GPU     : {base._device_text(gpu_ids)}", flush=True)
    print(f"Mode    : pipelined full-frame | parallel_frames={len(gpu_ids)} | IPC=shared-memory", flush=True)
    print(flush=True)

    reader = writer = workers = pump = progress = None
    started = time.monotonic()
    next_frame = next_output = 0
    eof = False
    active: set[int] = set()
    pending: Dict[int, int] = {}
    gpu_work = decode_time = model_time = flush_time = audio_time = 0.0
    clean = False

    def submit(worker_id: int) -> bool:
        nonlocal eof, next_frame, decode_time
        if eof:
            return False
        t = time.monotonic()
        frame = reader.read()
        decode_time += time.monotonic() - t
        if frame is None:
            eof = True
            return False
        workers.submit(worker_id, next_frame, frame)
        active.add(worker_id)
        next_frame += 1
        return True

    try:
        reader = base.RawVideoReader(input_path, args.ffmpeg_bin, in_w, in_h, inference_rate_text, start, duration, info.bit_depth)
        writer = base._writer_type(temp_video, args.ffmpeg_bin, out_w, out_h, inference_rate_text, output_rate_text, args.video_codec, args.crf, args.preset, args.cq, args.nvenc_preset, args.encode_gpu)
        t = time.monotonic()
        workers = SharedWorkers(gpu_ids, config, input_shape, native_shape, dtype)
        model_time = time.monotonic() - t
        print(f"Pipeline: shared_memory={workers.memory_mib:.0f} MiB | GPU inference overlaps resize/encode", flush=True)
        print(flush=True)
        progress = PersistentTqdm(total=expected, desc="Real-ESRGAN", unit="frame", dynamic_ncols=True, mininterval=1.0, file=sys.stdout)
        pump = OutputPump(workers, writer, out_w, out_h, progress, started)

        for worker_id in range(workers.count):
            if not submit(worker_id):
                break
        while active:
            pump.check()
            worker_id, frame_id, seconds = _result(workers.result())
            gpu_work += seconds
            active.discard(worker_id)
            pending[frame_id] = worker_id
            submit(worker_id)
            while True:
                try:
                    worker_id, frame_id, seconds = _result(workers.result(False))
                except queue.Empty:
                    break
                gpu_work += seconds
                active.discard(worker_id)
                pending[frame_id] = worker_id
                submit(worker_id)
            while next_output in pending:
                worker_id = pending.pop(next_output)
                pump.put(next_output, worker_id)
                next_output += 1
        if pending:
            raise RuntimeError(f"Pipeline ended with {len(pending)} out-of-order frame(s).")
        pump.finish()
        reader.close(); reader = None
        t = time.monotonic(); writer.close(); flush_time = time.monotonic() - t; writer = None
        clean = True
    finally:
        if pump is not None:
            try: pump.stop()
            except Exception: pass
        if progress is not None:
            progress.close()
        if reader is not None:
            try: reader.close()
            except Exception: pass
        if writer is not None:
            try: writer.close()
            except Exception: pass
        if workers is not None:
            workers.close()

    processed = pump.processed if pump is not None else 0
    if not clean or processed == 0:
        raise RuntimeError("No complete video was encoded.")
    actual_duration = processed / inference_fps
    t = time.monotonic()
    base.mux_audio(temp_video, input_path, output_path, args.ffmpeg_bin, start, actual_duration, info.has_audio, args.audio_codec, args.audio_bitrate)
    audio_time = time.monotonic() - t
    if temp_video.exists():
        temp_video.unlink()

    elapsed = time.monotonic() - started
    size_mib = output_path.stat().st_size / 2**20
    bitrate = output_path.stat().st_size * 8 / max(actual_duration, 1e-6) / 1_000_000
    fps = processed / max(elapsed, 1e-6)
    gpu_avg = gpu_work / max(processed, 1)
    print("\n=== Completed ===", flush=True)
    print(f"Frames  : {processed} | {base.format_seconds(start)} -> {base.format_seconds(start + actual_duration)} | duration={actual_duration:.3f}s", flush=True)
    print(f"Speed   : {fps:.3f} frame/s | processing={elapsed:.1f}s", flush=True)
    print(f"Timing  : model={model_time:.1f}s | decode={decode_time:.1f}s | gpu_avg={gpu_avg:.3f}s/frame | resize={pump.resize_seconds:.1f}s | write={pump.write_seconds:.1f}s | flush={flush_time:.1f}s | audio={audio_time:.1f}s", flush=True)
    print(f"File    : {size_mib:.2f} MiB | {bitrate:.2f} Mb/s", flush=True)
    print(f"Output  : {output_path}", flush=True)
