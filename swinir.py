#!/usr/bin/env python3
"""SwinIR v1.1: high-fidelity 1080p->4K video SR on one CUDA GPU.

Uses the official SwinIR-M Classical SR x2 DF2K checkpoint. v1.1 intentionally
has no temporal filtering: measure the framewise model first, then add temporal
logic only if needed. Video probing, color metadata, encoding and audio mux reuse
the validated v4.4 I/O layer; decoding is upgraded to RGB48LE before SwinIR.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, math, subprocess, time, urllib.request
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from tqdm import tqdm

import realesrgan as legacy

base = legacy.base
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "third_party" / "swinir"
WEIGHTS = ROOT / "weights"
NETWORK = CACHE / "network_swinir.py"
WEIGHT = WEIGHTS / "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
NETWORK_URL = "https://raw.githubusercontent.com/JingyunLiang/SwinIR/v0.0/models/network_swinir.py"
WEIGHT_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
WINDOW, SCALE = 8, 2


def download(url: str, path: Path, label: str) -> Path:
    if path.is_file() and path.stat().st_size:
        print(f"[{label}] cached: {path}", flush=True); return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"[{label}] downloading {url}", flush=True)
    try: urllib.request.urlretrieve(url, tmp); tmp.replace(path)
    finally:
        if tmp.exists(): tmp.unlink()
    return path


def load_official_module() -> ModuleType:
    download(NETWORK_URL, NETWORK, "swinir-source")
    spec = importlib.util.spec_from_file_location("official_swinir", NETWORK)
    if spec is None or spec.loader is None: raise ImportError(f"Cannot load {NETWORK}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def build_model(device: torch.device) -> torch.nn.Module:
    net = load_official_module()
    model = net.SwinIR(upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.,
        depths=[6]*6, embed_dim=180, num_heads=[6]*6, mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv")
    path = download(WEIGHT_URL, WEIGHT, "swinir-model")
    try: ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError: ckpt = torch.load(path, map_location="cpu")
    state = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    print(f"[model] SwinIR-M classical DF2K x2, params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model


def starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length: return [0]
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride)); last = length - tile
    if values[-1] != last: values.append(last)
    return values


def pad_window(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]; ph = (-h) % WINDOW; pw = (-w) % WINDOW
    if ph or pw:
        mode = "reflect" if h > ph and w > pw else "replicate"
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode=mode)
    return x, h, w


def ramp(n: int, width: int, left: bool, right: bool) -> np.ndarray:
    a = np.ones(n, np.float32); width = min(width, n//2)
    if width <= 0: return a
    x = np.linspace(0, math.pi/2, width, endpoint=False, dtype=np.float32)
    c = np.maximum(np.sin(x)**2, 1e-3)
    if left: a[:width] *= c
    if right: a[-width:] *= c[::-1]
    return a


class Processor:
    def __init__(self, model, device, fp16: bool, tile: int, overlap: int):
        if tile < 256 or tile % WINDOW: raise ValueError("tile must be >=256 and divisible by 8")
        if not 0 <= overlap < tile: raise ValueError("overlap must satisfy 0 <= overlap < tile")
        self.model, self.device, self.fp16, self.tile, self.overlap = model, device, fp16, tile, overlap
        self.last = 0.0

    def patch(self, image: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).unsqueeze(0).to(self.device)
        x, h, w = pad_window(x); t = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16): y = self.model(x)
        torch.cuda.synchronize(self.device); self.last = time.monotonic() - t
        y = y[..., :h*SCALE, :w*SCALE].squeeze(0).permute(1,2,0).float().clamp_(0,1).cpu().numpy()
        return np.ascontiguousarray(y, dtype=np.float32)

    def frame(self, image: np.ndarray) -> np.ndarray:
        h,w,_ = image.shape; ys=starts(h,self.tile,self.overlap); xs=starts(w,self.tile,self.overlap)
        out=np.zeros((h*SCALE,w*SCALE,3),np.float32); den=np.zeros((h*SCALE,w*SCALE),np.float32); ro=self.overlap*SCALE
        for y0 in ys:
            y1=min(h,y0+self.tile)
            for x0 in xs:
                x1=min(w,x0+self.tile); sr=self.patch(image[y0:y1,x0:x1]); oh,ow=sr.shape[:2]
                wy=ramp(oh,ro,y0>0,y1<h); wx=ramp(ow,ro,x0>0,x1<w); wt=wy[:,None]*wx[None,:]
                oy0,oy1=y0*SCALE,y1*SCALE; ox0,ox1=x0*SCALE,x1*SCALE
                out[oy0:oy1,ox0:ox1]+=sr*wt[...,None]; den[oy0:oy1,ox0:ox1]+=wt
        if den.min() <= 0: raise RuntimeError("tile stitch coverage failure")
        out /= den[...,None]; return np.ascontiguousarray(np.clip(out,0,1),dtype=np.float32)


def auto_tile(model, device, fp16, frame, maximum, overlap) -> int:
    h,w,_=frame.shape; best=None
    for raw in (1024,896,768,640,512,384,256):
        tile=min(raw,maximum,max(h,w)); tile-=tile%WINDOW
        if tile<256 or tile<=overlap: continue
        p=Processor(model,device,fp16,tile,overlap); sample=frame[:min(tile,h),:min(tile,w)]
        try:
            p.patch(sample); count=len(starts(h,tile,overlap))*len(starts(w,tile,overlap)); est=p.last*count
            print(f"[auto-tile] tile={tile}, tiles={count}, sample={p.last:.3f}s, estimate={est:.3f}s/frame",flush=True)
            if best is None or est<best[0]: best=(est,tile)
        except torch.cuda.OutOfMemoryError:
            print(f"[auto-tile] tile={tile}: OOM",flush=True); torch.cuda.empty_cache()
    if best is None: raise RuntimeError("No SwinIR tile candidate fits GPU memory")
    print(f"[auto-tile] selected={best[1]}, estimated={best[0]:.3f}s/frame",flush=True); return best[1]


class RGB48Reader:
    def __init__(self,path,ffmpeg,w,h,fps,start,duration):
        self.n=w*h*6; self.w=w; self.h=h
        cmd=[ffmpeg,"-hide_banner","-loglevel","error"]
        if start>0: cmd += ["-ss",f"{start:.6f}"]
        cmd += ["-i",str(path),"-t",f"{duration:.6f}","-vf",f"fps={fps}","-an","-f","rawvideo","-pix_fmt","rgb48le","pipe:1"]
        self.p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def read(self):
        raw=self.p.stdout.read(self.n)
        if not raw: return None
        if len(raw)!=self.n: raise RuntimeError(f"partial RGB48 frame {len(raw)}/{self.n}")
        u=np.frombuffer(raw,dtype="<u2").reshape(self.h,self.w,3); return np.ascontiguousarray(u.astype(np.float32)/65535.),raw
    def close(self):
        self.p.stdout.close(); err=self.p.stderr.read(); self.p.stderr.close(); code=self.p.wait()
        if code: raise RuntimeError(f"ffmpeg decode failed ({code}):\n{err.decode(errors='replace')}")


def validate(a):
    if a.gpu<0: raise ValueError("gpu must be non-negative")
    for name in ("tile_size","max_tile_size"):
        v=getattr(a,name)
        if v<256 or v%WINDOW: raise ValueError(f"--{name.replace('_','-')} must be >=256 and divisible by 8")
    if not 0<=a.tile_overlap<a.tile_size: raise ValueError("tile overlap must be below tile size")
    if a.video_codec=="libsvtav1" and not 0<=a.crf<=63: raise ValueError("SVT-AV1 CRF must be 0..63")
    if not 0<=a.svtav1_preset<=13: raise ValueError("SVT-AV1 preset must be 0..13")


def process(a):
    base.require_binary(a.ffmpeg_bin); base.require_binary(a.ffprobe_bin); base.require_encoder(a.ffmpeg_bin,a.video_codec)
    if not torch.cuda.is_available(): raise RuntimeError("SwinIR v1.1 requires CUDA")
    if a.gpu>=torch.cuda.device_count(): raise ValueError(f"GPU {a.gpu} unavailable")
    src=Path(a.input).expanduser().resolve(); dst=Path(a.output).expanduser().resolve()
    if not src.is_file(): raise FileNotFoundError(src)
    if src==dst: raise ValueError("input and output must differ")
    dst.parent.mkdir(parents=True,exist_ok=True); tmp=dst.with_name(dst.stem+".video_only.tmp.mp4")
    info=base.probe_video(src,a.ffprobe_bin); color=base.resolve_color_spec(info,a.color_policy,a.hdr_policy)
    source=base.parse_rate(f"{info.fps_num}/{info.fps_den}"); output=base.parse_output_rate(a.fps,source); infer=min(source,output)
    start,duration,expected=base.resolve_range(info,a.start_time,a.test_seconds,infer)
    fpsin=f"{infer.numerator}/{infer.denominator}"; fpsout=f"{output.numerator}/{output.denominator}"
    pix=base.resolve_output_pix_fmt(a.ffmpeg_bin,a.video_codec,a.output_pix_fmt)
    legacy._SVTAV1_PRESET=a.svtav1_preset
    device=torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(device); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    model=build_model(device); reader=RGB48Reader(src,a.ffmpeg_bin,info.width,info.height,fpsin,start,duration)
    writer=base.RawVideoWriter(tmp,a.ffmpeg_bin,info.width*SCALE,info.height*SCALE,fpsin,fpsout,a.video_codec,a.crf,"medium",18,"p7",a.gpu,pix,base.color_filter_chain(color,pix),color)
    print(f"[input] {info.width}x{info.height} -> {info.width*SCALE}x{info.height*SCALE}, inference_fps={float(infer):.6f}",flush=True)
    print(f"[precision] decode=rgb48le, model={'fp16 autocast' if a.fp16 else 'fp32'}, stitch=fp32, encode={pix}",flush=True)
    n=dups=0; last_hash=None; last_out=None; tile=None; t0=time.monotonic(); bar=tqdm(total=expected,desc="SwinIR",unit="frame",dynamic_ncols=True,mininterval=1.)
    clean=False
    try:
        while True:
            item=reader.read()
            if item is None: break
            frame,raw=item; digest=hashlib.blake2b(raw,digest_size=8).digest() if a.reuse_identical_frames else None
            if digest is not None and digest==last_hash and last_out is not None: out=last_out; dups+=1
            else:
                if tile is None:
                    tile=auto_tile(model,device,a.fp16,frame,a.max_tile_size,a.tile_overlap) if a.auto_tile else a.tile_size
                    print(f"[tiles] tile={tile}, overlap={a.tile_overlap}, window=8, blend=cosine",flush=True)
                out=Processor(model,device,a.fp16,tile,a.tile_overlap).frame(frame); last_hash=digest; last_out=out
            writer.write(out); n+=1; bar.update(1); bar.set_postfix(fps=f"{n/max(time.monotonic()-t0,1e-6):.3f}",dup=dups,refresh=False)
        clean=n>0
    finally:
        bar.close()
        try: reader.close()
        except Exception:
            if clean: raise
        try: writer.close()
        except Exception:
            if clean: raise
    if not clean: raise RuntimeError("No frames processed")
    actual=n/float(infer); base.mux_audio(tmp,src,dst,a.ffmpeg_bin,start,actual,info.has_audio,a.audio_codec,a.audio_bitrate)
    if tmp.exists(): tmp.unlink()
    elapsed=time.monotonic()-t0; print(f"[done] frames={n}, duplicates={dups}, elapsed={elapsed:.1f}s, average={n/max(elapsed,1e-6):.3f} frame/s",flush=True); print(f"[output] {dst}",flush=True)


def parser():
    p=argparse.ArgumentParser(description="SwinIR v1.1 high-fidelity 2x video SR")
    p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--fps",default="source"); p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--fp16",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--auto-tile",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--max-tile-size",type=int,default=1024); p.add_argument("--tile-size",type=int,default=512); p.add_argument("--tile-overlap",type=int,default=32)
    p.add_argument("--reuse-identical-frames",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--video-codec",choices=("libsvtav1","libx265"),default="libsvtav1"); p.add_argument("--output-pix-fmt",choices=("auto","yuv420p","yuv420p10le","p010le"),default="auto")
    p.add_argument("--crf",type=int,default=18); p.add_argument("--svtav1-preset",type=int,default=6); p.add_argument("--color-policy",choices=("preserve","bt709"),default="preserve"); p.add_argument("--hdr-policy",choices=("reject","passthrough"),default="reject")
    p.add_argument("--audio-codec",choices=("copy","aac"),default="copy"); p.add_argument("--audio-bitrate",default="192k"); p.add_argument("--start-time",type=float,default=0.); p.add_argument("--test-seconds",type=float,default=0.)
    p.add_argument("--ffmpeg-bin",default="ffmpeg"); p.add_argument("--ffprobe-bin",default="ffprobe"); return p


def main():
    a=parser().parse_args(); validate(a); process(a)

if __name__=="__main__": main()
