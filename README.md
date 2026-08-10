# Real-ESRGAN 动画视频推理

仓库只保留一套以画质优先的 full-frame 视频推理流程，并按职责拆分为推理、流水线调度与编码。

## 版本

- v4.3 代码基线：`90208548939f7b59ae08ff7db7f338b41b703e22`

## 结构

- `inference.py`：统一命令行入口。
- `realesrgan.ipynb`：Kaggle Notebook。
- `inference/runtime.py`：视频探测/解码、8/10-bit 精度保持、模型加载与单帧 full-frame 推理等基础能力。
- `inference/pipeline.py`：多 GPU full-frame 流水线、共享内存帧传输、输出顺序控制以及 Lanczos/编码并行。
- `inference/models/`：`SRVGGNetCompact` 等推理模型结构。
- `inference/weights/`：随仓库提供的推理权重。
- `encode/runtime.py`：编码后端选择和 CLI 参数。
- `encode/hevc.py`：H.264/HEVC 编码实现。
- `encode/av1.py`：AV1 编码实现。
- `requirements.txt`：运行依赖。

## 推理策略

推理固定使用 full-frame，不包含 tile、batch 自动调参或 tile 融合逻辑。

- CUDA：自动使用 FP16 + channels-last。
- CPU：自动使用 FP32。
- 输入分辨率固定保持源视频分辨率。
- 模型始终输出原生倍率。
- 当目标倍率小于模型原生倍率时，完整帧推理结束后仅做一次全帧 Lanczos4 缩放。
- 8-bit 源输出 8-bit；10-bit 源使用高精度 RGB 推理并输出 10-bit。

## 流水线

当前公开入口使用 `inference/pipeline.py`：

- 输入帧和模型原生倍率输出都通过共享内存传递，`multiprocessing.Queue` 只传递 frame ID / worker ID 等小型元数据。
- 某张 GPU 完成一帧后会立即接收下一帧，不再等待其它 GPU、Lanczos 缩放和视频编码全部完成。
- 上一帧的全帧 Lanczos4 与 FFmpeg/NVENC 写入在独立输出线程执行，与后续 GPU 推理重叠。
- 输出仍按原始帧顺序写入视频。

## 编码

支持：

- CPU HEVC：`libx265`
- GPU HEVC：`hevc_nvenc`
- CPU AV1：`libsvtav1`，缺失时自动尝试 `libaom-av1`
- GPU AV1：`av1_nvenc`
- 兼容 H.264：`libx264` / `h264_nvenc`

具体 codec 参数、encoder 检测和视频 writer 都集中在 `encode/`。

## 使用

在 Kaggle 中打开根目录 `realesrgan.ipynb`，修改输入路径、输出倍率、GPU 和编码参数后按顺序运行。建议先使用 `TEST_SECONDS = 10` 验证显存、编码和音画同步，再改为 `0` 处理完整视频。

本仓库保留 Real-ESRGAN 的 BSD 3-Clause `LICENSE`。
