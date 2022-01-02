# 动漫视频模型

[English](anime_video_model.md) **|** 简体中文

:white_check_mark: 我们添加了为动漫视频优化的小模型 :-)

| 模型                                                                                                                                | 规模  | 描述                    |
| ---------------------------------------------------------------------------------------------------------------------------------- | :---- | :-------------------- |
| [RealESRGANv2-animevideo-xsx2](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth) |  X2   | 具有XS尺寸的动漫视频模型 |
| [RealESRGANv2-animevideo-xsx4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth) |  X4   | 具有XS尺寸的动漫视频模型 |

- [动漫视频模型](#动漫视频模型)
  - [如何使用](#如何使用)
    - [PyTorch推理](#PyTorch推理)
    - [ncnn可执行文件](#ncnn可执行文件)
      - [第1步：使用ffmpeg从视频中提取帧](#第1步：使用ffmpeg从视频中提取帧)
      - [第2步：用Real-ESRGAN可执行文件进行推断](#第2步：用Real-ESRGAN可执行文件进行推断)
      - [第3步：将增强的帧合并到视频中](#第3步：将增强的帧合并成视频)
  - [更多演示](#更多演示)

---

以下是一些演示（最好在全屏模式下观看）。

https://user-images.githubusercontent.com/17445847/145706977-98bc64a4-af27-481c-8abe-c475e15db7ff.MP4

https://user-images.githubusercontent.com/17445847/145707055-6a4b79cb-3d9d-477f-8610-c6be43797133.MP4

https://user-images.githubusercontent.com/17445847/145783523-f4553729-9f03-44a8-a7cc-782aadf67b50.MP4

## 如何使用

### PyTorch推理

```bash
# 下载模型
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth -P experiments/pretrained_models
# 推理
python inference_realesrgan_video.py -i inputs/video/onepiece_demo.mp4 -n RealESRGANv2-animevideo-xsx2 -s 2 -v -a -half -suffix outx2
```

### ncnn可执行文件

#### 第1步：使用ffmpeg从视频中提取帧

```bash
ffmpeg -i onepiece_demo.mp4 -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 tmp_frames/frame%08d.png
```

- 记住要提前创建`tmp_frames`文件夹。

#### 第2步：用Real-ESRGAN可执行文件进行推断

1. 下载预先训练好的模型[Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/realesrgan-ncnn-vulkan-20211212-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/realesrgan-ncnn-vulkan-20211212-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/realesrgan-ncnn-vulkan-20211212-macos.zip) **适用于Intel/AMD/Nvidia GPU的可执行文件**。

1. 以Windows为例，运行。

    ```bash
    ./realesrgan-ncnn-vulkan.exe -i tmp_frames -o out_frames -n RealESRGANv2-animevideo-xsx2 -s 2 -f jpg
    ```

    - 记住要提前创建`out_frames`文件夹。

#### 第3步：将增强的帧合并成视频

1. 首先通过以下方法获得输入视频的帧数

    ```bash
    ffmpeg -i onepiece_demo.mp4
    ```

    ```控制台
    使用方法。
    -i 输入视频路径
    ```

    你将得到类似于以下截图的输出。

    <p align="center">
        <img src="https://user-images.githubusercontent.com/17445847/145710145-c4f3accf-b82f-4307-9f20-3803a2c73f57.png">
    </p>

2. 合并框架

    ```bash
    ffmpeg -r 23.98 -i out_frames/frame%08d.jpg -c:v libx264 -r 23.98 -pix_fmt yuv420p output.mp4
    ```

    ```控制台
    使用方法。
    -i 输入视频路径
    -c:v 视频编码器（通常我们使用libx264）
    -r fps，记得修改它以满足你的需要
    -pix_fmt 视频中的像素格式
    ```

    如果你还想从输入视频中复制音频，请运行。

    ```bash
    ffmpeg -r 23.98 -i out_frames/frame%08d.jpg -i onepiece_demo.mp4 -map 0:v:0 -map 1:a:0 -c:a copy -c:v libx264 -r 23.98 -pix_fmt yuv420p output_w_audio.mp4
    ```

    ```控制台
    使用方法。
    -i 输入视频路径，这里我们使用两个输入流
    -c:v 视频编码器(通常我们使用libx264)
    -r fps，记得修改它以满足你的需要
    -pix_fmt 视频中的像素格式
    ```

## 更多演示

- 为One Piece输入视频。

    https://user-images.githubusercontent.com/17445847/145706822-0e83d9c4-78ef-40ee-b2a4-d8b8c3692d17.mp4

- 输出视频：One Piece

    https://user-images.githubusercontent.com/17445847/145706827-384108c0-78f6-4aa7-9621-99d1aaf65682.mp4

**更多的比较**

https://user-images.githubusercontent.com/17445847/145707458-04a5e9b9-2edd-4d1f-b400-380a72e5f5e6.MP4
