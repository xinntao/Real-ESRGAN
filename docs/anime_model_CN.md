# 动漫模型

[English](anime_model.md) **|** 简体中文

:white_check_mark: 我们添加了[*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)，这是为**动漫**图像优化的，模型的尺寸小得多。

- [动漫模型](#动漫模型)
  - [如何使用](#如何使用)
    - [PyTorch推理](#PyTorch推理)
    - [ncnn可执行文件](#ncnn可执行文件)
  - [与waifu2x的比较](#与waifu2x的比较)
  - [与带滑动条的比较](#与带滑动条的比较])

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>

下面是一个带滑动条的视频比较。你可能需要使用全屏模式以获得更好的视觉效果，因为原始图像尺寸很大；否则，你可能会遇到混叠现象。

https://user-images.githubusercontent.com/17445847/131535127-613250d4-f754-4e20-9720-2f9608ad0675.mp4

## 如何使用

### PyTorch推理

预先训练的模型。[RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

```bash
# 下载模型
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P experiments/pretrained_models
# 推理
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```

### ncnn可执行文件

下载预先训练好的模型 [Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/realesrgan-ncnn-vulkan-20211212-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/realesrgan-ncnn-vulkan-20211212-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/realesrgan-ncnn-vulkan-20211212-macos.zip) **适用于Intel/AMD/Nvidia GPU的可执行文件**。

以Windows为例，运行。

```bash
./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesrgan-x4plus-anime
```

## 与waifu2x的比较

我们将Real-ESRGAN-anime与[waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)进行比较。我们使用waifu2x的`-n 2 -s 4`。

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_2.png">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_3.png">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_4.png">
</p>
<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_5.png">
</p>

## 与带滑动条的比较

以下是与带滑动条的视频比较。你可能需要使用全屏模式以获得更好的视觉效果，因为原始图像很大；否则，你可能会遇到混叠现象。

https://user-images.githubusercontent.com/17445847/131536647-a2fbf896-b495-4a9f-b1dd-ca7bbc90101a.mp4

https://user-images.githubusercontent.com/17445847/131536742-6d9d82b6-9765-4296-a15f-18f9aeaa5465.mp4
