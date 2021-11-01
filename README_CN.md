# Real-ESRGAN

[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/realesrgan)](https://pypi.org/project/realesrgan/)
[![Open issue](https://img.shields.io/github/issues/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/Real-ESRGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/Real-ESRGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/publish-pip.yml)

[English](README.md) | 中文

1. Real-ESRGAN的[Colab Demo](https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing) <a href="https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>.
2. **支持Intel/AMD/Nvidia显卡**的绿色版exe文件： [Windows版](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-windows.zip) / [Linux版](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-ubuntu.zip) / [macOS版](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-macos.zip)，详情请移步[这里](#Portable-executable-files)。

Real-ESRGAN的目标是成为一个**通用实用的图像修复算法**
我们在强大的ESRGAN的基础上完全使用人工合成的数据来进行训练，以让其能被应用于图片修复的使用场景（顾名思义：Real-ESRGAN）。

:art: Real-ESRGAN需要，也很欢迎你的贡献，如新功能、模型、错误修复、建议、维护等等。详情可以查看[CONTRIBUTING.md](CONTRIBUTING.md)，所有的贡献者都会被列在[此处](CONTRIBUTING.md#Contributors)。

:question: 大部分的问题你都能在[FAQ.md](FAQ.md)中找到答案

:triangular_flag_on_post: **更新**
- :white_check_mark: 添加了 [*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)，对二次元照片进行了优化，并减少了model的大小。详情以及与[waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)的对比请查看[**anime_model.md**](docs/anime_model.md)
- :white_check_mark: 支持用户细调自己的数据组：[详情](Training.md#Finetune-Real-ESRGAN-on-your-own-dataset)
- :white_check_mark: 通过[GFPGAN](https://github.com/TencentARC/GFPGAN)**增强了人脸的画面**
- ：white_check_mar: 通过[Gradio](https://github.com/gradio-app/gradio)添加了UI并加入了[Huggingface Spaces](https://huggingface.co/spaces)（一个机器学习应用的在线平台）：[Gradio在线版](https://huggingface.co/spaces/akhaliq/Real-ESRGAN)。感谢[@AK391](https://github.com/AK391)
- :white_check_mark: 现在支持任意比例的缩放了：`--outscale`（可以用`LANCZOS4`来更进一步调整输出图像的尺寸）。添加了*RealESRGAN_x2plus.pth*模型
- :white_check_mark: 这个[推断程序](inference_realesrgan.py)支持: 1) **倾斜**相关选项; 2) 带**alpha通道**的图像; 3) **灰色**图像; 4) **16-bit**图像.
- :white_check_mark: 训练模型的代码已经提交了，具体的做法可以查看这里：[Training.md](Training.md)。

---

如果本项目帮到了你的照片或项目，麻烦给本项目一个star，或者推荐给你的朋友们，谢谢！:blush:<br/>
其他相关的项目：<br/>
:arrow_forward: [GFPGAN](https://github.com/TencentARC/GFPGAN): 一个对人脸进行还原的算法 <br>
:arrow_forward: [BasicSR](https://github.com/xinntao/BasicSR): 开源的图像和视频的还原工具<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): 一套提供与人脸相关的工具集<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): 基于PyQt5的照片查看器，方便查看以及比较 <br>

---

### :book: Real-ESRGAN: 使用人工合成数据训练的超分辨率算法

> [[论文](https://arxiv.org/abs/2107.10833)] &emsp; [项目] &emsp; [演示] <br>
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> 应用科学中心 (ARC), 腾讯PCG<br>
> 中国科学院 深圳先进技术研究院

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

我们提供了一套训练好的模型（*RealESRGAN_x4plus.pth*)，可以进行4倍的超分辨率。<br>
**现在Real-ESRGAN还是有几率失败的，因为现实生活的降分辨率还是比较难的。**<br>
而且本项目对**人脸以及文字之类**的效果还不是太好，但是我们会持续进行优化的。<br>

目前计划中Real-ESRGAN将会被长期支持，我会在空闲的时间中持续维护更新。

这些是未来计划好的几个新功能：


- [ ] 优化人脸
- [ ] 优化文字
- [x] 优化动画图像
- [ ] 支持更多的超分辨率比
- [ ] 更多可控制选项

如果你有好主意或需求，欢迎在issue或discussion中提出。<br/>
如果你有一些Real-ESRGAN中有问题的照片，你也可以在issue或者discussion中发上来。我会留意（但是不一定能解决:stuck_out_tongue:）。如果有必要的话，我还会专门开一页来记录那些有待解决的有问题的图像，但是以现在的技术来说还是有点难。

---

### 绿色版可执行文件

你可以下载**支持Intel/AMD/Nvidia显卡**的绿色版exe文件： [Windows版](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-windows.zip) / [Linux版](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-ubuntu.zip) / [macOS版](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/realesrgan-ncnn-vulkan-20210901-macos.zip)。

绿色版指的是这些exe你直接放U盘里拷走都没问题。因为里面已经有所需的文件和模型了。不需要CUDA或者PyTorch运行环境。

你可以通过这行指令来运行（Windows版本，更多信息请查看对应版本的README.md）：

```bash
./realesrgan-ncnn-vulkan.exe -i 输入图像.jpg -o 输出图像.png
```

我们提供这三种模型：
1. realesrgan-x4plus（默认）
2. reaesrnet-x4plus
3. realesrgan-x4plus-anime（针对动画图像，有更小的体积）

你可以通过`-n`参数来使用其他模型，例如`./realesrgan-ncnn-vulkan.exe -i 二次元图片.jpg -o 二刺螈图片.png -n realesrgan-x4plus-anime`

由于这些exe文件会把图像分成几个板块，然后来分别进行处理，再合成导出，输出的图像可能会有一点割裂感（而且可能跟PyTorch的输出不太一样）

这些exe文件均基于[Tencent/ncnn](https://github.com/Tencent/ncnn)以及[nihui](https://github.com/nihui)的[realsr-ncnn-vulkan](https://github.com/nihui/realsr-ncnn-vulkan)

---

## :wrench: 依赖以及安装

- Python >= 3.7 (推荐使用[Anaconda](https://www.anaconda.com/download/#linux)或[Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

#### 安装

1. 把项目克隆到本地

    ```bash
    git clone https://github.com/xinntao/Real-ESRGAN.git
    cd Real-ESRGAN
    ```

2. 安装各种依赖

    ```bash
    # 安装basicsr - https://github.com/xinntao/BasicSR
    # 我们使用BasicSR来训练以及推断
    pip install basicsr
    # facexlib和gfpgan是用来增强面部表现的
    pip install facexlib
    pip install gfpgan
    pip install -r requirements.txt
    python setup.py develop
    ```

## :zap: 快速上手

### 普通图片

下载我们训练好的模型: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
```

推断!

```bash
python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --input inputs --face_enhance
```

结果在`results`文件夹

### 动画图片

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>

训练好的模型: [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)<br>
有关[waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)的更多信息和对比在[**anime_model.md**](docs/anime_model.md)中

```bash
# 下载模型
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P experiments/pretrained_models
# 推断
python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth --input inputs
```

结果在`results`文件夹

## :european_castle: 各种模型

- [RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
- [RealESRGAN_x4plus_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth)
- [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)
- [RealESRGAN_x4plus_anime_6B_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B_netD.pth)
- [RealESRNet_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth)
- [RealESRGAN_x2plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)
- [RealESRGAN_x2plus_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x2plus_netD.pth)
- [official ESRGAN_x4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth)

## :computer: 训练并优化你的数据

这里有一份详细的指南：[Training.md](Training.md).

## BibTeX

    @Article{wang2021realesrgan,
        title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        journal={arXiv:2107.10833},
        year={2021}
    }

## :e-mail: 联系我们

如果你有任何问题，请通过`xintao.wang@outlook.com`或`xintaowang@tencent.com`联系我们。

