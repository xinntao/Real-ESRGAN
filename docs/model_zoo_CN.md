# :european_castle: 模型库

[English](model_zoo.md) **|** 简体中文

- [:European_castle: 模型库](#european_castle-模型库)
  - [对于一般图片](#对于一般图片)
  - [对于动漫图片](#对于动漫图片)
  - [对于动漫视频](#对于动漫视频)

---

## 对于一般图片

| 模型                                                                                                                             |  规模  | 描述                                         |
| ------------------------------------------------------------------------------------------------------------------------------- | :---- | :------------------------------------------- |
| [RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)                      | X4    | 一般图像的放大4倍模型                           |
| [RealESRGAN_x2plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)                      | X2    | 一般图像的放大2倍模型                           |
| [RealESRNet_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth)                      | X4    | 有MSE损失的X4模型（过度平滑效应）                 |
| [official ESRGAN_x4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth) | X4    | 官方ESRGAN模型                                |

以下模型为**判别器**，通常用于微调。

| 模型                                                                                                                    |      对应的模型      |
| ---------------------------------------------------------------------------------------------------------------------- | :------------------ |
| [RealESRGAN_x4plus_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth) | RealESRGAN_x4plus   |
| [RealESRGAN_x2plus_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x2plus_netD.pth) | RealESRGAN_x2plus   |

## 对于动漫图片

| 模型                                                                                                                            |  规模  |描述                                                         |
| ------------------------------------------------------------------------------------------------------------------------------ | :---- | :---------------------------------------------------------- |
| [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) | X4 | 针对动漫图像的优化；6个RRDB块（较小的网络）                          |

以下模型为**判别器**，通常用于微调。

| 模型                                                                                                                                      | 对应的模型                   |
| ---------------------------------------------------------------------------------------------------------------------------------------- | :------------------------- |
| [RealESRGAN_x4plus_anime_6B_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B_netD.pth) | RealESRGAN_x4plus_anime_6B |

## 对于动漫视频

| 模型                                                                                                                                |  规模  | 描述                           |
| ---------------------------------------------------------------------------------------------------------------------------------- | :---- | :----------------------------- |
| [RealESRGANv2-animevideo-xsx2](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth) | X2 | 具有XS尺寸的动漫视频模型              |
| [RealESRGANv2-animevideo-xsx4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth) | X4 | 具有XS尺寸的动漫视频模型              |

以下模型为**判别器**，通常用于微调。

TODO
