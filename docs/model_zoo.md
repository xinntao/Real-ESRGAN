# :european_castle: Model Zoo

- [:european_castle: Model Zoo](#european_castle-model-zoo)
  - [For General Images](#for-general-images)
  - [For Anime Images](#for-anime-images)
  - [For Anime Videos](#for-anime-videos)

---

## For General Images

| Models                                                                                                                          | Scale | Description                                  |
| ------------------------------------------------------------------------------------------------------------------------------- | :---- | :------------------------------------------- |
| [RealESRGAN_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)                      | X4    | X4 model for general images                  |
| [RealESRGAN_x2plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth)                      | X2    | X2 model for general images                  |
| [RealESRNet_x4plus](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth)                      | X4    | X4 model with MSE loss (over-smooth effects) |
| [official ESRGAN_x4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth) | X4    | official ESRGAN model                        |

The following models are **discriminators**, which are usually used for fine-tuning.

| Models                                                                                                                 | Corresponding model |
| ---------------------------------------------------------------------------------------------------------------------- | :------------------ |
| [RealESRGAN_x4plus_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth) | RealESRGAN_x4plus   |
| [RealESRGAN_x2plus_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x2plus_netD.pth) | RealESRGAN_x2plus   |

## For Anime Images

| Models                                                                                                                         | Scale | Description                                                 |
| ------------------------------------------------------------------------------------------------------------------------------ | :---- | :---------------------------------------------------------- |
| [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) | X4    | Optimized for anime images; 6 RRDB blocks (smaller network) |

The following models are **discriminators**, which are usually used for fine-tuning.

| Models                                                                                                                                   | Corresponding model        |
| ---------------------------------------------------------------------------------------------------------------------------------------- | :------------------------- |
| [RealESRGAN_x4plus_anime_6B_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B_netD.pth) | RealESRGAN_x4plus_anime_6B |

## For Anime Videos

| Models                                                                                                                             | Scale | Description                    |
| ---------------------------------------------------------------------------------------------------------------------------------- | :---- | :----------------------------- |
| [RealESRGANv2-animevideo-xsx2](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth) | X2    | Anime video model with XS size |
| [RealESRGANv2-animevideo-xsx4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth) | X4    | Anime video model with XS size |

The following models are **discriminators**, which are usually used for fine-tuning.

TODO
