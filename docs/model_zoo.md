# :european_castle: Model Zoo

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
| [realesr-general-x4v3](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) | X4 (can also be used for X1, X2, X3) | A tiny small model (consume much fewer GPU memory and time); not too strong deblur and denoise capacity |

The following models are **discriminators**, which are usually used for fine-tuning.

| Models                                                                                                                 | Corresponding model |
| ---------------------------------------------------------------------------------------------------------------------- | :------------------ |
| [RealESRGAN_x4plus_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth) | RealESRGAN_x4plus   |
| [RealESRGAN_x2plus_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x2plus_netD.pth) | RealESRGAN_x2plus   |

## For Anime Images / Illustrations

| Models                                                                                                                         | Scale | Description                                                 |
| ------------------------------------------------------------------------------------------------------------------------------ | :---- | :---------------------------------------------------------- |
| [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) | X4    | Optimized for anime images; 6 RRDB blocks (smaller network) |

The following models are **discriminators**, which are usually used for fine-tuning.

| Models                                                                                                                                   | Corresponding model        |
| ---------------------------------------------------------------------------------------------------------------------------------------- | :------------------------- |
| [RealESRGAN_x4plus_anime_6B_netD](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B_netD.pth) | RealESRGAN_x4plus_anime_6B |

## For Animation Videos

| Models                                                                                                                             | Scale | Description                    |
| ---------------------------------------------------------------------------------------------------------------------------------- | :---- | :----------------------------- |
| [realesr-animevideov3](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth) | X4<sup>1</sup>    | Anime video model with XS size |

Note: <br>
<sup>1</sup> This model can also be used for X1, X2, X3.

The following models are **discriminators**, which are usually used for fine-tuning.

TODO
