# Anime Model

:white_check_mark: We add [*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth), which is optimized for **anime** images with much smaller model size.

- [How to Use](#how-to-use)
  - [PyTorch Inference](#pytorch-inference)
  - [ncnn Executable File](#ncnn-executable-file)
- [Comparisons with waifu2x](#comparisons-with-waifu2x)
- [Comparisons with Sliding Bars](#comparisons-with-sliding-bars)

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>

The following is a video comparison with sliding bar. You may need to use the full-screen mode for better visual quality, as the original image is large; otherwise, you may encounter aliasing issue.

<https://user-images.githubusercontent.com/17445847/131535127-613250d4-f754-4e20-9720-2f9608ad0675.mp4>

## How to Use

### PyTorch Inference

Pre-trained models: [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)

```bash
# download model
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights
# inference
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```

### ncnn Executable File

Download the latest portable [Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip) **executable files for Intel/AMD/Nvidia GPU**.

Taking the Windows as example, run:

```bash
./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesrgan-x4plus-anime
```

## Comparisons with waifu2x

We compare Real-ESRGAN-anime with [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan). We use the `-n 2 -s 4` for waifu2x.

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

## Comparisons with Sliding Bars

The following are video comparisons with sliding bar. You may need to use the full-screen mode for better visual quality, as the original image is large; otherwise, you may encounter aliasing issue.

<https://user-images.githubusercontent.com/17445847/131536647-a2fbf896-b495-4a9f-b1dd-ca7bbc90101a.mp4>

<https://user-images.githubusercontent.com/17445847/131536742-6d9d82b6-9765-4296-a15f-18f9aeaa5465.mp4>
