# Real-ESRGAN

[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![Open issue](https://isitmaintained.com/badge/open/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/Real-ESRGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/pylint.yml)

1. [Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo) for Real-ESRGAN <a href="https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>.
2. [Portable Windows executable file](https://github.com/xinntao/Real-ESRGAN/releases). You can find more information [here](#Portable-executable-files).

Real-ESRGAN aims at developing **Practical Algorithms for General Image Restoration**.<br>
We extend the powerful ESRGAN to a practical restoration application (namely, Real-ESRGAN), which is trained with pure synthetic data.

:triangular_flag_on_post: The training codes have been released. A detailed guide can be found in [Training.md](Training.md).

### :book: Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

> [[Paper](https://arxiv.org/abs/2107.10833)] &emsp; [Project Page] &emsp; [Demo] <br>
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> Applied Research Center (ARC), Tencent PCG<br>
> Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

We have provided a pretrained model (*RealESRGAN_x4plus.pth*) with upsampling X4.<br>
**Note that RealESRGAN may still fail in some cases as the real-world degradations are really too complex.**<br>
Moreover, it **may not** perform well on **human faces, text**, *etc*, which will be optimized later.
<br>

Real-ESRGAN will be a long-term supported project (in my current plan :smiley:). It will be continuously updated
in my spare time.

Here is a TODO list in the near future:

- [ ] optimize for human faces
- [ ] optimize for texts
- [ ] optimize for animation images
- [ ] support more scales
- [ ] support controllable restoration strength

If you have any good ideas or demands, please open an issue/discussion to let me know. <br>
If you have some images that Real-ESRGAN could not well restored, please also open an issue/discussion. I will record it (but I cannot guarantee to resolve it:stuck_out_tongue:). If necessary, I will open a page to specially record these real-world cases that need to be solved, but the current technology is difficult to handle well.

---

### Portable executable files

You can download **Windows executable files** from https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN-ncnn-vulkan.zip

This executable file is **portable** and includes all the binaries and models required. No CUDA or PyTorch environment is needed.<br>

You can simply run the following command:

```bash
./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png
```

Note that it may introduce block inconsistency (and also generate slightly different results from the PyTorch implementation), because this executable file first crops the input image into several tiles, and then processes them separately, finally stitches together.

This executable file is based on the wonderful [Tencent/ncnn](https://github.com/Tencent/ncnn) and [realsr-ncnn-vulkan](https://github.com/nihui/realsr-ncnn-vulkan) by [nihui](https://github.com/nihui).

---

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

### Installation

1. Clone repo

    ```bash
    git clone https://github.com/xinntao/Real-ESRGAN.git
    cd Real-ESRGAN
    ```

1. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr
    pip install -r requirements.txt
    ```

## :zap: Quick Inference

Download pre-trained models: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

Download pretrained models:

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
```

Inference!

```bash
python inference_realesrgan.py --model_path experiments/pretrained_models/RealESRGAN_x4plus.pth --input inputs
```

Results are in the `results` folder

## :computer: Training

A detailed guide can be found in [Training.md](Training.md).

## BibTeX

    @Article{wang2021realesrgan,
        title={Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        author={Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        journal={arXiv:2107.10833},
        year={2021}
    }

## :e-mail: Contact

If you have any question, please email `xintao.wang@outlook.com` or `xintaowang@tencent.com`.
