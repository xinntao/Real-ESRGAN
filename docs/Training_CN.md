# :computer: 如何训练/微调 Real-ESRGAN

- [训练 Real-ESRGAN](#训练-real-esrgan)
  - [概述](#概述)
  - [准备数据集](#准备数据集)
  - [训练 Real-ESRNet 模型](#训练-real-esrnet-模型)
  - [训练 Real-ESRGAN 模型](#训练-real-esrgan-模型)
- [用自己的数据集微调 Real-ESRGAN](#用自己的数据集微调-real-esrgan)
  - [动态生成降级图像](#动态生成降级图像)
  - [使用已配对的数据](#使用已配对的数据)

[English](Training.md) **|** [简体中文](Training_CN.md)

## 训练 Real-ESRGAN

### 概述

训练分为两个步骤。除了 loss 函数外，这两个步骤拥有相同数据合成以及训练的一条龙流程。具体点说：

1. 首先使用 L1 loss 训练 Real-ESRNet 模型，其中 L1 loss 来自预先训练的 ESRGAN 模型。

2. 然后我们将 Real-ESRNet 模型作为生成器初始化，结合L1 loss、感知 loss、GAN loss 三者的参数对 Real-ESRGAN 进行训练。

### 准备数据集

我们使用 DF2K ( DIV2K 和 Flickr2K ) + OST 数据集进行训练。只需要HR图像！<br>
下面是网站链接:
1. DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
3. OST: https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip

以下是数据的准备步骤。

#### 第1步：【可选】生成多尺寸图片

针对 DF2K 数据集，我们使用多尺寸缩放策略，*换言之*，我们对 HR 图像进行下采样，就能获得多尺寸的标准参考（Ground-Truth）图像。 <br>
您可以使用这个 [scripts/generate_multiscale_DF2K.py](scripts/generate_multiscale_DF2K.py) 脚本快速生成多尺寸的图像。<br>
注意：如果您只想简单试试，那么可以跳过此步骤。

```bash
python scripts/generate_multiscale_DF2K.py --input datasets/DF2K/DF2K_HR --output datasets/DF2K/DF2K_multiscale
```

#### 第2步：【可选】裁切为子图像

我们可以将 DF2K 图像裁切为子图像，以加快 IO 和处理速度。<br>
如果你的 IO 够好或储存空间有限，那么此步骤是可选的。<br>

您可以使用脚本 [scripts/extract_subimages.py](scripts/extract_subimages.py)。这是使用示例:

```bash
 python scripts/extract_subimages.py --input datasets/DF2K/DF2K_multiscale --output datasets/DF2K/DF2K_multiscale_sub --crop_size 400 --step 200
```

#### 第3步：准备元信息 txt

您需要准备一个包含图像路径的 txt 文件。下面是 `meta_info_DF2Kmultiscale+OST_sub.txt` 中的部分展示（由于各个用户可能有截然不同的子图像划分，这个文件不适合你的需求，你得准备自己的 txt 文件)：

```txt
DF2K_HR_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
...
```

你可以使用该脚本 [scripts/generate_meta_info.py](scripts/generate_meta_info.py) 生成包含图像路径的 txt 文件。<br>
你还可以合并多个文件夹的图像路径到一个元信息（meta_info）txt。这是使用示例:

```bash
 python scripts/generate_meta_info.py --input datasets/DF2K/DF2K_HR, datasets/DF2K/DF2K_multiscale --root datasets/DF2K, datasets/DF2K --meta_info datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt
```

### 训练 Real-ESRNet 模型

1. 下载预先训练的模型 [ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth)，放到 `experiments/pretrained_models`目录下。
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -P experiments/pretrained_models
    ```
2. 相应地修改选项文件 `options/train_realesrnet_x4plus.yml` 中的内容：
    ```yml
    train:
        name: DF2K+OST
        type: RealESRGANDataset
        dataroot_gt: datasets/DF2K  # 修改为你的数据集文件夹根目录
        meta_info: realesrgan/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt  # 修改为你自己生成的元信息txt
        io_backend:
            type: disk
    ```
3. 如果你想在训练过程中执行验证，就取消注释这些内容并进行相应的修改：
    ```yml
      # 取消注释这些以进行验证
      # val:
      #   name: validation
      #   type: PairedImageDataset
      #   dataroot_gt: path_to_gt
      #   dataroot_lq: path_to_lq
      #   io_backend:
      #     type: disk

    ...

      # 取消注释这些以进行验证
      # 验证设置
      # val:
      #   val_freq: !!float 5e3
      #   save_img: True

      #   metrics:
      #     psnr: # 指标名称，可以是任意的
      #       type: calculate_psnr
      #       crop_border: 4
      #       test_y_channel: false
    ```
4. 正式训练之前，你可以用 `--debug` 模式检查是否正常运行。我们用了4个GPU进行训练：
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --debug
    ```

    用 **1个GPU** 训练的 debug 模式示例:
    ```bash
    python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --debug
    ```
5. 正式训练开始。我们用了4个GPU进行训练。还可以使用参数 `--auto_resume` 在必要时自动恢复训练。
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --auto_resume
    ```

    用 **1个GPU** 训练：
    ```bash
    python realesrgan/train.py -opt options/train_realesrnet_x4plus.yml --auto_resume
    ```

### 训练 Real-ESRGAN 模型

1. 训练 Real-ESRNet 模型后，您得到了这个 `experiments/train_RealESRNetx4plus_1000k_B12G4_fromESRGAN/model/net_g_1000000.pth` 文件。如果需要指定预训练路径到其他文件，请修改选项文件 `train_realesrgan_x4plus.yml` 中 `pretrain_network_g` 的值。
1. 修改选项文件 `train_realesrgan_x4plus.yml` 的内容。大多数修改与上节提到的类似。
1. 正式训练之前，你可以以 `--debug` 模式检查是否正常运行。我们使用了4个GPU进行训练：
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --launcher pytorch --debug
    ```

    用 **1个GPU** 训练的 debug 模式示例:
    ```bash
    python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --debug
    ```
1. 正式训练开始。我们使用4个GPU进行训练。还可以使用参数 `--auto_resume` 在必要时自动恢复训练。
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --launcher pytorch --auto_resume
    ```

    用 **1个GPU** 训练：
    ```bash
    python realesrgan/train.py -opt options/train_realesrgan_x4plus.yml --auto_resume
    ```

## 用自己的数据集微调 Real-ESRGAN

你可以用自己的数据集微调 Real-ESRGAN。一般地，微调（Fine-Tune）程序可以分为两种类型:

1. [动态生成降级图像](#动态生成降级图像)
2. [使用**已配对**的数据](#使用已配对的数据)

### 动态生成降级图像

只需要高分辨率图像。在训练过程中，使用 Real-ESRGAN 描述的降级模型生成低质量图像。

**1. 准备数据集**

完整信息请参见[本节](#准备数据集)。

**2. 下载预训练模型**

下载预先训练的模型到 `experiments/pretrained_models` 目录下。

- *RealESRGAN_x4plus.pth*:
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
    ```

- *RealESRGAN_x4plus_netD.pth*:
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
    ```

**3. 微调**

修改选项文件 [options/finetune_realesrgan_x4plus.yml](options/finetune_realesrgan_x4plus.yml) ，特别是 `datasets` 部分：

```yml
train:
    name: DF2K+OST
    type: RealESRGANDataset
    dataroot_gt: datasets/DF2K   # 修改为你的数据集文件夹根目录
    meta_info: realesrgan/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt  # 修改为你自己生成的元信息txt
    io_backend:
        type: disk
```

我们使用4个GPU进行训练。还可以使用参数 `--auto_resume` 在必要时自动恢复训练。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --launcher pytorch --auto_resume
```

用 **1个GPU** 训练：
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --auto_resume
```

### 使用已配对的数据

你还可以用自己已经配对的数据微调 RealESRGAN。这个过程更类似于微调 ESRGAN。

**1. 准备数据集**

假设你已经有两个文件夹（folder）:

- **gt folder**（标准参考，高分辨率图像）：*datasets/DF2K/DIV2K_train_HR_sub*
- **lq folder**（低质量，低分辨率图像）：*datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub*

然后，您可以使用脚本 [scripts/generate_meta_info_pairdata.py](scripts/generate_meta_info_pairdata.py) 生成元信息（meta_info）txt 文件。

```bash
python scripts/generate_meta_info_pairdata.py --input datasets/DF2K/DIV2K_train_HR_sub datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub --meta_info datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt
```

**2. 下载预训练模型**

下载预先训练的模型到 `experiments/pretrained_models` 目录下。

- *RealESRGAN_x4plus.pth*:
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
    ```

- *RealESRGAN_x4plus_netD.pth*:
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
    ```

**3. 微调**

修改选项文件 [options/finetune_realesrgan_x4plus_pairdata.yml](options/finetune_realesrgan_x4plus_pairdata.yml) ，特别是 `datasets` 部分：

```yml
train:
    name: DIV2K
    type: RealESRGANPairedDataset
    dataroot_gt: datasets/DF2K  # 修改为你的 gt folder 文件夹根目录
    dataroot_lq: datasets/DF2K  # 修改为你的 lq folder 文件夹根目录
    meta_info: datasets/DF2K/meta_info/meta_info_DIV2K_sub_pair.txt  # 修改为你自己生成的元信息txt
    io_backend:
        type: disk
```

我们使用4个GPU进行训练。还可以使用参数 `--auto_resume` 在必要时自动恢复训练。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --launcher pytorch --auto_resume
```

用 **1个GPU** 训练：
```bash
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus_pairdata.yml --auto_resume
```
