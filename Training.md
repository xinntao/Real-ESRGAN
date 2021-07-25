# :computer: How to Train Real-ESRGAN

The training codes have been released. <br>
Note that the codes have a lot of refactoring. So there may be some bugs/performance drops. Welcome to report issues and I will also retrain the models.

## Overview

The training has been divided into two stages. These two stages have the same data synthesis process and training pipeline, except for the loss functions. Specifically,

1. We first train Real-ESRNet with L1 loss from the pre-trained model ESRGAN.
1. We then use the trained Real-ESRNet model as an initialization of the generator, and train the Real-ESRGAN with a combination of L1 loss, perceptual loss and GAN loss.

## Dataset Preparation

We use DF2K (DIV2K and Flickr2K) + OST datasets for our training. Only HR images are required. <br>
You can download from :

1. DIV2K: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
2. Flickr2K: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
3. OST: https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip

For the DF2K dataset, we use a multi-scale strategy, *i.e.*, we downsample HR images to obtain several Ground-Truth images with different scales.

We then crop DF2K images into sub-images for faster IO and processing.

You need to prepare a txt file containing the image paths. The following are some examples in `meta_info_DF2Kmultiscale+OST_sub.txt` (As different users may have different sub-images partitions, this file is not suitable for your purpose and you need to prepare your own txt file):

```txt
DF2K_HR_sub/000001_s001.png
DF2K_HR_sub/000001_s002.png
DF2K_HR_sub/000001_s003.png
...
```

## Train Real-ESRNet

1. Download pre-trained model [ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth) into `experiments/pretrained_models`.
    ```bash
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth -P experiments/pretrained_models
    ```
1. Modify the content in the option file `options/train_realesrnet_x4plus.yml` accordingly:
    ```yml
    train:
        name: DF2K+OST
        type: RealESRGANDataset
        dataroot_gt: datasets/DF2K  # modify to the root path of your folder
        meta_info: data/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt  # modify to your own generate meta info txt
        io_backend:
            type: disk
    ```
1. If you want to perform validation during training, uncomment those lines and modify accordingly:
    ```yml
      # Uncomment these for validation
      # val:
      #   name: validation
      #   type: PairedImageDataset
      #   dataroot_gt: path_to_gt
      #   dataroot_lq: path_to_lq
      #   io_backend:
      #     type: disk

    ...

      # Uncomment these for validation
      # validation settings
      # val:
      #   val_freq: !!float 5e3
      #   save_img: True

      #   metrics:
      #     psnr: # metric name, can be arbitrary
      #       type: calculate_psnr
      #       crop_border: 4
      #       test_y_channel: false
    ```
1. Before the formal training, you may run in the `--debug` mode to see whether everything is OK. We use four GPUs for training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --debug
    ```
1. The formal training. We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_realesrnet_x4plus.yml --launcher pytorch --auto_resume
    ```

## Train Real-ESRGAN

1. After the training of Real-ESRNet, you now have the file `experiments/train_RealESRNetx4plus_1000k_B12G4_fromESRGAN/model/net_g_1000000.pth`. If you need to specify the pre-trained path to other files, modify the `pretrain_network_g` value in the option file `train_realesrgan_x4plus.yml`.
1. Modify the option file `train_realesrgan_x4plus.yml` accordingly. Most modifications are similar to those listed above.
1. Before the formal training, you may run in the `--debug` mode to see whether everything is OK. We use four GPUs for training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_realesrgan_x4plus.yml --launcher pytorch --debug
    ```
1. The formal training. We use four GPUs for training. We use the `--auto_resume` argument to automatically resume the training if necessary.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 train.py -opt options/train_realesrgan_x4plus.yml --launcher pytorch --auto_resume
    ```
