# general settings
name: train_RealESRNetx2plus_1000k_B12G4
model_type: RealESRNetModel
scale: 2
num_gpu: auto  # auto: can infer from your visible devices automatically. official: 4 GPUs
manual_seed: 0

# ----------------- options for synthesizing training data in RealESRNetModel ----------------- #
gt_usm: True  # USM the ground-truth

# the first degradation process
resize_prob: [0.2, 0.7, 0.1]  # up, down, keep
resize_range: [0.15, 1.5]
gaussian_noise_prob: 0.5
noise_range: [1, 30]
poisson_scale_range: [0.05, 3]
gray_noise_prob: 0.4
jpeg_range: [30, 95]

# the second degradation process
second_blur_prob: 0.8
resize_prob2: [0.3, 0.4, 0.3]  # up, down, keep
resize_range2: [0.3, 1.2]
gaussian_noise_prob2: 0.5
noise_range2: [1, 25]
poisson_scale_range2: [0.05, 2.5]
gray_noise_prob2: 0.4
jpeg_range2: [30, 95]

gt_size: 256
queue_size: 180

# dataset and data loader settings
datasets:
  train:
    name: DF2K+OST
    type: RealESRGANDataset
    dataroot_gt: datasets/DF2K
    meta_info: datasets/DF2K/meta_info/meta_info_DF2Kmultiscale+OST_sub.txt
    io_backend:
      type: disk

    blur_kernel_size: 21
    kernel_list: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob: 0.1
    blur_sigma: [0.2, 3]
    betag_range: [0.5, 4]
    betap_range: [1, 2]

    blur_kernel_size2: 21
    kernel_list2: ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob2: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sinc_prob2: 0.1
    blur_sigma2: [0.2, 1.5]
    betag_range2: [0.5, 4]
    betap_range2: [1, 2]

    final_sinc_prob: 0.8

    gt_size: 256
    use_hflip: True
    use_rot: False

    # data loader
    use_shuffle: true
    num_worker_per_gpu: 5
    batch_size_per_gpu: 12
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  # Uncomment these for validation
  # val:
  #   name: validation
  #   type: PairedImageDataset
  #   dataroot_gt: path_to_gt
  #   dataroot_lq: path_to_lq
  #   io_backend:
  #     type: disk

# network structures
network_g:
  type: RRDBNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  num_block: 23
  num_grow_ch: 32
  scale: 2

# path
path:
  pretrain_network_g: experiments/pretrained_models/RealESRGAN_x4plus.pth
  param_key_g: params_ema
  strict_load_g: False
  resume_state: ~

# training settings
train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [1000000]
    gamma: 0.5

  total_iter: 1000000
  warmup_iter: -1  # no warm up

  # losses
  pixel_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

# Uncomment these for validation
# validation settings
# val:
#   val_freq: !!float 5e3
#   save_img: True

#   metrics:
#     psnr: # metric name
#       type: calculate_psnr
#       crop_border: 4
#       test_y_channel: false

# logging settings
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# dist training settings
dist_params:
  backend: nccl
  port: 29500
