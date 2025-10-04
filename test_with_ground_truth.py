#!/usr/bin/env python3
"""Test trained model by downsampling a HR image and upscaling it back."""

import argparse
import cv2
import numpy as np
import torch
from basicsr.archs.swinir_arch import SwinIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='High-resolution input image path')
    parser.add_argument('-o', '--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--scale', type=int, default=4, help='Downscale/upscale factor')
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Read original HR image
    print(f'Reading HR image: {args.input}')
    hr_img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if hr_img is None:
        print(f'Error: Could not read image from {args.input}')
        return

    print(f'Original HR image shape: {hr_img.shape}')
    
    # Create downsampled LR version
    h, w = hr_img.shape[:2]
    lr_h, lr_w = h // args.scale, w // args.scale
    lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    print(f'Downsampled LR image shape: {lr_img.shape}')
    
    # Save LR image
    basename = os.path.splitext(os.path.basename(args.input))[0]
    lr_path = os.path.join(args.output_dir, f'{basename}_LR.png')
    cv2.imwrite(lr_path, lr_img)
    print(f'✅ Saved LR image to: {lr_path}')

    # Create SwinIR model with same config as training
    print('\nLoading trained model...')
    model = SwinIR(
        upscale=args.scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],  # 9 layers - CORRECTED
        embed_dim=240,  # CORRECTED from 180
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],  # CORRECTED from 6
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    # Load checkpoint
    print(f'Loading checkpoint from: {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device)

    # Handle different checkpoint formats
    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
        print('Loaded EMA params')
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
        print('Loaded params')
    else:
        model.load_state_dict(checkpoint, strict=True)
        print('Loaded direct state dict')

    model.eval()

    # Prepare LR image for model
    print('\nUpscaling LR image...')
    img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # Pad image to be divisible by window_size (8)
    h_lr, w_lr = img.shape[:2]
    window_size = 8
    mod_pad_h = (window_size - h_lr % window_size) % window_size
    mod_pad_w = (window_size - w_lr % window_size) % window_size

    if mod_pad_h > 0 or mod_pad_w > 0:
        img = np.pad(img, ((0, mod_pad_h), (0, mod_pad_w), (0, 0)), mode='reflect')
        print(f'Padded LR image from {h_lr}x{w_lr} to {img.shape[0]}x{img.shape[1]}')

    # Convert to tensor
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    print('Running inference...')
    with torch.no_grad():
        output = model(img_tensor)

    # Post-process
    output = output.squeeze(0).cpu().numpy()
    output = np.transpose(output, (1, 2, 0))

    # Crop output to remove padding
    output_h = h_lr * args.scale
    output_w = w_lr * args.scale
    output = output[:output_h, :output_w, :]

    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    print(f'Upscaled SR image shape: {output.shape}')

    # Save SR image
    sr_path = os.path.join(args.output_dir, f'{basename}_SR.png')
    cv2.imwrite(sr_path, output)
    print(f'✅ Saved SR image to: {sr_path}')

    # Crop original HR to match SR dimensions (in case of rounding)
    hr_cropped = hr_img[:output_h, :output_w, :]
    
    # Save cropped HR (ground truth)
    hr_cropped_path = os.path.join(args.output_dir, f'{basename}_HR_cropped.png')
    cv2.imwrite(hr_cropped_path, hr_cropped)
    print(f'✅ Saved HR ground truth to: {hr_cropped_path}')

    # Create bicubic upscale for comparison
    lr_bicubic = cv2.resize(lr_img, (output_w, output_h), interpolation=cv2.INTER_CUBIC)
    bicubic_path = os.path.join(args.output_dir, f'{basename}_Bicubic.png')
    cv2.imwrite(bicubic_path, lr_bicubic)
    print(f'✅ Saved bicubic upscale to: {bicubic_path}')

    # Create comparison image: LR (upscaled) | Bicubic | SR (Model) | HR (Ground Truth)
    print('\nCreating comparison image...')
    lr_display = cv2.resize(lr_img, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    
    # Stack horizontally with labels
    comparison = np.hstack([lr_display, lr_bicubic, output, hr_cropped])
    
    # Add text labels
    label_height = 40
    labeled_comparison = np.zeros((comparison.shape[0] + label_height, comparison.shape[1], 3), dtype=np.uint8)
    labeled_comparison[label_height:, :, :] = comparison
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ['LR (x4)', 'Bicubic', 'SR (Model)', 'HR (Ground Truth)']
    section_width = output_w
    
    for i, label in enumerate(labels):
        x_pos = i * section_width + 10
        cv2.putText(labeled_comparison, label, (x_pos, 28), font, 0.8, (255, 255, 255), 2)
    
    comparison_path = os.path.join(args.output_dir, f'{basename}_comparison_full.png')
    cv2.imwrite(comparison_path, labeled_comparison)
    print(f'✅ Saved full comparison to: {comparison_path}')
    
    # Calculate PSNR and SSIM
    print('\nCalculating metrics...')
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    hr_rgb = cv2.cvtColor(hr_cropped, cv2.COLOR_BGR2RGB)
    sr_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    bicubic_rgb = cv2.cvtColor(lr_bicubic, cv2.COLOR_BGR2RGB)
    
    psnr_bicubic = psnr(hr_rgb, bicubic_rgb)
    psnr_sr = psnr(hr_rgb, sr_rgb)
    
    ssim_bicubic = ssim(hr_rgb, bicubic_rgb, channel_axis=2, data_range=255)
    ssim_sr = ssim(hr_rgb, sr_rgb, channel_axis=2, data_range=255)
    
    print(f'\n📊 Metrics (vs Ground Truth):')
    print(f'  Bicubic - PSNR: {psnr_bicubic:.2f} dB, SSIM: {ssim_bicubic:.4f}')
    print(f'  SR Model - PSNR: {psnr_sr:.2f} dB, SSIM: {ssim_sr:.4f}')
    print(f'  Improvement - PSNR: +{psnr_sr - psnr_bicubic:.2f} dB, SSIM: +{ssim_sr - ssim_bicubic:.4f}')


if __name__ == '__main__':
    main()

