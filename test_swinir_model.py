#!/usr/bin/env python3
"""Quick inference test for our trained SwinIR model."""

import argparse

import cv2
import numpy as np
import torch
from basicsr.archs.swinir_arch import SwinIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image path')
    parser.add_argument('-o', '--output', type=str, default='test_output.png', help='Output image path')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling scale')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create SwinIR model with same config as training (CORRECTED architecture)
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

    # Read input image
    print(f'Reading input image: {args.input}')
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f'Error: Could not read image from {args.input}')
        return

    print(f'Input image shape: {img.shape}')

    # Convert to RGB and normalize to [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # Pad image to be divisible by window_size (8)
    h, w = img.shape[:2]
    window_size = 8
    mod_pad_h = (window_size - h % window_size) % window_size
    mod_pad_w = (window_size - w % window_size) % window_size

    if mod_pad_h > 0 or mod_pad_w > 0:
        img = np.pad(img, ((0, mod_pad_h), (0, mod_pad_w), (0, 0)), mode='reflect')
        print(f'Padded image from {h}x{w} to {img.shape[0]}x{img.shape[1]}')

    # Convert to tensor and add batch dimension
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
    output_h = h * args.scale
    output_w = w * args.scale
    output = output[:output_h, :output_w, :]

    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    print(f'Output image shape: {output.shape}')

    # Save output
    cv2.imwrite(args.output, output)
    print(f'✅ Saved upscaled image to: {args.output}')

    # Also save a side-by-side comparison
    h_out, w_out = output.shape[:2]
    h_in, w_in = img.shape[:2]

    # Resize input to match output for comparison
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (w_out, h_out), interpolation=cv2.INTER_CUBIC)

    # Create side-by-side comparison
    comparison = np.hstack([img_resized, output])
    comparison_path = args.output.replace('.png', '_comparison.png')
    cv2.imwrite(comparison_path, comparison)
    print(f'✅ Saved comparison to: {comparison_path}')

if __name__ == '__main__':
    main()
