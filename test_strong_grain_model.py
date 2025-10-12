#!/usr/bin/env python3

import os

import cv2
import numpy as np
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet


def enhance_with_tiling(model, img, tile_size=512, tile_pad=10, device='cuda'):
    """Enhance image using tiling to handle large images"""
    batch, channel, height, width = img.shape
    output_height = height * 2
    output_width = width * 2
    output_shape = (batch, channel, output_height, output_width)

    # Start with black image
    output = img.new_zeros(output_shape)
    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size

    # Loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # Extract tile
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            # Input tile area with padding
            input_start_x = max(0, ofs_x - tile_pad)
            input_end_x = min(width, ofs_x + tile_size + tile_pad)
            input_start_y = max(0, ofs_y - tile_pad)
            input_end_y = min(height, ofs_y + tile_size + tile_pad)

            # Input tile dimensions
            input_tile = img[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

            # Process tile
            with torch.no_grad():
                output_tile = model(input_tile)

            # Calculate output tile position
            output_start_x = input_start_x * 2
            output_end_x = input_end_x * 2
            output_start_y = input_start_y * 2
            output_end_y = input_end_y * 2

            # Calculate tile offsets
            output_start_x_tile = (ofs_x - input_start_x) * 2
            output_end_x_tile = output_start_x_tile + (min(width, ofs_x + tile_size) - ofs_x) * 2
            output_start_y_tile = (ofs_y - input_start_y) * 2
            output_end_y_tile = output_start_y_tile + (min(height, ofs_y + tile_size) - ofs_y) * 2

            # Put tile into output
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

    return output

def test_strong_grain_model():
    """Test the strong grain removal model on image 622"""

    print("🎞️ Testing Strong Grain Removal Model (5000 iterations)")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'grain_removal_strong_5k.pth'
    input_image = 'datasets/DIV2K_ISO_x2/HR/0001.png'  # Test with 0001
    output_dir = 'results/test_batch_strong_grain'

    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Input: {input_image}")
    print(f"📁 Model: {model_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🖥️  Device: {device}")

    # Load model
    print("\n📦 Loading model...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)

    checkpoint = torch.load(model_path, map_location=device)
    if 'params_ema' in checkpoint:
        model.load_state_dict(checkpoint['params_ema'], strict=True)
    elif 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)

    model.eval()
    model = model.to(device)
    print("✅ Model loaded successfully!")

    # Load image
    print("\n📸 Loading image...")
    img = cv2.imread(input_image, cv2.IMREAD_COLOR)
    if img is None:
        print(f"❌ Could not load image: {input_image}")
        return

    print(f"   Original size: {img.shape[1]}x{img.shape[0]}")

    # Get image number from path
    img_num = os.path.basename(input_image).split('.')[0]

    # Save original
    cv2.imwrite(f"{output_dir}/{img_num}_original.png", img)

    # Resize to half for x2 upscaling
    h, w = img.shape[:2]
    img_half = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    print(f"   Input size (half): {img_half.shape[1]}x{img_half.shape[0]}")

    # Save half-size input
    cv2.imwrite(f"{output_dir}/{img_num}_input_half.png", img_half)

    # Convert to tensor
    img_tensor = torch.from_numpy(np.transpose(img_half, (2, 0, 1))).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Process image
    print("\n🔄 Processing image...")
    with torch.no_grad():
        output = model(img_tensor)

    # Convert back to image
    output_img = output.squeeze(0).cpu().numpy()
    output_img = np.transpose(output_img, (1, 2, 0))
    output_img = (output_img * 255.0).clip(0, 255).astype(np.uint8)

    print(f"   Output size: {output_img.shape[1]}x{output_img.shape[0]}")

    # Save result
    output_path = f"{output_dir}/{img_num}_strong_grain_removed.png"
    cv2.imwrite(output_path, output_img)
    print(f"✅ Result saved: {output_path}")

    # Create comparison
    print("\n📊 Creating comparison...")

    # Resize for comparison
    h_out, w_out = output_img.shape[:2]
    target_h = 800
    target_w = int(w_out * target_h / h_out)

    img_resized = cv2.resize(img, (target_w, target_h))
    output_resized = cv2.resize(output_img, (target_w, target_h))

    # Side by side (no text labels)
    comparison = np.hstack([img_resized, output_resized])

    comparison_path = f"{output_dir}/{img_num}_comparison.png"
    cv2.imwrite(comparison_path, comparison)
    print(f"✅ Comparison saved: {comparison_path}")

    print("\n🎉 Processing complete!")
    print(f"📁 All results saved in: {output_dir}/")

if __name__ == '__main__':
    test_strong_grain_model()
