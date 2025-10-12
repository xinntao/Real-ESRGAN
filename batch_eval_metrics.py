#!/usr/bin/env python3

import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

try:
    import lpips
    import torch
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_same_size(gt: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if gt.shape[:2] == test.shape[:2]:
        return gt, test, 'matched'
    # resize gt to test size
    resized_gt = cv2.resize(gt, (test.shape[1], test.shape[0]), interpolation=cv2.INTER_CUBIC)
    return resized_gt, test, 'resized-gt-to-test'


def compute_lpips(img_bgr_a: np.ndarray, img_bgr_b: np.ndarray, loss_fn) -> float | None:
    if not _HAS_LPIPS:
        return None
    # convert to RGB and normalize to [-1, 1]
    img_rgb_a = cv2.cvtColor(img_bgr_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb_b = cv2.cvtColor(img_bgr_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ten_a = torch.from_numpy(img_rgb_a).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    ten_b = torch.from_numpy(img_rgb_b).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        dist = loss_fn(ten_a, ten_b).item()
    return float(dist)


def find_matching_files(gt_dir, restored_dir):
    """Find matching ground truth and restored image pairs"""
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.*')))
    gt_files = [f for f in gt_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    pairs = []
    for gt_path in gt_files:
        gt_name = Path(gt_path).stem

        # Try to find matching restored file
        for ext in ['.png', '.jpg', '.jpeg']:
            restored_path = os.path.join(restored_dir, gt_name + ext)
            if os.path.exists(restored_path):
                pairs.append((gt_path, restored_path))
                break

    return pairs


def create_metrics_plots(psnr_rgb_list, ssim_rgb_list, psnr_y_list, ssim_y_list,
                        lpips_list, processing_times, output_dir='results'):
    """Create bar charts for all metrics with averages"""

    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Image Quality Metrics Evaluation', fontsize=16, fontweight='bold')

    # Colors for each metric
    colors = ['blue', 'green', 'cyan', 'magenta', 'purple', 'orange']

    # Data and labels
    metrics_data = [
        (psnr_rgb_list, 'PSNR RGB Values', 'PSNR RGB (dB)', 'blue', axes[0, 0]),
        (ssim_rgb_list, 'SSIM RGB Values', 'SSIM RGB', 'green', axes[0, 1]),
        (psnr_y_list, 'PSNR Y Values', 'PSNR Y (dB)', 'cyan', axes[1, 0]),
        (ssim_y_list, 'SSIM Y Values', 'SSIM Y', 'magenta', axes[1, 1]),
        (lpips_list, 'LPIPS Values', 'LPIPS (lower is better)', 'purple', axes[2, 0]),
        (processing_times, 'Processing Times', 'Processing Time (s)', 'orange', axes[2, 1])
    ]

    for metric_list, title, ylabel, color, ax in metrics_data:
        if not metric_list:  # Skip if no data
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Create bar chart
        x_pos = range(len(metric_list))
        bars = ax.bar(x_pos, metric_list, color=color, alpha=0.7)

        # Add average line
        avg_value = np.mean(metric_list)
        ax.axhline(y=avg_value, color='red', linestyle='--', linewidth=2,
                  label=f'Average: {avg_value:.4f}')

        # Customize plot
        ax.set_title(title)
        ax.set_xlabel('Images')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set y-axis limits with some padding
        y_min, y_max = min(metric_list), max(metric_list)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()

    # Save plot
    plot_path = f"{output_dir}/metrics_evaluation_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Metrics plots saved: {plot_path}")
    return plot_path


def main():
    parser = argparse.ArgumentParser(description='Compute PSNR, SSIM, LPIPS for multiple images.')
    parser.add_argument('--gt-dir', required=True, help='Directory with ground-truth images')
    parser.add_argument('--restored-dir', required=True, help='Directory with restored images')
    parser.add_argument('--crop-border', type=int, default=0, help='Pixels to crop from each border before metrics')
    parser.add_argument('--y', action='store_true', help='Compute PSNR/SSIM on Y (luma) channel as well')
    parser.add_argument('--max-images', type=int, default=100, help='Maximum number of images to evaluate')
    args = parser.parse_args()

    # Find matching image pairs
    image_pairs = find_matching_files(args.gt_dir, args.restored_dir)

    if not image_pairs:
        print("❌ No matching image pairs found!")
        print(f"   GT dir: {args.gt_dir}")
        print(f"   Restored dir: {args.restored_dir}")
        return

    # Limit to max_images
    if len(image_pairs) > args.max_images:
        image_pairs = image_pairs[:args.max_images]

    print(f"Found {len(image_pairs)} matching image pairs")
    print("Evaluating...")
    print()

    # Initialize LPIPS model once if available
    lpips_loss_fn = None
    if _HAS_LPIPS:
        lpips_loss_fn = lpips.LPIPS(net='alex')

    # Store results for plotting
    psnr_rgb_list = []
    ssim_rgb_list = []
    psnr_y_list = []
    ssim_y_list = []
    lpips_list = []
    processing_times = []

    # Process each image pair - output same format as eval_metrics.py
    for idx, (gt_path, restored_path) in enumerate(image_pairs, 1):
        start_time = time.time()
        try:
            gt = load_image_bgr(gt_path)
            restored = load_image_bgr(restored_path)

            gt_aligned, restored_aligned, align_info = ensure_same_size(gt, restored)

            # PSNR / SSIM (RGB)
            psnr_rgb = calculate_psnr(gt_aligned * 1.0, restored_aligned * 1.0,
                                     crop_border=args.crop_border, input_order='HWC')
            ssim_rgb = calculate_ssim(gt_aligned * 1.0, restored_aligned * 1.0,
                                     crop_border=args.crop_border, input_order='HWC')

            # Store for plotting
            psnr_rgb_list.append(psnr_rgb)
            ssim_rgb_list.append(ssim_rgb)

            # PSNR / SSIM (Y)
            psnr_y = ssim_y = None
            if args.y:
                psnr_y = calculate_psnr(gt_aligned * 1.0, restored_aligned * 1.0,
                                       crop_border=args.crop_border, input_order='HWC',
                                       test_y_channel=True)
                ssim_y = calculate_ssim(gt_aligned * 1.0, restored_aligned * 1.0,
                                       crop_border=args.crop_border, input_order='HWC',
                                       test_y_channel=True)

                # Store for plotting
                psnr_y_list.append(psnr_y)
                ssim_y_list.append(ssim_y)

            # LPIPS
            lpips_val = compute_lpips(gt_aligned, restored_aligned, lpips_loss_fn)
            if lpips_val is not None:
                lpips_list.append(lpips_val)

            # Calculate processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            # Print results in same format as eval_metrics.py
            print(f"[{idx:3d}/{len(image_pairs)}] {Path(gt_path).name}")
            print('alignment:', align_info)
            print(f'gt_shape: {gt.shape}, restored_shape: {restored.shape}, used_shapes: {gt_aligned.shape} vs {restored_aligned.shape}')
            print(f'PSNR_RGB: {psnr_rgb:.6f} dB')
            print(f'SSIM_RGB: {ssim_rgb:.6f}')
            if psnr_y is not None and ssim_y is not None:
                print(f'PSNR_Y:   {psnr_y:.6f} dB')
                print(f'SSIM_Y:   {ssim_y:.6f}')
            if lpips_val is None:
                print('LPIPS: not available (install with: pip install lpips)')
            else:
                print(f'LPIPS (alex): {lpips_val:.6f}')
            print()

        except Exception as e:
            print(f"⚠️  Error processing {Path(gt_path).name}: {e}")
            continue

    print(f"✅ Completed evaluation of {len(image_pairs)} images")

    # Generate plots
    print("\n📊 Generating metrics plots...")
    plot_path = create_metrics_plots(
        psnr_rgb_list, ssim_rgb_list, psnr_y_list, ssim_y_list,
        lpips_list, processing_times, output_dir='results'
    )

    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print(f"   PSNR_RGB: {np.mean(psnr_rgb_list):.4f} ± {np.std(psnr_rgb_list):.4f} dB")
    print(f"   SSIM_RGB: {np.mean(ssim_rgb_list):.4f} ± {np.std(ssim_rgb_list):.4f}")
    if psnr_y_list:
        print(f"   PSNR_Y:   {np.mean(psnr_y_list):.4f} ± {np.std(psnr_y_list):.4f} dB")
        print(f"   SSIM_Y:   {np.mean(ssim_y_list):.4f} ± {np.std(ssim_y_list):.4f}")
    if lpips_list:
        print(f"   LPIPS:    {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}")
    print(f"   Avg Time: {np.mean(processing_times):.2f} ± {np.std(processing_times):.2f} seconds")


if __name__ == '__main__':
    main()
