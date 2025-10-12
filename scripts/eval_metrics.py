import argparse

import cv2
import numpy as np

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

try:
    import lpips  # type: ignore
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


def compute_lpips(img_bgr_a: np.ndarray, img_bgr_b: np.ndarray) -> float | None:
    if not _HAS_LPIPS:
        return None
    # convert to RGB and normalize to [-1, 1]
    img_rgb_a = cv2.cvtColor(img_bgr_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_rgb_b = cv2.cvtColor(img_bgr_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ten_a = torch.from_numpy(img_rgb_a).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    ten_b = torch.from_numpy(img_rgb_b).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    loss_fn = lpips.LPIPS(net='alex')
    with torch.no_grad():
        dist = loss_fn(ten_a, ten_b).item()
    return float(dist)


def main():
    parser = argparse.ArgumentParser(description='Compute PSNR, SSIM, LPIPS between two images.')
    parser.add_argument('--gt', required=True, help='Path to ground-truth/original image')
    parser.add_argument('--restored', required=True, help='Path to restored/upscaled image')
    parser.add_argument('--crop-border', type=int, default=0, help='Pixels to crop from each border before metrics')
    parser.add_argument('--y', action='store_true', help='Compute PSNR/SSIM on Y (luma) channel as well')
    args = parser.parse_args()

    gt = load_image_bgr(args.gt)
    restored = load_image_bgr(args.restored)

    gt_aligned, restored_aligned, align_info = ensure_same_size(gt, restored)

    # PSNR / SSIM (RGB)
    psnr_rgb = calculate_psnr(gt_aligned * 1.0, restored_aligned * 1.0, crop_border=args.crop_border, input_order='HWC')
    ssim_rgb = calculate_ssim(gt_aligned * 1.0, restored_aligned * 1.0, crop_border=args.crop_border, input_order='HWC')

    # PSNR / SSIM (Y)
    psnr_y = ssim_y = None
    if args.y:
        psnr_y = calculate_psnr(gt_aligned * 1.0, restored_aligned * 1.0, crop_border=args.crop_border, input_order='HWC', test_y_channel=True)
        ssim_y = calculate_ssim(gt_aligned * 1.0, restored_aligned * 1.0, crop_border=args.crop_border, input_order='HWC', test_y_channel=True)

    # LPIPS
    lpips_val = compute_lpips(gt_aligned, restored_aligned)

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


if __name__ == '__main__':
    main()