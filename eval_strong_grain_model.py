#!/usr/bin/env python3

import os
import subprocess
import sys
import argparse

def run_eval_metrics(gt_path, restored_path):
    """Run eval metrics and display results"""
    
    print(f"\n{'='*70}")
    print(f"📊 Evaluating Strong Grain Removal Model")
    print(f"{'='*70}")
    print(f"📁 Ground Truth: {gt_path}")
    print(f"📁 Restored: {restored_path}")
    print()
    
    cmd = [
        sys.executable, 'scripts/eval_metrics.py',
        '--gt', gt_path,
        '--restored', restored_path,
        '--y'  # Include Y channel metrics
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running metrics: {e}")
        return False

def main():
    """Main function for evaluating strong grain removal model"""
    
    parser = argparse.ArgumentParser(
        description='Evaluate Strong Grain Removal Model with PSNR, SSIM, LPIPS'
    )
    parser.add_argument(
        '--gt', 
        required=True, 
        help='Path to ground truth / original image'
    )
    parser.add_argument(
        '--restored', 
        required=True, 
        help='Path to your strong grain removal model output'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.gt):
        print(f"❌ Ground truth image not found: {args.gt}")
        return
    
    if not os.path.exists(args.restored):
        print(f"❌ Restored image not found: {args.restored}")
        return
    
    if not os.path.exists('scripts/eval_metrics.py'):
        print("❌ Error: scripts/eval_metrics.py not found!")
        return
    
    # Run evaluation
    success = run_eval_metrics(args.gt, args.restored)
    
    if success:
        print(f"\n{'='*70}")
        print("✅ Evaluation Complete!")
        print(f"{'='*70}")
        print()
        print("📊 Metrics Explanation:")
        print("   • PSNR (Peak Signal-to-Noise Ratio): Higher is better (>30 dB is good)")
        print("   • SSIM (Structural Similarity): Higher is better (0-1 scale, >0.9 is good)")
        print("   • LPIPS (Learned Perceptual Image Patch Similarity): Lower is better")
        print()

if __name__ == '__main__':
    main()





