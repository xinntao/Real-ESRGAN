#!/usr/bin/env python3
"""Simple script to plot training losses from log file."""

import re
from collections import defaultdict

import matplotlib.pyplot as plt

# Parse training log
losses = defaultdict(list)
iterations = []

with open('training_output.log', 'r') as f:
    for line in f:
        if 'iter:' in line and 'l_g_pix:' in line:
            # Extract iteration number
            iter_match = re.search(r'iter:\s+(\d+)', line)
            if iter_match:
                iterations.append(int(iter_match.group(1)))

                # Extract all loss values
                loss_patterns = {
                    'l_g_pix': r'l_g_pix:\s+([\d.e\-+]+)',
                    'l_g_percep': r'l_g_percep:\s+([\d.e\-+]+)',
                    'l_g_style': r'l_g_style:\s+([\d.e\-+]+)',
                    'l_g_lpips': r'l_g_lpips:\s+([\d.e\-+]+)',
                    'l_g_dists': r'l_g_dists:\s+([\d.e\-+]+)',
                    'l_g_gan': r'l_g_gan:\s+([\d.e\-+]+)',
                }

                for loss_name, pattern in loss_patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        losses[loss_name].append(float(match.group(1)))
                    else:
                        losses[loss_name].append(None)

if not iterations:
    print("No training data found in log file!")
    exit(1)

# Create plots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Real-ESRGAN SwinIR Training Losses', fontsize=16)

plot_configs = [
    ('l_g_pix', 'Pixel L1 Loss', axes[0, 0]),
    ('l_g_percep', 'VGG Perceptual Loss', axes[0, 1]),
    ('l_g_lpips', 'LPIPS Loss (Perceptual)', axes[1, 0]),
    ('l_g_dists', 'DISTS Loss (Perceptual)', axes[1, 1]),
    ('l_g_style', 'VGG Style Loss', axes[2, 0]),
    ('l_g_gan', 'GAN Loss', axes[2, 1]),
]

for loss_name, title, ax in plot_configs:
    if loss_name in losses and any(v is not None for v in losses[loss_name]):
        valid_data = [(i, v) for i, v in zip(iterations, losses[loss_name]) if v is not None]
        if valid_data:
            iters, vals = zip(*valid_data)
            ax.plot(iters, vals, linewidth=2, color='#2E86AB')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss Value')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('training_losses.png', dpi=150, bbox_inches='tight')
print("✅ Loss curves saved to: training_losses.png")
print(f"📊 Plotted {len(iterations)} iterations")
print("🔥 Current losses:")
for loss_name, title, _ in plot_configs:
    if loss_name in losses and losses[loss_name]:
        latest = [v for v in losses[loss_name] if v is not None]
        if latest:
            print(f"  {title:30s}: {latest[-1]:.4f}")
