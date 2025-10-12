import re

import matplotlib.pyplot as plt
import numpy as np

# Parse the log file
log_file = '/home/henz1000/Real-ESRGAN-ProMax/training_correct.log'

iterations = []
l_g_pix = []
l_g_percep = []
l_g_style = []
l_g_lpips = []
l_g_dists = []
l_g_gan = []
l_d_real = []
l_d_fake = []

with open(log_file, 'r') as f:
    for line in f:
        # Look for lines with loss information
        if 'iter:' in line and 'l_g_pix:' in line:
            # Extract iteration number (handle commas in numbers like 1,000)
            iter_match = re.search(r'iter:\s*([\d,]+)', line)
            if iter_match:
                # Remove commas and convert to int
                iter_num = int(iter_match.group(1).replace(',', ''))
                iterations.append(iter_num)

            # Extract loss values
            loss_patterns = {
                'l_g_pix': r'l_g_pix:\s*([\d.e+-]+)',
                'l_g_percep': r'l_g_percep:\s*([\d.e+-]+)',
                'l_g_style': r'l_g_style:\s*([\d.e+-]+)',
                'l_g_lpips': r'l_g_lpips:\s*([\d.e+-]+)',
                'l_g_dists': r'l_g_dists:\s*([\d.e+-]+)',
                'l_g_gan': r'l_g_gan:\s*([\d.e+-]+)',
                'l_d_real': r'l_d_real:\s*([\d.e+-]+)',
                'l_d_fake': r'l_d_fake:\s*([\d.e+-]+)',
            }

            for loss_name, pattern in loss_patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = float(match.group(1))
                    if loss_name == 'l_g_pix':
                        l_g_pix.append(value)
                    elif loss_name == 'l_g_percep':
                        l_g_percep.append(value)
                    elif loss_name == 'l_g_style':
                        l_g_style.append(value)
                    elif loss_name == 'l_g_lpips':
                        l_g_lpips.append(value)
                    elif loss_name == 'l_g_dists':
                        l_g_dists.append(value)
                    elif loss_name == 'l_g_gan':
                        l_g_gan.append(value)
                    elif loss_name == 'l_d_real':
                        l_d_real.append(value)
                    elif loss_name == 'l_d_fake':
                        l_d_fake.append(value)

print(f"Parsed {len(iterations)} data points from log file")

# Filter to only first 25,000 iterations
max_iter = 25000
filtered_indices = [i for i, iter_val in enumerate(iterations) if iter_val <= max_iter]

iterations = [iterations[i] for i in filtered_indices]
l_g_pix = [l_g_pix[i] for i in filtered_indices]
l_g_percep = [l_g_percep[i] for i in filtered_indices]
l_g_style = [l_g_style[i] for i in filtered_indices]
l_g_lpips = [l_g_lpips[i] for i in filtered_indices]
l_g_dists = [l_g_dists[i] for i in filtered_indices]
l_g_gan = [l_g_gan[i] for i in filtered_indices]
l_d_real = [l_d_real[i] for i in filtered_indices]
l_d_fake = [l_d_fake[i] for i in filtered_indices]

print(f"Filtered to {len(iterations)} data points (up to iteration {max_iter})")

# Create the plots (2x3 grid for 6 generator losses)
fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle('Training Losses - SwinIR 25K', fontsize=16, fontweight='bold')

# Plot 1: Pixel Loss
axes[0, 0].plot(iterations, l_g_pix, linewidth=1, alpha=0.7)
axes[0, 0].set_title('Generator Pixel Loss (L1)')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Perceptual Loss
axes[0, 1].plot(iterations, l_g_percep, linewidth=1, alpha=0.7, color='orange')
axes[0, 1].set_title('Generator Perceptual Loss')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Style Loss
axes[0, 2].plot(iterations, l_g_style, linewidth=1, alpha=0.7, color='green')
axes[0, 2].set_title('Generator Style Loss')
axes[0, 2].set_xlabel('Iteration')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: LPIPS Loss
axes[1, 0].plot(iterations, l_g_lpips, linewidth=1, alpha=0.7, color='red')
axes[1, 0].set_title('Generator LPIPS Loss')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: DISTS Loss
axes[1, 1].plot(iterations, l_g_dists, linewidth=1, alpha=0.7, color='purple')
axes[1, 1].set_title('Generator DISTS Loss')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: GAN Loss (Generator)
axes[1, 2].plot(iterations, l_g_gan, linewidth=1, alpha=0.7, color='brown')
axes[1, 2].set_title('Generator GAN Loss')
axes[1, 2].set_xlabel('Iteration')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/henz1000/Real-ESRGAN-ProMax/training_losses_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved to: /home/henz1000/Real-ESRGAN-ProMax/training_losses_plot.png")
plt.close()

# Create a second figure with smoothed curves for better trend visualization
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Training Losses (Smoothed) - SwinIR 25K', fontsize=16, fontweight='bold')

# Simple moving average function
def smooth(data, window=50):
    if len(data) < window:
        return data
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed

# Adjust iterations to match smoothed data length
def get_smooth_iters(data, window=50):
    if len(data) < window:
        return iterations
    smoothed_len = len(np.convolve(data, np.ones(window)/window, mode='valid'))
    offset = (len(data) - smoothed_len) // 2
    return iterations[offset:offset+smoothed_len]

smooth_iters = get_smooth_iters(l_g_pix, 50)

# Smoothed Generator Losses
axes2[0, 0].plot(smooth_iters, smooth(l_g_pix, 50), label='Pixel', linewidth=2)
axes2[0, 0].plot(smooth_iters, smooth(l_g_percep, 50), label='Perceptual', linewidth=2)
axes2[0, 0].plot(smooth_iters, smooth(l_g_style, 50), label='Style', linewidth=2)
axes2[0, 0].set_title('Generator Losses (VGG-based)')
axes2[0, 0].set_xlabel('Iteration')
axes2[0, 0].set_ylabel('Loss')
axes2[0, 0].legend()
axes2[0, 0].grid(True, alpha=0.3)

# Smoothed LPIPS and DISTS
axes2[0, 1].plot(smooth_iters, smooth(l_g_lpips, 50), label='LPIPS', linewidth=2, color='red')
axes2[0, 1].plot(smooth_iters, smooth(l_g_dists, 50), label='DISTS', linewidth=2, color='purple')
axes2[0, 1].set_title('Perceptual Metrics')
axes2[0, 1].set_xlabel('Iteration')
axes2[0, 1].set_ylabel('Loss')
axes2[0, 1].legend()
axes2[0, 1].grid(True, alpha=0.3)

# Smoothed GAN Losses
axes2[1, 0].plot(smooth_iters, smooth(l_g_gan, 50), label='Generator GAN', linewidth=2, color='brown')
axes2[1, 0].set_title('GAN Loss')
axes2[1, 0].set_xlabel('Iteration')
axes2[1, 0].set_ylabel('Loss')
axes2[1, 0].legend()
axes2[1, 0].grid(True, alpha=0.3)

# Smoothed Discriminator Losses
axes2[1, 1].plot(smooth_iters, smooth(l_d_real, 50), label='D Real', linewidth=2, color='cyan')
axes2[1, 1].plot(smooth_iters, smooth(l_d_fake, 50), label='D Fake', linewidth=2, color='magenta')
axes2[1, 1].set_title('Discriminator Losses')
axes2[1, 1].set_xlabel('Iteration')
axes2[1, 1].set_ylabel('Loss')
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/henz1000/Real-ESRGAN-ProMax/training_losses_smoothed.png', dpi=150, bbox_inches='tight')
print("Smoothed plot saved to: /home/henz1000/Real-ESRGAN-ProMax/training_losses_smoothed.png")
plt.close()

print("\nTraining Summary:")
print(f"Total iterations: {iterations[-1]}")
print("Final losses:")
print(f"  - Pixel Loss: {l_g_pix[-1]:.6f}")
print(f"  - Perceptual Loss: {l_g_percep[-1]:.6f}")
print(f"  - Style Loss: {l_g_style[-1]:.6f}")
print(f"  - LPIPS Loss: {l_g_lpips[-1]:.6f}")
print(f"  - DISTS Loss: {l_g_dists[-1]:.6f}")
print(f"  - GAN Loss: {l_g_gan[-1]:.6f}")
