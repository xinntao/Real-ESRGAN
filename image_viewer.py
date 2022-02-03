# import libraries
import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(10, 7))

# setting values to rows and column variables
rows = 2
columns = 2
res_path = 'results_render'

testId = 'X5RUPVA679DE247E6624DDC'


def upscale(image, method='bicubic', scale=0.25):
    if method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)

# reading images
hr = cv2.imread(f'{res_path}/{testId}_SR(4k)_x3.0.png')
lr = cv2.imread(f'{res_path}/{testId}_SR(4k)_x2.0.png')
sr = cv2.imread(f'{res_path}/{testId}_SR(4k)_x1.5.png')
cv2.cvtColor(hr, cv2.COLOR_BGR2RGB, hr)
cv2.cvtColor(lr, cv2.COLOR_BGR2RGB, lr)
cv2.cvtColor(sr, cv2.COLOR_BGR2RGB, sr)

ax1 = plt.subplot(221)
plt.imshow(hr)
plt.axis('off')
plt.title("SR from std")

ax2 = plt.subplot(222)
plt.imshow(hr)
plt.axis('off')
plt.title("SR from std (x3.0)")

import numpy as np
ax3 = plt.subplot(223, sharex=ax2, sharey=ax2)
# plt.imshow(upscale(lr, scale=3.125))
plt.imshow(lr)
plt.axis('off')
plt.title("SR from FHD (x2.0)")

ax4 = plt.subplot(224, sharex=ax2, sharey=ax2)
plt.imshow(sr)
plt.axis('off')
plt.title("SR from QHD (x1.5)")

# # Adds a subplot at the 1st position
# fig.add_subplot(rows, columns, 1)
#
# # showing image
# plt.imshow(hr)
# plt.axis('off')
# plt.title("HR(4k)")
#
# # Adds a subplot at the 2nd position
# fig.add_subplot(rows, columns, 2)
#
# # showing image
# plt.imshow(hr)
# plt.axis('off')
# plt.title("hr")
#
# # Adds a subplot at the 3rd position
# fig.add_subplot(rows, columns, 3)
#
# # showing image
# plt.imshow(lr)
# plt.axis('off')
# plt.title("Third")
#
# # Adds a subplot at the 4th position
# fig.add_subplot(rows, columns, 4)
#
# # showing image
# plt.imshow(sr)
# plt.axis('off')
# plt.title("Fourth")

plt.show()