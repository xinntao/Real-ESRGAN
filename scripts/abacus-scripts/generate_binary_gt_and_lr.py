import os
import sys
import cv2
from tqdm import tqdm
import numpy as np

def convert_to_binary(input_dir, output_gt_path, output_lr_path):
    if not os.path.exists(output_gt_path):
        os.makedirs(output_gt_path)
    if not os.path.exists(output_lr_path):
        os.makedirs(output_lr_path)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path, 0)  # Read image as grayscale

                # Calculate the global threshold
                threshold = cv2.mean(image)[0]  # You can also use cv2.medianBlur() to calculate the median threshold

                # Define the threshold window
                min_threshold = 140  # Minimum threshold value
                max_threshold = 160  # Maximum threshold value

                threshold = min(max(threshold, min_threshold), max_threshold)

                # Apply adaptive thresholding
                _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

                # Save binary image to ground truth path
                output_gt_image_path = os.path.join(output_gt_path, f'{filename}')
                cv2.imwrite(output_gt_image_path, binary_image)

                # Scale down the original image
                scaled_image = cv2.resize(image, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

                # Save scaled image to low-resolution path
                output_lr_image_path = os.path.join(output_lr_path, f'{filename}')
                cv2.imwrite(output_lr_image_path, scaled_image)

                # Delete original image
                os.remove(image_path)
            except Exception as e:
                print(e)

    print('All images processed.')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python script.py input_dir output_gt_path output_lr_path')
        sys.exit(1)

    input_dir = sys.argv[1]
    output_gt_path = sys.argv[2]
    output_lr_path = sys.argv[3]

    convert_to_binary(input_dir, output_gt_path, output_lr_path)
