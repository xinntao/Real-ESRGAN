import os
import sys
import cv2
from tqdm import tqdm

MIN_THRESH = 70
MAX_THRESH = 245


def get_mean_threshold(filepath):
    image = cv2.imread(filepath, 0)  # Read image as grayscale

    # Calculate the global threshold
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    return cv2.mean(binary_image)[0]


def delete_outliers(input_dir, delete_dir):
    for filename in tqdm(os.listdir(input_dir)):
        try:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(input_dir, filename)
                mean_thresh = get_mean_threshold(image_path)
                if not (MIN_THRESH < mean_thresh < MAX_THRESH):
                    os.rename(image_path, os.path.join(delete_dir, filename))
        except:
            print("Error on " + filename)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python script.py filepath')
        sys.exit(1)

    input_dir = sys.argv[1]
    delete_dir = sys.argv[2]

    delete_outliers(input_dir, delete_dir)
