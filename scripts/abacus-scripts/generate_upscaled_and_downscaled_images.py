import os
import sys
from PIL import Image
from tqdm import tqdm


def check_visited(scales, filename, out_dir):
    for scale in scales:
        scale_suffix = f'T{int(scale * 100):03d}'
        scaled_path = os.path.join(out_dir, f'{filename}_{scale_suffix}.png')
        if os.path.exists(scaled_path):
            return True
    return False

def scale_images(input_dir, output_dir, image_scales):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = [
        filename
        for filename in os.listdir(input_dir)
        if filename.endswith(('.jpg', '.jpeg', '.png'))
    ]

    with tqdm(total=len(file_list), ncols=80, unit='image') as pbar:
        for filename in file_list:

            visited = check_visited(image_scales, filename, output_dir)

            if not visited:
                try:
                    image_path = os.path.join(input_dir, filename)
                    image = Image.open(image_path)

                    for scale in image_scales:
                        # Scale the image
                        scaled_image = image.resize((int(image.width * scale), int(image.height * scale)))
                        scale_suffix = f'T{int(scale * 100):03d}'
                        scaled_path = os.path.join(output_dir, f'{filename}_{scale_suffix}.png')
                        scaled_image.save(scaled_path)
                except Exception as e:
                    print(f"Exception on image {filename}: {e} Skipping...")

            pbar.update(1)

    print('All images processed.')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python script.py input_dir output_dir scale1 scale2 ...')
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    image_scales = [float(scale) for scale in sys.argv[3:]]

    scale_images(input_dir, output_dir, image_scales)
