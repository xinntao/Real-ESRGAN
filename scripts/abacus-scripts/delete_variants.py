import argparse
import os
import re

from tqdm import tqdm

from variants.variant_generator import META_INFO_PATH

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create variants of image pairs.')
    parser.add_argument('-m', '--meta_info', type=str, help='Name of the meta-info file (without extension0)')
    parser.add_argument('-o', '--out_meta_info', type=str, help='Name of the out meta-info file (without extension)')
    parser.add_argument('-d', '--dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()

    # Construct paths
    meta_info_path = os.path.join(META_INFO_PATH, f"{args.meta_info}.txt")

    # Read image pairs from meta-info file and create variants
    with open(meta_info_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []

    for line in tqdm(lines):
        original_path, variant_path = line.strip().split(',')

        # Remove leading/trailing spaces from paths
        original_path = original_path.strip()
        variant_path = variant_path.strip()

        if not check_pattern(variant_path):
            updated_lines.append(f'{original_path}, {variant_path}\n')
        else:
            os.remove(os.path.join(args.dataset_path, variant_path))

    with open(os.path.join(META_INFO_PATH, f"{args.out_meta_info}.txt"), "w") as f:
        f.writelines(updated_lines)



def check_pattern(string):
    pattern = r"_b_c_\d+\.png$"
    match = re.search(pattern, string)
    if match:
        return True
    else:
        return False


if __name__ == '__main__':
    main()
