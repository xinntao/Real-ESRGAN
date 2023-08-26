import argparse
import os

from tqdm import tqdm

from variants.variant_generator import ImageVariantGenerator, BrightnessContrastVariantGenerator, META_INFO_PATH

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create variants of image pairs.')
    parser.add_argument('-m', '--meta_info', type=str, help='Name of the meta-info file (without extension)')
    parser.add_argument('-b', '--brightness', action='store_true', help='Include brightness variants')
    parser.add_argument('-v', '--num_variants', type=int, default=6, help='Number of variants per type')
    parser.add_argument('-d', '--dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()

    # Construct paths
    meta_info_path = os.path.join(META_INFO_PATH, f"{args.meta_info}.txt")

    # Create variant generators based on arguments
    variant_generators = []
    if args.brightness:
        variant_generators.append(BrightnessContrastVariantGenerator(args.dataset_path, meta_info_path))

    # Read image pairs from meta-info file and create variants
    with open(meta_info_path, 'r') as f:
        lines = f.readlines()

    num_lines = len(lines)

    for i, line in enumerate(lines):
        original_path, variant_path = line.strip().split(',')

        # Remove leading/trailing spaces from paths
        original_path = original_path.strip()
        variant_path = variant_path.strip()

        # Create variants for the current image pair
        for variant_generator in variant_generators:
            variant_generator.create_variants(original_path, variant_path, args.num_variants)

        # Update the progress bar every 10 iterations
        if (i + 1) % 10 == 0 or (i + 1) == num_lines:
            tqdm.write(f"Processed {i + 1} out of {num_lines} lines.")


if __name__ == '__main__':
    main()
