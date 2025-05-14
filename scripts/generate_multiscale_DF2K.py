import argparse
import glob
import os
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp

def process_image(path, args, scale_list, shortest_edge):
    """Worker function to process a single image."""
    try:
        basename = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path)
        width, height = img.size

        # Process multi-scale versions
        for idx, scale in enumerate(scale_list):
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            rlt.save(os.path.join(args.output, f'{basename}T{idx}.png'))

        # Save the smallest image with shortest edge as 400
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(args.output, f'{basename}T{len(scale_list)}.png'))
    except Exception as e:
        print(f"Error processing {path}: {e}")

def process_image_wrapper(args_tuple):
    """Wrapper function to unpack arguments for imap_unordered."""
    path, args, scale_list, shortest_edge = args_tuple
    process_image(path, args, scale_list, shortest_edge)

def main(args):
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = [0.75, 0.5, 1 / 3]
    shortest_edge = 400

    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    
    # Validate number of processes
    if mp.cpu_count() < args.num_processes:
        num_processes = mp.cpu_count()
        print(f"Requested number of processes ({args.num_processes}) exceeds available CPU count ({num_processes}). Using {num_processes} processes.")
    else:
        num_processes = args.num_processes
    if num_processes < 1:
        num_processes = 1
        print(f"Warning: num_processes set to {num_processes} as provided value was invalid.")

    # Set up multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        # Prepare arguments for each image
        args_list = [(path, args, scale_list, shortest_edge) for path in path_list]
        # Use tqdm to show progress bar
        list(tqdm(
            pool.imap_unordered(process_image_wrapper, args_list),
            total=len(path_list),
            desc="Processing images"
        ))

if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='datasets/DF2K/DF2K_HR', help='Input folder')
    parser.add_argument('-o', '--output', type=str, default='datasets/DF2K/DF2K_multiscale', help='Output folder')
    parser.add_argument('-n', '--num_processes', type=int, default=4, help='Number of processes for multiprocessing (default: 4)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
