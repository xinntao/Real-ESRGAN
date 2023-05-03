import argparse
import numpy as np
import os
import sys
import requests
from tqdm import tqdm

from os import path as osp

def main(args):
    opt = {}
    opt['input_folder'] = args.input
    opt['save_folder'] = args.output

    batch_download_images(opt)

def batch_download_images(opt):
    input_list = opt['input_folder']
    save_folder = opt['save_folder']

    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    # Open the text file containing the image paths
    with open(input_list, 'r') as f:
        # Loop through each line in the file
        for line in tqdm(f):
            # Remove any whitespace from the beginning and end of the line
            line = line.strip()
            # Construct the save path for the image
            save_path = os.path.join(save_folder, line.split('/')[-2])
            # To save to a relative path.
            r = requests.get(line)
            with open(save_path + ".png", 'wb') as f:
                f.write(r.content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='Input folder')
    parser.add_argument('--output', type=str, default='', help='Output folder')
    args = parser.parse_args()

    main(args)
