import argparse
import cv2
import numpy as np
import os
import sys
from basicsr.utils import scandir
from multiprocessing import Pool
from os import path as osp
from os import remove as rm


def main(args):
    """Usage:
        For each folder, run this script.
        Typically, there are GT folder and LQ folder to be processed for DIV2K dataset.
        After process, each sub_folder should have the same number of subimages.
        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['input_folder'] = args.input
    opt['std'] = args.std
    scan_directory(opt)


def scan_directory(opt):
    """Scans directory for png images containing a single color.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            std (int): The threshold standard deviation.
    """
    input_folder = opt['input_folder']

    # scan all images
    img_list = list(scandir(input_folder, full_path=True))

    for path in img_list:
        worker(path, opt)
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            std (int): The threshold standard deviation.
    """
    threshold = opt['std']

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    for index in range(3):
        if img[:, :, index].std() > threshold:
            return
    #print(img[:, :, 0].std(), ", ", img[:, :, 1].std(), ", ", img[:, :, 2].std(), ", ", path.split("/")[-1])
    rm(path)
    #print(path.replace(" ", "\ "), end=' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DF2K/DF2K_HR', help='Input folder')
    parser.add_argument('--std', type=int, default=10, help='Threshold standard deviation of channels')

    args = parser.parse_args()

    main(args)
