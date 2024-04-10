import argparse
import numpy as np
import os
import sys
import requests
from tqdm import tqdm
from pdf2image import convert_from_path
import requests
import zipfile
from io import BytesIO
from PIL import Image
import shutil

from os import path as osp

def main(args):
    opt = {}
    opt['input_folder'] = args.input
    opt['save_folder'] = args.output

    batch_download_images(opt)

def batch_download_images(opt):
    input_list = opt['input_folder']
    save_folder = opt['save_folder']

    # Open the text file containing the image paths
    with open(input_list, 'r') as f:
        # Loop through each line in the file
        for url in tqdm(f):
            try:
                print(f"Downloading jp2 for {url}")
                # Extract the book identifier from the URL
                identifier = url.split("/")[-5]
                # Construct the download URL for the PDF file
                url = f"https://archive.org/download/{identifier}/{identifier}_jp2.zip"
                # Download the zip file and extract the JP2 images
                response = requests.get(url)
                zip_file = BytesIO(response.content)
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall()

                folder_name = os.path.splitext(os.path.basename(url))[0]
                extracted_folder = os.path.join(os.getcwd(), folder_name)


                # Loop through the extracted JP2 images and convert them to PNG
                for file_name in tqdm(os.listdir(extracted_folder)):
                    try:
                        if file_name.endswith(".jp2"):
                            file_path = os.path.join(extracted_folder, file_name)
                            with Image.open(file_path) as image:
                                png_file_name = os.path.splitext(file_name)[0] + ".png"
                                png_path = os.path.join(save_folder, png_file_name)
                                image.save(png_path, format="PNG")
                                os.remove(file_path)
                    except Exception as e:
                        print(f"Error at page level on {file_name}: {e}")

                # Delete the extracted folder (assumes that the folder name is the same as the zip file name without the extension)
                shutil.rmtree(extracted_folder)
            except Exception as e:
                print(f"Error at book level on {f}: {e}")
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='Input folder')
    parser.add_argument('--output', type=str, default='', help='Output folder')
    args = parser.parse_args()

    main(args)
