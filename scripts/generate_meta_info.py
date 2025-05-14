import argparse
import cv2
import glob
import os
from tqdm import tqdm


def main(args):
    txt_file = open(args.meta_info, 'w')
    for folder, root in zip(args.input, args.root):
        img_paths = sorted(glob.glob(os.path.join(folder, '*')))
        for img_path in tqdm(img_paths, desc=f"Processing {os.path.basename(folder)}"):
            status = True
            if args.check:
                try:
                    img = cv2.imread(img_path)
                except (IOError, OSError) as error:
                    print(f'Read {img_path} error: {error}')
                    status = False
                if img is None:
                    status = False
                    print(f'Img is None: {img_path}')
            if status:
                if root == "":
                    img_name = img_path
                else:
                    img_name = os.path.relpath(img_path, root)
                txt_file.write(f'{img_name}\n')
    txt_file.close()


if __name__ == '__main__':
    """Generate meta info (txt file) for only Ground-Truth images.

    It can also generate meta info from several folders into one txt file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['datasets/DF2K/DF2K_HR', 'datasets/DF2K/DF2K_multiscale'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['datasets/DF2K', 'datasets/DF2K'],
        help='Folder root for relative paths, should match input folders in length. '
             'Use "" to keep full image paths without computing relative paths.')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='datasets/DF2K/meta_info/meta_info_DF2Kmultiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)

    main(args)
