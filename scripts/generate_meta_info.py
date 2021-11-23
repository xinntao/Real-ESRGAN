import argparse
import cv2
import glob
import os


def main(args):
    txt_file = open(args.meta_info, 'w')
    for folder, root in zip(args.input, args.root):
        img_paths = sorted(glob.glob(os.path.join(folder, '*')))
        for img_path in img_paths:
            status = True
            if args.check:
                # read the image once for check, as some images may have errors
                try:
                    img = cv2.imread(img_path)
                except (IOError, OSError) as error:
                    print(f'Read {img_path} error: {error}')
                    status = False
                if img is None:
                    status = False
                    print(f'Img is None: {img_path}')
            if status:
                # get the relative path
                img_name = os.path.relpath(img_path, root)
                print(img_name)
                txt_file.write(f'{img_name}\n')


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
        help='Folder root, should have the length as input folders')
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
