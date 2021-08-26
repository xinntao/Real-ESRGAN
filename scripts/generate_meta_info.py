import argparse
import glob
import os


def main(args):
    txt_file = open(args.meta_info, 'w')
    for folder, root in zip(args.input, args.root):
        img_paths = sorted(glob.glob(os.path.join(folder, '*')))
        for img_path in img_paths:
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
    args = parser.parse_args()
    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    main(args)
