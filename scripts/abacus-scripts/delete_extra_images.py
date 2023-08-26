import argparse
import glob
import os


def main(args):
    img_paths_gt = set(os.listdir(args.input[0]))
    img_paths_lq = set(os.listdir(args.input[1]))

    for img_path_lq in img_paths_lq:
        if img_path_lq not in img_paths_gt:
            os.remove(os.path.join(args.input[1], img_path_lq))

    for img_path_gt in img_paths_gt:
        if img_path_gt not in img_paths_lq:
            os.remove(os.path.join(args.input[0], img_path_gt))


if __name__ == '__main__':
    """This script is used to generate meta info (txt file) for paired images.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['datasets/DF2K/DIV2K_train_HR_sub', 'datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub'],
        help='Input folder, should be [gt_folder, lq_folder]')
    args = parser.parse_args()

    main(args)