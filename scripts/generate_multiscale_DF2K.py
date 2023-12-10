import argparse
import glob
import os

import threading
import tqdm
from PIL import Image


class WorkerThread(threading.Thread):
    def __init__(self, lst, th_no):
        threading.Thread.__init__(self)
        self.lst = lst
        self.th_no = th_no

    def run(self):
        worker(self.lst)


def worker(path_list):
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = [0.75, 0.5, 1 / 3]
    shortest_edge = 400

    for path in tqdm.tqdm(path_list):
        # print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(scale_list):
            # print(f'\t{scale:.2f}')
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            rlt.save(os.path.join(args.output, f'{basename}_T{idx}.png'))

        # save the smallest image which the shortest edge is 400
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(args.output, f'{basename}_T{idx + 1}.png'))


def main():
    # the number of workers. set it according to your CPU cores 
    num_workers = 64
    worker_pool = []
    total_tasks = sorted(glob.glob(os.path.join(args.input, '*')))
    divided_tasks = [total_tasks[i:i + num_workers] for i in range(0, len(total_tasks), num_workers)]

    for i in range(num_workers):
        worker_pool.append(WorkerThread(divided_tasks[i], i))
        worker_pool[-1].start()

    for wt in tqdm.tqdm(worker_pool):
        wt.join()


if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/xteam/zhaohao/image-super-resolution/train/DF2K/HR',
                        help='Input folder')
    parser.add_argument('--output', type=str, default='/home/xteam/zhaohao/image-super-resolution/train/DF2K/HR[MS]',
                        help='Output folder')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    main()
