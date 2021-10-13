import argparse
import glob
import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class Resizer(Dataset):
    def __init__(self, input_path, output_path):
        super().__init__()
        self.output_path = output_path

        # For DF2K, we consider the following three scales,
        # and the smallest image whose shortest edge is 400
        self.scale_list = [0.75, 0.5, 1 / 3]
        self.shortest_edge = 400

        self.path_list = sorted(glob.glob(os.path.join(input_path, '*')))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, item):
        path = self.path_list[item]
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(self.scale_list):
            # print(f'\t{scale:.2f}')
            rlt = img.resize((int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            rlt.save(os.path.join(args.output, f'{basename}T{idx}.png'))

        # save the smallest image which the shortest edge is 400
        if width < height:
            ratio = height / width
            width = self.shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = self.shortest_edge
            width = int(height * ratio)
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        rlt.save(os.path.join(args.output, f'{basename}T{idx + 1}.png'))
        return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DF2K/DF2K_HR', help='Input folder')
    parser.add_argument('--output', type=str, default='datasets/DF2K/DF2K_multiscale', help='Output folder')
    parser.add_argument('--num_threads', type=int, )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    dataset = Resizer(args.input, args.output)
    loader = DataLoader(dataset, num_workers=16)

    for _ in tqdm(loader, desc="Resampling images to: {}".format(args.output)):
        pass
