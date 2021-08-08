import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs  # noqa: F401
import realesrgan.data  # noqa: F401
import realesrgan.models  # noqa: F401

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
