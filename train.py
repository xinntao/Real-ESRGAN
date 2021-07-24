import os.path as osp
from basicsr.train import train_pipeline

import archs  # noqa: F401
import data  # noqa: F401
import models  # noqa: F401

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
