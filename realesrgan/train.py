# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

from .archs import *
from .data import *
from .models import *

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
