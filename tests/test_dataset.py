import pytest
import yaml

from realesrgan.data.realesrgan_dataset import RealESRGANDataset
from realesrgan.data.realesrgan_paired_dataset import RealESRGANPairedDataset


def test_realesrgan_dataset():

    with open('tests/data/test_realesrgan_dataset.yml', mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    dataset = RealESRGANDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info
    assert dataset.kernel_list == [
        'iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'
    ]  # correct initialization the degradation configurations
    assert dataset.betag_range2 == [0.5, 4]

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['gt', 'kernel1', 'kernel2', 'sinc_kernel', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 400, 400)
    assert result['kernel1'].shape == (21, 21)
    assert result['kernel2'].shape == (21, 21)
    assert result['sinc_kernel'].shape == (21, 21)
    assert result['gt_path'] == 'tests/data/gt/baboon.png'

    # ------------------ test lmdb backend -------------------- #
    opt['dataroot_gt'] = 'tests/data/gt.lmdb'
    opt['io_backend']['type'] = 'lmdb'

    dataset = RealESRGANDataset(opt)
    assert dataset.io_backend_opt['type'] == 'lmdb'  # io backend
    assert len(dataset.paths) == 2  # whether to read correct meta info
    assert dataset.kernel_list == [
        'iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'
    ]  # correct initialization the degradation configurations
    assert dataset.betag_range2 == [0.5, 4]

    # test __getitem__
    result = dataset.__getitem__(1)
    # check returned keys
    expected_keys = ['gt', 'kernel1', 'kernel2', 'sinc_kernel', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 400, 400)
    assert result['kernel1'].shape == (21, 21)
    assert result['kernel2'].shape == (21, 21)
    assert result['sinc_kernel'].shape == (21, 21)
    assert result['gt_path'] == 'comic'

    # ------------------ test with sinc_prob = 0 -------------------- #
    opt['dataroot_gt'] = 'tests/data/gt.lmdb'
    opt['io_backend']['type'] = 'lmdb'
    opt['sinc_prob'] = 0
    opt['sinc_prob2'] = 0
    opt['final_sinc_prob'] = 0
    dataset = RealESRGANDataset(opt)
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['gt', 'kernel1', 'kernel2', 'sinc_kernel', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 400, 400)
    assert result['kernel1'].shape == (21, 21)
    assert result['kernel2'].shape == (21, 21)
    assert result['sinc_kernel'].shape == (21, 21)
    assert result['gt_path'] == 'baboon'

    # ------------------ lmdb backend should have paths ends with lmdb -------------------- #
    with pytest.raises(ValueError):
        opt['dataroot_gt'] = 'tests/data/gt'
        opt['io_backend']['type'] = 'lmdb'
        dataset = RealESRGANDataset(opt)


def test_realesrgan_paired_dataset():

    with open('tests/data/test_realesrgan_paired_dataset.yml', mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    dataset = RealESRGANPairedDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['gt', 'lq', 'gt_path', 'lq_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 128, 128)
    assert result['lq'].shape == (3, 32, 32)
    assert result['gt_path'] == 'tests/data/gt/baboon.png'
    assert result['lq_path'] == 'tests/data/lq/baboon.png'

    # ------------------ test lmdb backend -------------------- #
    opt['dataroot_gt'] = 'tests/data/gt.lmdb'
    opt['dataroot_lq'] = 'tests/data/lq.lmdb'
    opt['io_backend']['type'] = 'lmdb'

    dataset = RealESRGANPairedDataset(opt)
    assert dataset.io_backend_opt['type'] == 'lmdb'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info

    # test __getitem__
    result = dataset.__getitem__(1)
    # check returned keys
    expected_keys = ['gt', 'lq', 'gt_path', 'lq_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 128, 128)
    assert result['lq'].shape == (3, 32, 32)
    assert result['gt_path'] == 'comic'
    assert result['lq_path'] == 'comic'

    # ------------------ test paired_paths_from_folder -------------------- #
    opt['dataroot_gt'] = 'tests/data/gt'
    opt['dataroot_lq'] = 'tests/data/lq'
    opt['io_backend'] = dict(type='disk')
    opt['meta_info'] = None

    dataset = RealESRGANPairedDataset(opt)
    assert dataset.io_backend_opt['type'] == 'disk'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['gt', 'lq', 'gt_path', 'lq_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 128, 128)
    assert result['lq'].shape == (3, 32, 32)

    # ------------------ test normalization -------------------- #
    dataset.mean = [0.5, 0.5, 0.5]
    dataset.std = [0.5, 0.5, 0.5]
    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['gt', 'lq', 'gt_path', 'lq_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (3, 128, 128)
    assert result['lq'].shape == (3, 32, 32)
