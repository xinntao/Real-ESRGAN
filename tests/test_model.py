import torch
import yaml
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.losses.losses import GANLoss, L1Loss, PerceptualLoss

from realesrgan.archs.discriminator_arch import UNetDiscriminatorSN
from realesrgan.models.realesrgan_model import RealESRGANModel
from realesrgan.models.realesrnet_model import RealESRNetModel


def test_realesrnet_model():
    with open('tests/data/test_realesrnet_model.yml', mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    # build model
    model = RealESRNetModel(opt)
    # test attributes
    assert model.__class__.__name__ == 'RealESRNetModel'
    assert isinstance(model.net_g, RRDBNet)
    assert isinstance(model.cri_pix, L1Loss)
    assert isinstance(model.optimizers[0], torch.optim.Adam)

    # prepare data
    gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    kernel1 = torch.rand((1, 5, 5), dtype=torch.float32)
    kernel2 = torch.rand((1, 5, 5), dtype=torch.float32)
    sinc_kernel = torch.rand((1, 5, 5), dtype=torch.float32)
    data = dict(gt=gt, kernel1=kernel1, kernel2=kernel2, sinc_kernel=sinc_kernel)
    model.feed_data(data)
    # check dequeue
    model.feed_data(data)
    # check data shape
    assert model.lq.shape == (1, 3, 8, 8)
    assert model.gt.shape == (1, 3, 32, 32)

    # change probability to test if-else
    model.opt['gaussian_noise_prob'] = 0
    model.opt['gray_noise_prob'] = 0
    model.opt['second_blur_prob'] = 0
    model.opt['gaussian_noise_prob2'] = 0
    model.opt['gray_noise_prob2'] = 0
    model.feed_data(data)
    # check data shape
    assert model.lq.shape == (1, 3, 8, 8)
    assert model.gt.shape == (1, 3, 32, 32)

    # ----------------- test nondist_validation -------------------- #
    # construct dataloader
    dataset_opt = dict(
        name='Demo',
        dataroot_gt='tests/data/gt',
        dataroot_lq='tests/data/lq',
        io_backend=dict(type='disk'),
        scale=4,
        phase='val')
    dataset = PairedImageDataset(dataset_opt)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    assert model.is_train is True
    model.nondist_validation(dataloader, 1, None, False)
    assert model.is_train is True


def test_realesrgan_model():
    with open('tests/data/test_realesrgan_model.yml', mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    # build model
    model = RealESRGANModel(opt)
    # test attributes
    assert model.__class__.__name__ == 'RealESRGANModel'
    assert isinstance(model.net_g, RRDBNet)  # generator
    assert isinstance(model.net_d, UNetDiscriminatorSN)  # discriminator
    assert isinstance(model.cri_pix, L1Loss)
    assert isinstance(model.cri_perceptual, PerceptualLoss)
    assert isinstance(model.cri_gan, GANLoss)
    assert isinstance(model.optimizers[0], torch.optim.Adam)
    assert isinstance(model.optimizers[1], torch.optim.Adam)

    # prepare data
    gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    kernel1 = torch.rand((1, 5, 5), dtype=torch.float32)
    kernel2 = torch.rand((1, 5, 5), dtype=torch.float32)
    sinc_kernel = torch.rand((1, 5, 5), dtype=torch.float32)
    data = dict(gt=gt, kernel1=kernel1, kernel2=kernel2, sinc_kernel=sinc_kernel)
    model.feed_data(data)
    # check dequeue
    model.feed_data(data)
    # check data shape
    assert model.lq.shape == (1, 3, 8, 8)
    assert model.gt.shape == (1, 3, 32, 32)

    # change probability to test if-else
    model.opt['gaussian_noise_prob'] = 0
    model.opt['gray_noise_prob'] = 0
    model.opt['second_blur_prob'] = 0
    model.opt['gaussian_noise_prob2'] = 0
    model.opt['gray_noise_prob2'] = 0
    model.feed_data(data)
    # check data shape
    assert model.lq.shape == (1, 3, 8, 8)
    assert model.gt.shape == (1, 3, 32, 32)

    # ----------------- test nondist_validation -------------------- #
    # construct dataloader
    dataset_opt = dict(
        name='Demo',
        dataroot_gt='tests/data/gt',
        dataroot_lq='tests/data/lq',
        io_backend=dict(type='disk'),
        scale=4,
        phase='val')
    dataset = PairedImageDataset(dataset_opt)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    assert model.is_train is True
    model.nondist_validation(dataloader, 1, None, False)
    assert model.is_train is True

    # ----------------- test optimize_parameters -------------------- #
    model.feed_data(data)
    model.optimize_parameters(1)
    assert model.output.shape == (1, 3, 32, 32)
    assert isinstance(model.log_dict, dict)
    # check returned keys
    expected_keys = ['l_g_pix', 'l_g_percep', 'l_g_gan', 'l_d_real', 'out_d_real', 'l_d_fake', 'out_d_fake']
    assert set(expected_keys).issubset(set(model.log_dict.keys()))
