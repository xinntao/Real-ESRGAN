import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan.utils import RealESRGANer


def test_realesrganer():
    # initialize with default model
    restorer = RealESRGANer(
        scale=4,
        model_path='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        model=None,
        tile=10,
        tile_pad=10,
        pre_pad=2,
        half=False)
    assert isinstance(restorer.model, RRDBNet)
    assert restorer.half is False
    # initialize with user-defined model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    restorer = RealESRGANer(
        scale=4,
        model_path='experiments/pretrained_models/RealESRGAN_x4plus_anime_6B.pth',
        model=model,
        tile=10,
        tile_pad=10,
        pre_pad=2,
        half=True)
    # test attribute
    assert isinstance(restorer.model, RRDBNet)
    assert restorer.half is True

    # ------------------ test pre_process ---------------- #
    img = np.random.random((12, 12, 3)).astype(np.float32)
    restorer.pre_process(img)
    assert restorer.img.shape == (1, 3, 14, 14)
    # with modcrop
    restorer.scale = 1
    restorer.pre_process(img)
    assert restorer.img.shape == (1, 3, 16, 16)

    # ------------------ test process ---------------- #
    restorer.process()
    assert restorer.output.shape == (1, 3, 64, 64)

    # ------------------ test post_process ---------------- #
    restorer.mod_scale = 4
    output = restorer.post_process()
    assert output.shape == (1, 3, 60, 60)

    # ------------------ test tile_process ---------------- #
    restorer.scale = 4
    img = np.random.random((12, 12, 3)).astype(np.float32)
    restorer.pre_process(img)
    restorer.tile_process()
    assert restorer.output.shape == (1, 3, 64, 64)

    # ------------------ test enhance ---------------- #
    img = np.random.random((12, 12, 3)).astype(np.float32)
    result = restorer.enhance(img, outscale=2)
    assert result[0].shape == (24, 24, 3)
    assert result[1] == 'RGB'

    # ------------------ test enhance with 16-bit image---------------- #
    img = np.random.random((4, 4, 3)).astype(np.uint16) + 512
    result = restorer.enhance(img, outscale=2)
    assert result[0].shape == (8, 8, 3)
    assert result[1] == 'RGB'

    # ------------------ test enhance with gray image---------------- #
    img = np.random.random((4, 4)).astype(np.float32)
    result = restorer.enhance(img, outscale=2)
    assert result[0].shape == (8, 8)
    assert result[1] == 'L'

    # ------------------ test enhance with RGBA---------------- #
    img = np.random.random((4, 4, 4)).astype(np.float32)
    result = restorer.enhance(img, outscale=2)
    assert result[0].shape == (8, 8, 4)
    assert result[1] == 'RGBA'

    # ------------------ test enhance with RGBA, alpha_upsampler---------------- #
    restorer.tile_size = 0
    img = np.random.random((4, 4, 4)).astype(np.float32)
    result = restorer.enhance(img, outscale=2, alpha_upsampler=None)
    assert result[0].shape == (8, 8, 4)
    assert result[1] == 'RGBA'
