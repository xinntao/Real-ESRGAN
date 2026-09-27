import cv2
import numpy as np
import os
import pytest
import tempfile
import torch

from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer


@pytest.fixture(scope='module')
def cpu_restorer():
    model = SRVGGNetCompact(num_feat=4, num_conv=1, upscale=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # A known nonzero residual exposes normalization errors as well as dtype loss.
        model.body[-1].bias.fill_(0.125)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = os.path.join(directory, 'model.pth')
        torch.save({'params': model.state_dict()}, checkpoint)
        yield RealESRGANer(
            scale=2, model_path=checkpoint, model=model, tile_pad=1, pre_pad=0, device=torch.device('cpu'))


@pytest.mark.parametrize('dtype,value', [(np.uint16, 0), (np.uint16, 128), (np.uint16, 256), (np.uint16, 512),
                                         (np.uint8, 128)])
@pytest.mark.parametrize('channels,mode', [(1, 'L'), (3, 'RGB'), (4, 'RGBA')])
@pytest.mark.parametrize('tile', [0, 2])
def test_enhance_uses_uint16_range_for_dark_images(cpu_restorer, dtype, value, channels, mode, tile):
    shape = (4, 4) if channels == 1 else (4, 4, channels)
    image = np.full(shape, value, dtype=dtype)
    # Use the same lossless PNG decoding path as the command-line inference tool.
    encoded, png = cv2.imencode('.png', image)
    assert encoded
    image = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    cpu_restorer.tile_size = tile

    output, output_mode = cpu_restorer.enhance(image)

    max_range = np.iinfo(dtype).max
    expected_value = round(value + max_range * 0.125)
    expected_shape = (8, 8) if channels == 1 else (8, 8, channels)
    assert output_mode == mode
    assert output.dtype == dtype
    np.testing.assert_array_equal(output, np.full(expected_shape, expected_value, dtype=dtype))


def test_dark_uint16_alpha_with_resize(cpu_restorer):
    image = np.full((4, 4, 4), 128, dtype=np.uint16)
    cpu_restorer.tile_size = 0

    output, mode = cpu_restorer.enhance(image, alpha_upsampler=None)

    assert mode == 'RGBA'
    assert output.dtype == np.uint16
    np.testing.assert_array_equal(output[:, :, :3], np.full((8, 8, 3), 8320, dtype=np.uint16))
    np.testing.assert_array_equal(output[:, :, 3], np.full((8, 8), 128, dtype=np.uint16))
