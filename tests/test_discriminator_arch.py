import torch

from realesrgan.archs.discriminator_arch import UNetDiscriminatorSN


def test_unetdiscriminatorsn():
    """Test arch: UNetDiscriminatorSN."""

    # model init and forward (cpu)
    net = UNetDiscriminatorSN(num_in_ch=3, num_feat=4, skip_connection=True)
    img = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    output = net(img)
    assert output.shape == (1, 1, 32, 32)

    # model init and forward (gpu)
    if torch.cuda.is_available():
        net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 1, 32, 32)
