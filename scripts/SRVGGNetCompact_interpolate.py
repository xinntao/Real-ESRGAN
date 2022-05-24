import sys
import torch
from collections import OrderedDict


def load_nets(net_a_path, net_b_path, loc='cpu'):
    net_a = torch.load(net_a_path, map_location=torch.device(loc))
    net_b = torch.load(net_b_path, map_location=torch.device(loc))
    return net_a, net_b


def interpolate(net_a, net_b, net_interp, alpha):
    for k, v_a in net_a['params'].items():
        print(k)
        v_b = net_b['params'][k]
        net_interp['params'][k] = (1 - alpha) * v_a + alpha * v_b


def main():
    alpha = float(sys.argv[1])
    print('Interpolating with alpha = ', alpha)

    net_a_path = './models/RRDB_PSNR_x4.pth'
    net_b_path = './models/RRDB_ESRGAN_x4.pth'

    net_interp_path = './models/interp_{:02d}.pth'.format(int(alpha*10))
    net_a, net_b = load_nets(net_a_path, net_b_path)

    net_interp = OrderedDict()
    net_interp['params'] = OrderedDict()

    interpolate(net_a, net_b, net_interp, alpha)
    torch.save(net_interp, net_interp_path)


if __name__ == '__main__':
    main()
