# -*- coding: utf-8 -*-

import argparse

import yaml
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet


def main(args):
    # An instance of the model
    with open(args.config, 'r') as reader:
        config = yaml.load(reader, Loader=yaml.FullLoader)
    print('network_g config:', config['network_g'])
    model = RRDBNet(num_in_ch=config['network_g']['num_in_ch'],
                    num_out_ch=config['network_g']['num_out_ch'],
                    num_feat=config['network_g']['num_feat'],
                    num_block=config['network_g']['num_block'],
                    num_grow_ch=config['network_g']['num_grow_ch'],
                    scale=config['scale'])

    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    model.load_state_dict(torch.load(args.input)[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    # An example input
    x = torch.rand(1, 3, config['network_g']['num_feat'], config['network_g']['num_feat'])
    # Export the model
    with torch.no_grad():
        torch_out = torch.onnx._export(
            model, x, args.output,
            opset_version=args.opset_version,
            export_params=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
    print(torch_out.shape)
    print(f"Done! Exported to {args.output}")


if __name__ == '__main__':
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='specify input model path')
    parser.add_argument('--output', type=str, required=True, help='specify output model path')
    parser.add_argument('--config', type=str, required=True, help='specify config path')
    parser.add_argument('--opset_version', type=int, default=11, help='specify opset version, defaults to 11')
    parser.add_argument('--params', action='store_false', help='Use params instead of params_ema')
    args = parser.parse_args()

    main(args)
