# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple, List

import cv2
import numpy as np
import onnxruntime as ort


class BaseModel:
    """ Inference with ONNXRuntime
    """
    def __init__(self,
                 model_path: str,
                 intra_op_num_threads: int = -1,
                 providers: List[str] = ['CPUExecutionProvider']):
        """ Initializer
        Args:
          model_path (str): path to model
          intra_op_num_threads (int): num threads, defaults to -1
          providers (List[str]): onnxruntime providers, defaults to ['CPUExecutionProvider']
        """
        if intra_op_num_threads > 0:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = intra_op_num_threads
            self.sess = ort.InferenceSession(model_path, sess_options, providers=providers)
        else:
            self.sess = ort.InferenceSession(model_path, providers=providers)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        input = self.sess.get_inputs()[0].name
        output = self.sess.get_outputs()[0].name
        return self.sess.run([output], {input: img})[0]


class RealESRGAN:
    def __init__(self,
                 model_path: str,
                 scale: int = 4,
                 tile: int = 0,
                 tile_pad: int = 10,
                 pre_pad: int = 10,
                 verbose: bool = True,
                 **kwargs):
        """A helper class for upsampling images with RealESRGAN.

        Args:
            model_path (str): The path to the pretrained model.
            scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
            tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
                input images into tiles, and then process each of them. Finally, they will be merged into one image.
                0 denotes for do not use tile. Default: 0.
            tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
            pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
            verbose (bool):  whether to verbose log. Default: True
        """
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.verbose = verbose
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        else:
            self.mod_scale = None
        self.model = BaseModel(model_path, **kwargs)

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible
        """
        img = np.transpose(img, (2, 0, 1)).astype('float32')
        img = np.expand_dims(img, 0)

        # pre_pad
        if self.pre_pad != 0:
            img = np.pad(img, [(0, 0), (0, 0), (0, self.pre_pad), (0, self.pre_pad)], 'reflect')
        # mod pad for divisible borders
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            img = np.pad(img, [(0, 0), (0, 0), (0, self.mod_pad_h), (0, self.mod_pad_w)], 'reflect')
        return img

    def predict(self, img: np.ndarray) -> np.ndarray:
        # model inference
        return self.model(img)

    def tile_predict(self, img: np.ndarray) -> np.ndarray:
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = np.zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    output_tile = self.model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                if self.verbose:
                    print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                       output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                  output_start_x_tile:output_end_x_tile]
        return output

    def post_process(self, output: np.ndarray) -> np.ndarray:
        _, _, h, w = output.shape
        # remove extra pad
        if self.mod_scale is not None:
            output = output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            output = output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return output

    def enhance(self,
                img: np.ndarray,
                outscale: Optional[int] = None,
                alpha_upsampler: str = 'realesrgan') -> Tuple[np.ndarray, str]:
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        max_range = 65535 if np.max(img) > 256 else 255
        img = img / max_range
        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'realesrgan':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # process image (without the alpha channel)
        img = self.pre_process(img)
        if self.tile_size > 0:
            logits = self.tile_predict(img)
        else:
            logits = self.predict(img)
        output_img = self.post_process(logits)
        output_img = np.clip(np.squeeze(output_img, 0), 0, 1)
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        # process the alpha channel if necessary
        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                alpha_img = self.pre_process(alpha)
                if self.tile_size > 0:
                    logits = self.tile_predict(alpha_img)
                else:
                    logits = self.predict(alpha_img)
                output_alpha = self.post_process(logits)
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode


if __name__ == '__main__':
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='specify onnx model path')
    parser.add_argument('--input_path', type=str, required=True, help='specify input model path')
    parser.add_argument('--output_path', type=str, required=True, help='specify output model path')
    parser.add_argument('--output_scale', type=int, default=4, help='specify the output scale, defaults to 4')
    parser.add_argument('--num_threads', type=int, default=-1, help='specify num threads of onnx')
    args = parser.parse_args()

    model = RealESRGAN(args.model_path, intra_op_num_threads=args.num_threads)
    img = cv2.imread(args.input_path, cv2.IMREAD_UNCHANGED)
    print('start to enhance...')
    start_time = time.time()
    output, _ = model.enhance(img, outscale=args.output_scale)
    print('inference time:', time.time() - start_time)
    cv2.imwrite(args.output_path, output)
    print(f'the enhanced image successfully saved to {args.output_path}')
