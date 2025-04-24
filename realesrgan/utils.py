import cv2
import math
import numpy as np
import os
import queue
import threading
import torch
import torch.cuda as cuda
from basicsr.utils.download_util import load_file_from_url
from torch.nn import functional as F
import torchvision
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_device_config():
    """Verifica a configuração dos dispositivos."""
    print("\nDevice Configuration:")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"Current PyTorch device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    # Verifica se torchvision está usando CUDA
    dummy_tensor = torch.randn(1, 3, 64, 64)
    if torch.cuda.is_available():
        dummy_tensor = dummy_tensor.cuda()
    try:
        # Tenta uma operação do torchvision
        result = torchvision.transforms.functional.resize(dummy_tensor, (32, 32))
        print(f"Torchvision device: {result.device}")
    except RuntimeError:
        print("Warning: Torchvision operations may not be using CUDA properly")
    print()

def print_gpu_info(initial=False):
    """Print GPU information concisely."""
    if cuda.is_available():
        if initial:
            print(f'Using {cuda.get_device_name(0)} | Memory: {cuda.memory_allocated(0) / 1024**2:.1f}MB')
        else:
            print(f'\rGPU Memory: {cuda.memory_allocated(0) / 1024**2:.1f}MB | Peak: {cuda.max_memory_allocated(0) / 1024**2:.1f}MB', end='')

class RealESRGANer():
    """A helper class for upsampling images with RealESRGAN."""

    def __init__(self,
                 scale,
                 model_path,
                 dni_weight=None,
                 model=None,
                 tile=0,
                 tile_pad=10,
                 pre_pad=10,
                 half=False,
                 device=None,
                 gpu_id=None):
        self.scale = scale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = None
        self.half = half

        # Verifica configuração de dispositivos
        check_device_config()

        # initialize model and device
        if gpu_id:
            self.device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        print_gpu_info(initial=True)

        if isinstance(model_path, list):
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = self.dni(model_path[0], model_path[1], dni_weight, loc=self.device)
        else:
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=self.device, weights_only=True)

        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)

        model.eval()
        self.model = model.to(self.device)
        if self.half:
            self.model = self.model.half()

    def dni(self, net_a, net_b, dni_weight, key='params', loc='cpu'):
        """Deep network interpolation."""
        net_a = torch.load(net_a, map_location=loc, weights_only=True)
        net_b = torch.load(net_b, map_location=loc, weights_only=True)
        for k, v_a in net_a[key].items():
            net_a[key][k] = dni_weight[0] * v_a + dni_weight[1] * net_b[key][k]
        return net_a

    def pre_process(self, img):
        """Pre-process, such as pre-pad and mod pad, so that the images can be divisible"""
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)

        if self.half:
            self.img = self.img.half()

        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')

        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        """Model inference with GPU monitoring"""
        if not self.img.is_cuda and str(self.device) != 'cpu':
            self.img = self.img.to(self.device)
        self.output = self.model(self.img)
        torch.cuda.synchronize()  # Ensure GPU operation is completed
        print_gpu_info()

    def tile_process(self):
        """Process image tiles with progress monitoring"""
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        total_tiles = tiles_x * tiles_y

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                if not input_tile.is_cuda and str(self.device) != 'cpu':
                    input_tile = input_tile.to(self.device)

                try:
                    with torch.no_grad():
                        output_tile = self.model(input_tile)
                        torch.cuda.synchronize()  # Ensure GPU operation is completed
                        print(f'\tTile {tile_idx}/{total_tiles} | GPU: {cuda.memory_allocated(0) / 1024**2:.1f}MB', end='\r')
                except RuntimeError as error:
                    print('Error', error)

                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                self.output[:, :, output_start_y:output_end_y,
                          output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                   output_start_x_tile:output_end_x_tile]

    def post_process(self):
        """Remove padding and extra scaling."""
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='realesrgan'):
        h_input, w_input = img.shape[0:2]
        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
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

        print(f'Processing {w_input}x{h_input} -> {int(w_input*self.scale)}x{int(h_input*self.scale)}')
        self.pre_process(img)

        if self.tile_size > 0:
            self.tile_process()
        else:
            self.process()

        output_img = self.post_process()
        output_img = output_img.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        if img_mode == 'L':
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        if img_mode == 'RGBA':
            if alpha_upsampler == 'realesrgan':
                self.pre_process(alpha)
                if self.tile_size > 0:
                    self.tile_process()
                else:
                    self.process()
                output_alpha = self.post_process()
                output_alpha = output_alpha.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output_alpha = np.transpose(output_alpha[[2, 1, 0], :, :], (1, 2, 0))
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (int(w_input * outscale), int(h_input * outscale)), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode

class PrefetchReader(threading.Thread):
    """Prefetch images."""

    def __init__(self, img_list, num_prefetch_queue):
        super().__init__()
        self.que = queue.Queue(num_prefetch_queue)
        self.img_list = img_list

    def run(self):
        for img_path in self.img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.que.put(img)
        self.que.put(None)

    def __next__(self):
        next_item = self.que.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self

class IOConsumer(threading.Thread):
    """IO consumer for saving images."""

    def __init__(self, opt, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.opt = opt

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break
            output = msg['output']
            save_path = msg['save_path']
            cv2.imwrite(save_path, output)
        print(f'IO worker {self.qid} is done.')