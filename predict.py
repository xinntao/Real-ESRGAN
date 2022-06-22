# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File
from basicsr.archs.rrdbnet_arch import RRDBNet
import os, cv2
import subprocess

subprocess.call(['python', '/src/setup.py', 'develop'])

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer
import tempfile

model_name = 'RealESRGAN_x4plus'
model_path = os.path.join('/root/.cache/realesrgan', model_name + ".pth")

class Predictor(BasePredictor):
    def setup(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=4
        ),
        face_enhance: bool = Input(description="Face enhance", default=True)
    ) -> Path:

        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        if face_enhance:
            face_enhancer = GFPGANer(
                model_path='/root/.cache/realesrgan/GFPGANv1.3.pth',
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler
            )
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = self.upsampler.enhance(img, outscale=scale)
        save_path=os.path.join(tempfile.mkdtemp(), "output.png")
        cv2.imwrite(save_path, output)
        return Path(save_path)