# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import subprocess
import time
import cv2
import os
import tempfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from cog import BasePredictor, Input, Path, emit_metric

from gfpgan import GFPGANer
from realesrgan import RealESRGANer

WEIGHTS_URL = "https://weights.replicate.delivery/default/official-models/tencent/real-esrgan/real-esrgan-models.tar"
EXTRA_URL = "https://weights.replicate.delivery/default/official-models/tencent/real-esrgan/esrgan-extra-models.tar"

MODEL_FOLDER = "/src/weights/esrgan/"
GFPGAN_FOLDER = "/src/gfpgan/weights/"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        if not os.path.exists(MODEL_FOLDER):
            download_weights(WEIGHTS_URL, MODEL_FOLDER)
        if not os.path.exists(GFPGAN_FOLDER):
            download_weights(EXTRA_URL, GFPGAN_FOLDER)

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=os.path.join(MODEL_FOLDER, "RealESRGAN_x4plus.pth"),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        self.face_enhancer = GFPGANer(
            model_path=os.path.join(MODEL_FOLDER, "GFPGANv1.3.pth"),
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
        )

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=10, default=4
        ),
        face_enhance: bool = Input(
            description="Run GFPGAN face enhancement along with upscaling",
            default=False,
        ),
    ) -> Path:
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)

        if face_enhance:
            print("running with face enhancement")
            self.face_enhancer.upscale = scale
            _, _, output = self.face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            print("running without face enhancement")
            output, _ = self.upsampler.enhance(img, outscale=scale)
        save_path =  "output.png"

        cv2.imwrite(save_path, output)
        emit_metric("scale", scale)
        emit_metric("original_height", img.shape[0])
        emit_metric("original_width", img.shape[1])
        emit_metric("face_enhance", face_enhance)
        return Path(save_path)
