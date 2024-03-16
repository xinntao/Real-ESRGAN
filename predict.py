# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import cv2
import os
import tempfile
from basicsr.archs.rrdbnet_arch import RRDBNet
from cog import BasePredictor, Input, Path
import sentry_sdk

from gfpgan import GFPGANer
from realesrgan import RealESRGANer

MODEL_NAME = "RealESRGAN_x4plus"
ESRGAN_PATH = os.path.join("/root/.cache/realesrgan", MODEL_NAME + ".pth")
GFPGAN_PATH = "/root/.cache/realesrgan/GFPGANv1.3.pth"


class Predictor(BasePredictor):
    def setup(self):
        sentry_dsn = os.getenv("SENTRY_DSN", "no-op")
        if sentry_dsn != "no-op":
            sentry_sdk.init(
                dsn=sentry_dsn,
            )
            print("sentry init")

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
            model_path=ESRGAN_PATH,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        self.face_enhancer = GFPGANer(
            model_path=GFPGAN_PATH,
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
        try:
            img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)

            if face_enhance:
                print("running with face enhancement")
                self.face_enhancer.upscale = scale
                _, _, output = self.face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                print("running without face enhancement!!")
                print(img.shape)
                output, _ = self.upsampler.enhance(img, outscale=scale)
            save_path = os.path.join(tempfile.mkdtemp(), "output.png")
            cv2.imwrite(save_path, output)

        except Exception as e:
            if img is not None:
                img_shape = img.shape
                print(f"capturing {img_shape} for logging")
                del image
            sentry_sdk.capture_exception(e)
            raise e
        print("beep")
        return Path(save_path)
