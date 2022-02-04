import os
from flask import Flask, request, render_template
import numpy as np
import cv2

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


upsampler = None
curr_dir = os.path.dirname(__file__)
res_path = os.path.join(curr_dir, 'static', 'uploads')
os.makedirs(res_path, exist_ok=True)

# start app
application = Flask(__name__)


def load_model():
    global upsampler
    model_name = 'RealESRGAN_x4plus'
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    tile = 960

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile)


def super_resolution(img, scale=3.0):
    sr, _ = upsampler.enhance(img, outscale=scale)
    return sr


@application.route('/')
def index():
    return render_template('index.html')


@application.route('/infer', methods=['POST'])
def inferring_image():
    if request.method != 'POST':
        return {}

    scale_class = request.form.get('scale_class')
    file = request.files['image']

    if not file:
        return render_template('index.html', label="No Files")

    ext = file.filename.split('.')[-1].lower()
    if ext not in set(['png', 'jpg', 'jpeg']):
        return render_template('index.html', label='not allowed file format')

    image = np.asarray(bytearray(file.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    sr = super_resolution(image, scale=float(scale_class.split('_')[-1]))
    filepath = os.path.join(res_path, file.filename.split('.')[0] + '_HR.' + ext)
    cv2.imwrite(filepath, sr)


    return render_template('index.html', uploaded=True, filepath=os.path.join('uploads', os.path.basename(filepath)))

if __name__ == '__main__':
    print(' [*] Starting application...')
    load_model()
    application.run(port=8080)