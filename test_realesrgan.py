import os
import urllib.request as request
import cv2
import numpy as np
import unittest
import time

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

output = 'results_render'

# os.makedirs(input, exist_ok=True)
os.makedirs(output, exist_ok=True)
resolutions = {
    "low": [550, 960],
    "std": [1280],
    "fhd": [1920],
    "qhd": [2560],
    "4k": [4000, 3840],
    "8k": [8000, 7680]
}


class Test_realesrgan(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

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
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile)

    def get_resolution(self, width):
        return [list(resolutions.keys())[i] for i, val in enumerate(resolutions.values()) if width in val][0]

    def url_to_image(self, url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image

    def save_results(self, testId, sr, lr, hr=None):
        save_path = os.path.join(output, f'{testId}')
        cv2.imwrite(f"{save_path}_LR({self.get_resolution(lr.shape[1])}).png", lr)
        if hr is not None:
            cv2.imwrite(f"{save_path}_HR({self.get_resolution(hr.shape[1])}).png", hr)
        cv2.imwrite(f"{save_path}_SR({self.get_resolution(sr.shape[1])})_x{sr.shape[1] / lr.shape[1]}.png", sr)

    def save_downscale_results(self, testId, sr_down, lr=None):
        save_path = os.path.join(output, f'{testId}')
        if lr is not None:
            cv2.imwrite(f"{save_path}_LR({self.get_resolution(lr.shape[1])}).png", lr)
        cv2.imwrite(f"{save_path}_SR({self.get_resolution(sr_down.shape[1])})_down.png", sr_down)

    def downscale(self, image, method='bicubic', scale=0.25):
        if method == 'bicubic':
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=interpolation)

    def test_a_realesrgan(self):
        testId = 'X5G8IYXB1A2FB9205824FC3'
        lr_url = 'https://resources.archisketch.com/images/X5G8IYXB1A2FB9205824FC3/1280x960/X5G8IYXB1A2FB9205824FC3.png'
        hr_url = 'https://resources.archisketch.com/images/X5G8MVK31DFA7F290A940C1/4000x3000/X5G8MVK31DFA7F290A940C1.png'
        lr = self.url_to_image(lr_url)
        hr = self.url_to_image(hr_url)

        sr, _ = self.upsampler.enhance(lr, outscale=hr.shape[1]/lr.shape[1])
        self.save_results(testId, sr, lr, hr)

    def test_b_realesrgan(self):
        testId = 'X4tunRaB8879C8FBD06468F'
        lr_url = 'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X4tunRaB8879C8FBD06468F/550xAUTO/X4tunRaB8879C8FBD06468F.png'
        hr_url = 'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X4tunRaB8879C8FBD06468F/3840x2160/X4tunRaB8879C8FBD06468F.png'
        lr = self.url_to_image(lr_url)
        hr = self.url_to_image(hr_url)

        sr, _ = self.upsampler.enhance(lr, outscale=hr.shape[1]/lr.shape[1])
        self.save_results(testId, sr, lr, hr)

    def test_c_realesrgan(self):
        testId = 'X4sDhDD16A8A7592EC842FE'
        lr_url = ''
        hr_url = 'https://resources.archisketch.com/images/X4sDhDD16A8A7592EC842FE/3840x2160/X4sDhDD16A8A7592EC842FE.png'
        hr = self.url_to_image(hr_url)
        lr = self.downscale(hr)

        sr, _ = self.upsampler.enhance(lr, outscale=hr.shape[1]/lr.shape[1])
        self.save_results(testId, sr, lr, hr)

    def test_d_realesrgan(self):
        testId = 'X5RUPVA679DE247E6624DDC'
        lr_urls = ['https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUPVA679DE247E6624DDC/1280x720/X5RUPVA679DE247E6624DDC.png',
                'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUuVWC8AB408CC27240D0/1920x1080/X5RUuVWC8AB408CC27240D0.png',
                'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUykd3C6BC1ABF5244C07/2560x1440/X5RUykd3C6BC1ABF5244C07.png']
        hr_url = 'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUqFGE751F0583D6040E8/3840x2160/X5RUqFGE751F0583D6040E8.png'

        for lr_url in lr_urls:
            hr = self.url_to_image(hr_url)
            lr = self.url_to_image(lr_url)

            sr, _ = self.upsampler.enhance(lr, outscale=hr.shape[1]/lr.shape[1])
            self.save_results(testId, sr, lr, hr)

    def test_e_realesrgan(self):
        testId = 'X6QfDPQ8501E1AC4CD749BB'
        lr_urls = ['https://resources.archisketch.com/images/X6QfDPQ8501E1AC4CD749BB/1280x720/X6QfDPQ8501E1AC4CD749BB.png',
                'https://resources.archisketch.com/images/X6QfHzTC867AEEB593248F9/1920x1080/X6QfHzTC867AEEB593248F9.png',
                'https://resources.archisketch.com/images/X6QfJqG76059C82C2A64FC9/2560x1440/X6QfJqG76059C82C2A64FC9.png']
        hr_url = 'https://resources.archisketch.com/images/X6QfF999365B8A1DF5B4B1A/3840x2160/X6QfF999365B8A1DF5B4B1A.png'

        for lr_url in lr_urls:
            hr = self.url_to_image(hr_url)
            lr = self.url_to_image(lr_url)

            sr, _ = self.upsampler.enhance(lr, outscale=hr.shape[1]/lr.shape[1])
            self.save_results(testId, sr, lr, hr)

    def test_f_8k(self):
        testIds = [
            'X6QfDPQ8501E1AC4CD749BB',
            'X5RUPVA679DE247E6624DDC',
            'X4sDhDD16A8A7592EC842FE',
            'X4tunRaB8879C8FBD06468F',
            'X5G8IYXB1A2FB9205824FC3'
        ]
        lr_urls = [
            'https://resources.archisketch.com/images/X6QfF999365B8A1DF5B4B1A/3840x2160/X6QfF999365B8A1DF5B4B1A.png',
            'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUqFGE751F0583D6040E8/3840x2160/X5RUqFGE751F0583D6040E8.png',
            'https://resources.archisketch.com/images/X4sDhDD16A8A7592EC842FE/3840x2160/X4sDhDD16A8A7592EC842FE.png',
            'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X4tunRaB8879C8FBD06468F/3840x2160/X4tunRaB8879C8FBD06468F.png',
            'https://resources.archisketch.com/images/X5G8MVK31DFA7F290A940C1/4000x3000/X5G8MVK31DFA7F290A940C1.png'
        ]

        for i, testId in enumerate(testIds):
            lr = self.url_to_image(lr_urls[i])
            sr, _ = self.upsampler.enhance(lr, outscale=2.0)
            self.save_results(testId, sr, lr)

    def test_g_processing_time(self):
        low_url = 'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X4tunRaB8879C8FBD06468F/550xAUTO/X4tunRaB8879C8FBD06468F.png'
        std_url = 'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUPVA679DE247E6624DDC/1280x720/X5RUPVA679DE247E6624DDC.png'
        fhd_url = 'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUuVWC8AB408CC27240D0/1920x1080/X5RUuVWC8AB408CC27240D0.png'
        qhd_url = 'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUykd3C6BC1ABF5244C07/2560x1440/X5RUykd3C6BC1ABF5244C07.png'
        k4_url = 'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUqFGE751F0583D6040E8/3840x2160/X5RUqFGE751F0583D6040E8.png'

        low = self.url_to_image(low_url)
        std = self.url_to_image(std_url)
        fhd = self.url_to_image(fhd_url)
        qhd = self.url_to_image(qhd_url)
        k4 = self.url_to_image(k4_url)

        max_num = 10

        for img in [low, std, fhd, qhd]:
            num = 0
            times = 0
            while num != max_num:
                start = time.time()
                sr, _ = self.upsampler.enhance(img, outscale=3840/img.shape[1])
                # print(time.time() - start)
                times += (time.time() - start)
                num += 1
                time.sleep(1)
            print(f'processing time ({self.get_resolution(img.shape[1])}) x{3840/img.shape[1]}-> 4k : {times/max_num}sec')

        # 4k -> 8k
        num = 0
        times = 0
        while max_num != max_num:
            start = time.time()
            sr, _ = self.upsampler.enhance(k4, outscale=2.0)
            # print(time.time() - start)
            times += (time.time() - start)
            num += 1
            time.sleep(1)
        print(f'processing time ({self.get_resolution(k4.shape[1])}) x{2}-> 8k : {times/max_num}sec')

    def test_h_8k_downscale(self):
        testIds = [
            'X6QfDPQ8501E1AC4CD749BB',
            'X5RUPVA679DE247E6624DDC',
            'X4sDhDD16A8A7592EC842FE',
            'X4tunRaB8879C8FBD06468F',
            'X5G8IYXB1A2FB9205824FC3'
        ]
        lr_urls = [
            'https://resources.archisketch.com/images/X6QfF999365B8A1DF5B4B1A/3840x2160/X6QfF999365B8A1DF5B4B1A.png',
            'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUqFGE751F0583D6040E8/3840x2160/X5RUqFGE751F0583D6040E8.png',
            'https://resources.archisketch.com/images/X4sDhDD16A8A7592EC842FE/3840x2160/X4sDhDD16A8A7592EC842FE.png',
            'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X4tunRaB8879C8FBD06468F/3840x2160/X4tunRaB8879C8FBD06468F.png',
            'https://resources.archisketch.com/images/X5G8MVK31DFA7F290A940C1/4000x3000/X5G8MVK31DFA7F290A940C1.png'
        ]

        for i, lr_url in enumerate(lr_urls):
            lr = self.url_to_image(lr_url)
            sr, _ = self.upsampler.enhance(lr, outscale=2.0)
            sr_down = self.downscale(sr, scale=0.5)
            self.save_downscale_results(testIds[i], sr_down)

    def test_i_4k_x1scale(self):
        testIds = [
            'X6QfDPQ8501E1AC4CD749BB',
            'X5RUPVA679DE247E6624DDC',
            'X4sDhDD16A8A7592EC842FE',
            'X4tunRaB8879C8FBD06468F',
            'X5G8IYXB1A2FB9205824FC3'
        ]
        lr_urls = [
            'https://resources.archisketch.com/images/X6QfF999365B8A1DF5B4B1A/3840x2160/X6QfF999365B8A1DF5B4B1A.png',
            'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X5RUqFGE751F0583D6040E8/3840x2160/X5RUqFGE751F0583D6040E8.png',
            'https://resources.archisketch.com/images/X4sDhDD16A8A7592EC842FE/3840x2160/X4sDhDD16A8A7592EC842FE.png',
            'https://archisketch-resources.s3.ap-northeast-2.amazonaws.com/images/X4tunRaB8879C8FBD06468F/3840x2160/X4tunRaB8879C8FBD06468F.png',
            'https://resources.archisketch.com/images/X5G8MVK31DFA7F290A940C1/4000x3000/X5G8MVK31DFA7F290A940C1.png'
        ]

        for i, lr_url in enumerate(lr_urls):
            lr = self.url_to_image(lr_url)
            sr, _ = self.upsampler.enhance(lr, outscale=1.0)
            self.save_results(testIds[i], sr, lr)


if __name__ == '__main__':
    unittest.main()
