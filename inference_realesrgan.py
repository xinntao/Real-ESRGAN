import argparse
import cv2
import glob
import os

from realesrgan import RealESRGANer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '--model_path',
        type=str,
        default='experiments/pretrained_models/RealESRGAN_x4plus.pth',
        help='Path to the pre-trained model')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--netscale', type=int, default=4, help='Upsample scale factor of the network')
    parser.add_argument('--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()

    upsampler = RealESRGANer(
        scale=args.netscale,
        model_path=args.model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[0:2]
        if max(h, w) > 1000 and args.netscale == 4:
            print('WARNING: The input image is large, try X2 model for better performace.')
        if max(h, w) < 500 and args.netscale == 2:
            print('WARNING: The input image is small, try X4 model for better performace.')

        try:
            output, img_mode = upsampler.enhance(img, outscale=args.outscale)
        except Exception as error:
            print('Error', error)
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)


if __name__ == '__main__':
    main()
