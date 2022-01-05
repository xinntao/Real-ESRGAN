import argparse
import os
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main():
    """Inference demo for Real-ESRGAN.
    It mainly for restoring anime videos.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input video')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus'
              'RealESRGANv2-anime-xsx2 | RealESRGANv2-animevideo-xsx2-nousm | RealESRGANv2-animevideo-xsx2'
              'RealESRGANv2-anime-xsx4 | RealESRGANv2-animevideo-xsx4-nousm | RealESRGANv2-animevideo-xsx4'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored video')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')

    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    args = parser.parse_args()

    # ---------------------- determine models according to model names ---------------------- #
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in [
        'RealESRGANv2-anime-xsx2', 'RealESRGANv2-animevideo-xsx2-nousm', 'RealESRGANv2-animevideo-xsx2'
    ]:  # x2 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
    elif args.model_name in [
        'RealESRGANv2-anime-xsx4', 'RealESRGANv2-animevideo-xsx4-nousm', 'RealESRGANv2-animevideo-xsx4'
    ]:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # ---------------------- determine model paths ---------------------- #
    model_path = os.path.join('experiments/pretrained_models', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {args.model_name} does not exist.')

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    os.makedirs(args.output, exist_ok=True)

    try:
        import ffmpeg
    except ImportError as e:
        print("please install ffmpeg-python package! The command line may be: pip3 install ffmpeg-python")
        raise e

    video_name = os.path.splitext(os.path.basename(args.input))[0]
    # get input video info
    probe = ffmpeg.probe(args.input)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    if args.fps is None:
        args.fps = eval(video_streams[0]['avg_frame_rate'])
    nb_frames = int(video_streams[0]['nb_frames'])
    width = video_streams[0]['width']
    height = video_streams[0]['height']

    pbar = tqdm(total=nb_frames, unit='frame', desc='inference')
    # set up frame decoder
    decoder = (ffmpeg.input(args.input)
               .output('pipe:', format='rawvideo', pix_fmt='rgb24', loglevel='warning')
               .run_async(pipe_stdout=True, cmd=args.ffmpeg_bin))

    video_save_path = os.path.join(args.output, f'{video_name}_{args.suffix}.mp4')
    out_width, out_height = int(width * args.outscale), int(height * args.outscale)
    # set up frame encoder
    video = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{out_width}x{out_height}',
                         framerate=args.fps)  # Specify frame rate for input that avoid warning
    audio = ffmpeg.input(args.input).audio
    encoder = (ffmpeg.output(video, audio, video_save_path,
                             pix_fmt='yuv420p', vcodec='libx264', loglevel='info', acodec='copy')
               .overwrite_output()
               .run_async(pipe_stdin=True, cmd=args.ffmpeg_bin))

    last_bytes = None
    last_output = None
    for idx in range(nb_frames):

        img_bytes = decoder.stdout.read(width * height * 3)  # 3 bytes for one pixel
        img = np.frombuffer(img_bytes, np.uint8).reshape([height, width, 3])

        if img_bytes == last_bytes:
            # if current frame equals last frame we needn't to inference
            encoder.stdin.write(last_output)
            continue
        last_bytes = img_bytes

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False,
                                                     paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            last_output = output.astype(np.uint8).tobytes()
            encoder.stdin.write(last_output)

        pbar.update(1)
        torch.cuda.synchronize()

    encoder.stdin.close()
    decoder.wait()
    encoder.wait()
    pbar.close()


if __name__ == '__main__':
    main()
