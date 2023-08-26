# RealESRGAN

We have provided the following models:

1. realesr-animevideov3 (default)
2. realesrgan-x4plus
3. realesrgan-x4plus-anime

Command:

1. ./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png
2. ./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesr-animevideov3
3. ./realesrgan-ncnn-vulkan.exe -i input_folder -o outputfolder -n realesr-animevideov3 -s 2 -f jpg
4. ./realesrgan-ncnn-vulkan.exe -i input_folder -o outputfolder -n realesr-animevideov3 -s 4 -f jpg


Commands for enhancing anime videos:

1. Use ffmpeg to extract frames from a video (Remember to create the folder `tmp_frames` ahead)

    ffmpeg -i onepiece_demo.mp4 -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 tmp_frames/frame%08d.jpg

2. Inference with Real-ESRGAN executable file (Remember to create the folder `out_frames` ahead)

    ./realesrgan-ncnn-vulkan.exe -i tmp_frames -o out_frames -n realesr-animevideov3 -s 2 -f jpg

3. Merge the enhanced frames back into a video

    ffmpeg -i out_frames/frame%08d.jpg -i onepiece_demo.mp4 -map 0:v:0 -map 1:a:0 -c:a copy -c:v libx264 -r 23.98 -pix_fmt yuv420p output_w_audio.mp4

------------------------

GitHub: https://github.com/xinntao/Real-ESRGAN/
Paper: https://arxiv.org/abs/2107.10833

------------------------

This executable file is **portable** and includes all the binaries and models required. No CUDA or PyTorch environment is needed.

Note that it may introduce block inconsistency (and also generate slightly different results from the PyTorch implementation), because this executable file first crops the input image into several tiles, and then processes them separately, finally stitches together.

This executable file is based on the wonderful [Tencent/ncnn](https://github.com/Tencent/ncnn) and [realsr-ncnn-vulkan](https://github.com/nihui/realsr-ncnn-vulkan) by [nihui](https://github.com/nihui).
