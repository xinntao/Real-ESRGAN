# import os
# from pathlib import Path
# import shutil
# import subprocess
# import re

# def get_frame_rate(video_path):
#     # Run ffmpeg to get video info
#     result = subprocess.run(
#         ['ffmpeg', '-i', str(video_path)],
#         stderr=subprocess.PIPE,
#         stdout=subprocess.PIPE,
#         text=True
#     )
#     output = result.stderr
#     # Search for frame rate in ffmpeg output
#     match = re.search(r'(\d+(?:\.\d+)?)\s*fps', output)
#     if match:
#         return float(match.group(1))
#     else:
#         print("Could not determine frame rate.")
#         exit(1)

# # check ffmpeg on path
# if os.system('where ffmpeg >nul 2>nul') != 0:
#     print('ffmpeg not found')
#     exit(1)

# import sys

# def run_or_exit(cmd, **kwargs):
#     result = subprocess.run(cmd, **kwargs)
#     if result.returncode != 0:
#         print(f"Command failed: {' '.join(str(x) for x in cmd)}")
#         shutil.rmtree(TEMP_PATH, ignore_errors=True)
#         sys.exit(1)

# movie_path = Path(input('Enter the path to the movie: '))
# TEMP_PATH = Path('temp')
# if TEMP_PATH.exists():
#     shutil.rmtree(TEMP_PATH)
# os.makedirs(TEMP_PATH, exist_ok=True)
# if movie_path.exists():
#     shutil.copy(movie_path, TEMP_PATH / 'movie.mp4')
# else:
#     print("Movie file does not exist.")
#     sys.exit(1)

# os.makedirs(TEMP_PATH / "temp_frames", exist_ok=True)
# run_or_exit([
#     'ffmpeg', '-hwaccel', 'cuda', '-i', str(TEMP_PATH / "movie.mp4"),
#     '-qscale:v', '1', '-qmin', '1', '-qmax', '1', '-fps_mode', 'vfr',
#     str(TEMP_PATH / "temp_frames" / "frame%08d.jpg")
# ])

# frame_rate = get_frame_rate(TEMP_PATH / "movie.mp4")
# print(f'Frame rate: {frame_rate}')

# os.makedirs(TEMP_PATH / "realesrgan", exist_ok=True)
# run_or_exit([
#     'uv', 'run', 'inference_realesrgan.py',
#     '-i', str(TEMP_PATH / "temp_frames"),
#     '-o', str(TEMP_PATH / "realesrgan"),
#     '-n', 'Real-ESRGAN-General-x4v3',
#     '-s', '4'
# ])

# run_or_exit([
#     'ffmpeg', '-r', str(frame_rate),
#     '-i', str(TEMP_PATH / "realesrgan" / "frame%08d.jpg"),
#     '-c:v', 'av1_nvenc', '-rc', 'vbr', '-cq', '28', '-b:v', '0',
#     '-r', str(frame_rate), '-pix_fmt', 'yuv420p',
#     f"{movie_path.stem}_realesrgan.mp4"
# ])

# shutil.rmtree(TEMP_PATH)

import subprocess
import shutil
from pathlib import Path
import sys
import os

movie_path = Path(input('Enter the path to the movie: '))
TEMP_PATH = Path('temp')
if TEMP_PATH.exists():
    shutil.rmtree(TEMP_PATH)
os.makedirs(TEMP_PATH, exist_ok=True)
if movie_path.exists():
    shutil.copy(movie_path, TEMP_PATH / 'movie.mp4')
else:
    print("Movie file does not exist.")
    sys.exit(1)
os.makedirs(TEMP_PATH / "realesrgan", exist_ok=True)
subprocess.run(["uv", "run", "inference_realesrgan_video.py", "-i", TEMP_PATH / "movie.mp4", "-o", movie_path.absolute().as_posix() + "_realesrgan", "-n", "realesr-general-x4v3"])