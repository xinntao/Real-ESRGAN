# OpenVINO Enable RealESGAN

## Environment Setup

```bash
conda create -n ov-RealESGAN python=3.12
conda activate ov-RealESGAN
```

## OpenVINO Quickstart

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/18582088138/OpenVINO-Real-ESRGAN.git
   cd OpenVINO-Real-ESRGAN
   ```

2. **Install Dependent Packages**
   - Install `BasicSR` (used for both training and inference):
     ```bash
     pip install basicsr
     ```
   - Install `facexlib` and `gfpgan` (used for face enhancement):
     ```bash
     pip install facexlib
     pip install gfpgan
     ```
   - Install `OpenVINO`:
     ```bash
     pip install openvino
     ```
   - Install other dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Set up the project:
     ```bash
     python setup.py develop
     ```

### Python Script Usage

You can use the X4 model for arbitrary output sizes with the `outscale` argument. The program performs a cheap resize operation after the Real-ESRGAN output.

#### Example Command
```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/0014.jpg -o outputs/ov -ov True --ov_device GPU
```

#### Command-Line Options

| Option              | Description                                                                 | Default Value |
|---------------------|-----------------------------------------------------------------------------|---------------|
| `-h`               | Show help information.                                                     | -             |
| `-i`, `--input`    | Input image or folder.                                                     | `inputs`      |
| `-o`, `--output`   | Output folder.                                                             | `results`     |
| `-n`, `--model_name` | Model name.                                                              | `RealESRGAN_x4plus` |
| `-s`, `--outscale` | Final upsampling scale of the image.                                       | `4`           |
| `--suffix`         | Suffix of the restored image.                                              | `out`         |
| `-t`, `--tile`     | Tile size, `0` for no tile during testing.                                 | `0`           |
| `--face_enhance`   | Use GFPGAN to enhance face.                                                | `False`       |
| `--fp32`           | Use `fp32` precision during inference.                                    | `fp16` (half precision) |
| `--ext`            | Image extension (`auto`, `jpg`, `png`). `auto` uses the same extension as inputs. | `auto`        |
| `-ov`, `--openvino` | Use OpenVINO to optimize performance.                                     | `True`        |
| `--ov_device`      | OpenVINO development device (`CPU`, `GPU`). `GPU` means uses Intel Core iGPU.    | `CPU`         |

## Result

![OpenVINO and Pytorch result compare](inputs/RealESGAN_result_compare.jpg)
