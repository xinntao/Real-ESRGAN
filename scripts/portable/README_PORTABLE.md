# Real-ESRGAN Windows portable package

This package uses the official `realesrgan-ncnn-vulkan` Windows release. It
contains the executable, Vulkan runtime DLLs, and the bundled NCNN models.
CUDA, PyTorch, Python, and an installer are not required.

## Quick start

1. Extract this ZIP to a writable folder.
2. Run `upscale_image.bat input.jpg output.png`.
3. Or run `upscale_folder.bat input_folder output_folder`.

The batch files use the bundled executable and models by relative path, so the
folder can be copied to a USB drive or another Windows computer unchanged.

## Direct command

```console
realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesrgan-x4plus
```

Available bundled models are `realesrgan-x4plus`,
`realesrgan-x4plus-anime`, and `realesr-animevideov3`. The default model is
`realesr-animevideov3`; use `-n realesrgan-x4plus` for general photographs.

If Vulkan initialization fails, install a current Intel, AMD, or NVIDIA
graphics driver with Vulkan support. For large images, use `-t tile_size`,
for example `-t 400`.

This NCNN executable may produce slightly different results from the Python
implementation and does not provide every Python option (for example,
`--outscale` or face enhancement).

## License and provenance

The executable and model files are redistributed from the official release
listed in `THIRD_PARTY_NOTICES.txt`. Keep `LICENSE-Real-ESRGAN.txt`,
`THIRD_PARTY_NOTICES.txt`, and the upstream `README_windows.md` when
redistributing this package.
