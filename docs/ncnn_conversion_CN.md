# 关于转换为NCNN模型的说明

[English](ncnn_conversion.md) **|** 简体中文

1. 用`scripts/pytorch2onnx.py`转换为onnx模型。记住要相应地修改代码
1. 将onnx模型转换为ncnn模型
    1. `cd ncnn-master\ncnn\build\toolsonnx`。
    1. `onnx2ncnn.exe realesrgan-x4.onnx realesrgan-x4-raw.param realesrgan-x4-raw.bin`。
1. 优化ncnn模型
    1. fp16模式
        1. `cd ncnn-master\ncnn\build\tools`。
        1. `ncnnoptimize.exe realesrgan-x4-raw.param realesrgan-x4-raw.bin realesrgan-x4.param realesrgan-x4.bin 1`.
1. 修改`realesrgan-x4.param`中的blob名称：`data`和`output`。
