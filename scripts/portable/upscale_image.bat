@echo off
setlocal
if "%~2"=="" (
  echo Usage: %~nx0 input-image output-image [model-name]
  exit /b 2
)
set "MODEL=%~3"
if "%MODEL%"=="" set "MODEL=realesrgan-x4plus"
"%~dp0realesrgan-ncnn-vulkan.exe" -i "%~1" -o "%~2" -m "%~dp0models" -n "%MODEL%"
exit /b %ERRORLEVEL%
