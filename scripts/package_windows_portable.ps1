param(
    [string]$OutputDirectory = 'dist',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$version = '20220424'
$assetName = "realesrgan-ncnn-vulkan-$version-windows.zip"
$assetUrl = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/$assetName"
$assetSha256 = 'ABC02804E17982A3BE33675E4D471E91EA374E65B70167ABC09E31ACB412802D'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputRoot = $OutputDirectory
if (-not [IO.Path]::IsPathRooted($outputRoot)) {
    $outputRoot = Join-Path $repoRoot $outputRoot
}
$outputRoot = [IO.Path]::GetFullPath($outputRoot)
$outputRoot = (New-Item -ItemType Directory -Force -Path $outputRoot).FullName
$downloadPath = Join-Path $outputRoot $assetName
$extractPath = Join-Path $outputRoot "realesrgan-ncnn-vulkan-$version-windows"
$packagePath = Join-Path $outputRoot "Real-ESRGAN-Windows-Portable-$version"
$packageZip = "$packagePath.zip"

if ((Test-Path $packagePath) -or (Test-Path $packageZip)) {
    if (-not $Force) {
        throw "Output already exists. Use -Force to replace it: $packagePath"
    }
    Remove-Item -LiteralPath $packagePath, $packageZip -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path $downloadPath)) {
    Write-Host "Downloading official release asset..."
    Invoke-WebRequest -Uri $assetUrl -OutFile $downloadPath -UseBasicParsing
}

$actualHash = (Get-FileHash -LiteralPath $downloadPath -Algorithm SHA256).Hash
if ($actualHash -ne $assetSha256) {
    throw "SHA256 mismatch for $downloadPath. Expected $assetSha256, got $actualHash."
}

if (Test-Path $extractPath) {
    Remove-Item -LiteralPath $extractPath -Recurse -Force
}
Expand-Archive -LiteralPath $downloadPath -DestinationPath $extractPath
Move-Item -LiteralPath $extractPath -Destination $packagePath

Copy-Item -LiteralPath (Join-Path $repoRoot 'LICENSE') `
    -Destination (Join-Path $packagePath 'LICENSE-Real-ESRGAN.txt')
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\portable\README_PORTABLE.md') `
    -Destination (Join-Path $packagePath 'README_PORTABLE.md')
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\portable\upscale_image.bat') `
    -Destination (Join-Path $packagePath 'upscale_image.bat')
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\portable\upscale_folder.bat') `
    -Destination (Join-Path $packagePath 'upscale_folder.bat')

$notice = @"
This package redistributes the official Real-ESRGAN ncnn Vulkan Windows
release asset without modifying its executable, DLLs, or model files:
$assetUrl

Real-ESRGAN source and license: https://github.com/xinntao/Real-ESRGAN
ncnn source and license: https://github.com/Tencent/ncnn
realsr-ncnn-vulkan source and license: https://github.com/nihui/realsr-ncnn-vulkan

See LICENSE-Real-ESRGAN.txt for the repository license. The upstream
release README and attribution remain included in this package.
"@
Set-Content -LiteralPath (Join-Path $packagePath 'THIRD_PARTY_NOTICES.txt') `
    -Value $notice -Encoding UTF8

Compress-Archive -Path (Join-Path $packagePath '*') -DestinationPath $packageZip -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $packageZip -Algorithm SHA256).Hash
Set-Content -LiteralPath "$packageZip.sha256" -Value "$zipHash  $(Split-Path $packageZip -Leaf)" -Encoding ASCII

Remove-Item -LiteralPath $packagePath -Recurse -Force
Write-Host "Created $packageZip"
Write-Host "SHA256 $zipHash"
