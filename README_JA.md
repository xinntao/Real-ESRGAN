<p align="center">
  <img src="assets/realesrgan_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a> | <a href="README_JA.md">日本語</a></b></div>

<div align="center">

👀[**デモ**](#-demos-videos) **|** 🚩[**アップデート**](#-updates) **|** ⚡[**使用法**](#-quick-inference) **|** 🏰[**Model Zoo**](docs/model_zoo.md) **|** 🔧[Install](#-dependencies-and-installation)  **|** 💻[Train](docs/Training.md) **|** ❓[FAQ](docs/FAQ.md) **|** 🎨[Contribution](docs/CONTRIBUTING.md)

[![download](https://img.shields.io/github/downloads/xinntao/Real-ESRGAN/total.svg)](https://github.com/xinntao/Real-ESRGAN/releases)
[![PyPI](https://img.shields.io/pypi/v/realesrgan)](https://pypi.org/project/realesrgan/)
[![Open issue](https://img.shields.io/github/issues/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![Closed issue](https://img.shields.io/github/issues-closed/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN/issues)
[![LICENSE](https://img.shields.io/github/license/xinntao/Real-ESRGAN.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
[![python lint](https://github.com/xinntao/Real-ESRGAN/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/Real-ESRGAN/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/Real-ESRGAN/blob/master/.github/workflows/publish-pip.yml)

</div>

🔥 **AnimeVideo-v3 model (动漫视频小模型)**。 [[*アニメ動画モデル*](docs/anime_video_model.md)]と[[*比較*](docs/anime_comparisons.md)]を参照してください。<br>
🔥 **RealESRGAN_x4plus_anime_6B** アニメ画像用に **(动漫插图模型)**。詳しくは[[*anime_model*](docs/anime_model.md)]をご覧ください

<!-- 1. You can try in our website: [ARC Demo](https://arc.tencent.com/en/ai-demos/imgRestore) (now only support RealESRGAN_x4plus_anime_6B) -->
1. :boom: **アップデート** オンラインレプリカデモ: [![Replicate](https://img.shields.io/static/v1?label=Demo&message=Replicate&color=blue)](https://replicate.com/xinntao/realesrgan)
2. Real-ESRGAN用オンラインColabデモ: [![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing) **|** Real-ESRGAN用のオンラインColabデモ (**アニメ映像**): [![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1yNl9ORUxxlL4N0keJa2SEPB61imPQd1B?usp=sharing)
3. ポータブル[Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip) **Intel/AMD/Nvidia GPU用の実行可能ファイル**。詳しくは[こちら](#portable-executable-files-ncnn)にあります。ncnnの実装は[Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)にあります
<!-- 1. You can watch enhanced animations in [Tencent Video](https://v.qq.com/s/topic/v_child/render/fC4iyCAM.html). 欢迎观看[腾讯视频动漫修复](https://v.qq.com/s/topic/v_child/render/fC4iyCAM.html) -->

Real-ESRGANは、 **一般的な画像・映像の修復のための実用的なアルゴリズム**を開発することを目的としています。<br>
我々は、強力なESRGANを、純粋な合成データで学習する実用的な復元アプリケーション（すなわち、Real-ESRGAN）に拡張しました。

🌌 貴重なご意見・ご感想をありがとうございました。いただいたご意見は、[feedback.md](docs/feedback.md)にまとめています。

---

Real-ESRGANが役に立ったら、このリポジトリを⭐にしたり、友達に薦めたりしてください😊<br>
その他の推奨プロジェクト:<br>
▶️ [GFPGAN](https://github.com/TencentARC/GFPGAN): 実世界における顔面修復のための実用的なアルゴリズム <br>
▶️ [BasicSR](https://github.com/xinntao/BasicSR): オープンソースの画像・映像修復ツールボックス<br>
▶️ [facexlib](https://github.com/xinntao/facexlib): 便利な顔認識機能を提供するコレクションです。<br>
▶️ [HandyView](https://github.com/xinntao/HandyView): 閲覧や比較に便利なPyQt5ベースの画像ビューア <br>
▶️ [HandyFigure](https://github.com/xinntao/HandyFigure): 紙製フィギュアのオープンソース化 <br>

---

### 📖 Real-ESRGAN: 純粋な合成データを用いた実世界のブラインド超解像の学習

> [[Paper](https://arxiv.org/abs/2107.10833)] &emsp; [[YouTube Video](https://www.youtube.com/watch?v=fxHWoDSSvSc)] &emsp; [[B站讲解](https://www.bilibili.com/video/BV1H34y1m7sS/)] &emsp; [[Poster](https://xinntao.github.io/projects/RealESRGAN_src/RealESRGAN_poster.pdf)] &emsp; [[PPT slides](https://docs.google.com/presentation/d/1QtW6Iy8rm8rGLsJ0Ldti6kP-7Qyzy6XL/edit?usp=sharing&ouid=109799856763657548160&rtpof=true&sd=true)]<br>
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en) <br>
> [Tencent ARC Lab](https://arc.tencent.com/en/ai-demos/imgRestore); 中国科学院深圳先進技術研究院

<p align="center">
  <img src="assets/teaser.jpg">
</p>

---

<!---------------------------------- Updates --------------------------->
## 🚩 アップデート

- ✅ **realesr-general-x4v3**モデルを追加 - 一般的なシーンに対応した小さなモデルです。また、ノイズのバランスをとるための **-dn**オプションもサポートしています(滑らかすぎる結果を避けることができます)。**-dn**はdenoising strengthの略です。
- ✅ RealESRGAN AnimeVideo-v3**モデルを更新しました。詳しくは、[アニメビデオモデル](docs/anime_video_model.md)と[比較](docs/anime_comparisons.md) を参照してください。
- ✅ アニメビデオ用の小さなモデルを追加します。詳細は[アニメビデオモデル](docs/anime_video_model.md)を参照してください。
- ✅ ncnnの実装 [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) を追加。
- ✅ [*RealESRGAN_x4plus_anime_6B.pth*](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)を追加。モデルサイズがかなり小さい**アニメ**画像用に最適化されています。詳細と [waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)との比較は[**anime_model.md**](docs/anime_model.md) にあります
- ✅ 自分のデータやペアデータでのFinetuningをサポートする(*すなわち*, finetuning ESRGAN)。[ここ](docs/Training.md#Finetune-Real-ESRGAN-on-your-own-dataset)を参照
- ✅ [GFPGAN](https://github.com/TencentARC/GFPGAN)を統合し、**face enhancement**に対応。
- ✅ [Gradio](https://github.com/gradio-app/gradio)で[Huggingface Spaces](https://huggingface.co/spaces)に統合。[Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Real-ESRGAN)をご覧ください。ありがとうございます[@AK391](https://github.com/AK391)
- ✅ `--outscale`による任意のスケールをサポート(実際には`LANCZOS4`でさらに出力をリサイズします)。*RealESRGAN_x2plus.pth*モデルを追加しました。
- ✅ [推論コード](inference_realesrgan.py)は以下のものをサポートしています。1) **tile** オプション; 2) **alpha channel** を持つ画像; 3) **gray** 画像; 4) **16-bit** 画像をサポートしています。
- ✅ トレーニングコードを公開しました。詳細なガイドは[Training.md](docs/Training.md)に掲載されています。

---

<!---------------------------------- Demo videos --------------------------->
## 👀 デモ映像

#### Bilibili

- [大闹天宫片段](https://www.bilibili.com/video/BV1ja41117zb)
- [Anime dance cut 动漫魔性舞蹈](https://www.bilibili.com/video/BV1wY4y1L7hT/)
- [海贼王片段](https://www.bilibili.com/video/BV1i3411L7Gy/)

#### YouTube

## 🔧 依存関係とインストール

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)

### インストール

1. リポジトリのClone

    ```bash
    git clone https://github.com/xinntao/Real-ESRGAN.git
    cd Real-ESRGAN
    ```

2. 依存パッケージのインストール

    ```bash
    # basicsrのインストール - https://github.com/xinntao/BasicSR
    # 学習と推論の両方にBasicSRを使用します
    pip install basicsr
    # facexlibとgfpganは顔補正用です
    pip install facexlib
    pip install gfpgan
    pip install -r requirements.txt
    python setup.py develop
    ```

---

## ⚡ クイック推論

Real-ESRGANの推論方法は、通常3通りあります。

1. [オンライン推論](#online-inference)
2. [ポータブル実行ファイル(NCNN)](#portable-executable-files-ncnn)
3. [Pythonスクリプト](#python-script)

### オンライン推論

1. 私達のホームページで試用できます: [ARC Demo](https://arc.tencent.com/en/ai-demos/imgRestore) (RealESRGAN_x4plus_anime_6Bのみをサポートするようになりました)
2. Real-ESRGANのための[Colab Demo](https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing) **|** Real-ESRGANのための[Colab Demo](https://colab.research.google.com/drive/1yNl9ORUxxlL4N0keJa2SEPB61imPQd1B?usp=sharing) (**アニメ動画**)。

### ポータブル実行可能ファイル(NCNN)

それぞれ[Windows](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) / [Linux](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip) / [MacOS](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip) 用をダウンロードできます **Intel/AMD/Nvidia GPU用実行ファイル**です。

この実行ファイルは **ポータブル** で、必要なバイナリやモデルをすべて含んでいます。CUDAやPyTorchの環境は必要ありません。<br>

以下のコマンドを実行するだけです（Windowsの例、詳細は各実行ファイルのREADME.mdに記載されています）:

```bash
./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n model_name
```

5つのモデルを用意いたしました:

1. realesrgan-x4plus  (デフォルト)
2. realesrnet-x4plus
3. realesrgan-x4plus-anime (アニメ映像に最適化、小型モデル)
4. realesr-animevideov3 (アニメーション映像)

他のモデルには`-n`引数を使うことができます。例えば、`./realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png -n realesrnet-x4plus`のようにします

#### ポータブル実行ファイルの使用方法

1. 詳しくは[Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan#computer-usages)を参照してください。
2. Pythonスクリプト`inference_realesrgan.py`のように、すべての関数（`outscale`など）をサポートしているわけではないことに注意してください。

```console
Usage: realesrgan-ncnn-vulkan.exe -i infile -o outfile [options]...

  -h                   show this help
  -i input-path        input image path (jpg/png/webp) or directory
  -o output-path       output image path (jpg/png/webp) or directory
  -s scale             upscale ratio (can be 2, 3, 4. default=4)
  -t tile-size         tile size (>=32/0=auto, default=0) can be 0,0,0 for multi-gpu
  -m model-path        folder path to the pre-trained models. default=models
  -n model-name        model name (default=realesr-animevideov3, can be realesr-animevideov3 | realesrgan-x4plus | realesrgan-x4plus-anime | realesrnet-x4plus)
  -g gpu-id            gpu device to use (default=auto) can be 0,1,2 for multi-gpu
  -j load:proc:save    thread count for load/proc/save (default=1:2:2) can be 1:2,2,2:2 for multi-gpu
  -x                   enable tta mode"
  -f format            output image format (jpg/png/webp, default=ext/png)
  -v                   verbose output
```

この実行ファイルは、まず入力画像を複数のタイルに切り出し、それらを別々に処理し、最後につなぎ合わせるため、ブロックの不整合が生じる可能性があることに注意してください（また、PyTorchの実装とはわずかに異なる結果を生成します）。

### Pythonスクリプト

#### Pythonスクリプトの使用方法

1. 引数`outscale`により、**任意の出力サイズ** にX4モデルを使用することができます。このプログラムは、Real-ESRGAN出力の後に、さらに安価なリサイズ操作を行います。

```console
Usage: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]...

A common command: python inference_realesrgan.py -n RealESRGAN_x4plus -i infile --outscale 3.5 --face_enhance

  -h                   show this help
  -i --input           Input image or folder. Default: inputs
  -o --output          Output folder. Default: results
  -n --model_name      Model name. Default: RealESRGAN_x4plus
  -s, --outscale       The final upsampling scale of the image. Default: 4
  --suffix             Suffix of the restored image. Default: out
  -t, --tile           Tile size, 0 for no tile during testing. Default: 0
  --face_enhance       Whether to use GFPGAN to enhance face. Default: False
  --fp32               Use fp32 precision during inference. Default: fp16 (half precision).
  --ext                Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto
```

#### 一般的な画像の推論

学習済みモデルのダウンロード: [RealESRGAN_x4plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

インターフェイス!

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

結果は`results`フォルダにあります

#### アニメ画像の推論

<p align="center">
  <img src="https://raw.githubusercontent.com/xinntao/public-figures/master/Real-ESRGAN/cmp_realesrgan_anime_1.png">
</p>

学習済みモデル: [RealESRGAN_x4plus_anime_6B](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth)<br>
 詳細や[waifu2x](https://github.com/nihui/waifu2x-ncnn-vulkan)との比較は[**anime_model.md**](docs/anime_model.md)にあります

```bash
# ダウンロードモデル
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P weights
# インターフェイス
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i inputs
```

結果は`results`フォルダにあります

---

## BibTeX

    @InProceedings{wang2021realesrgan,
        author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
        title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
        booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
        date      = {2021}
    }

## 📧 連絡先

ご質問は、`xintao.wang@outlook.com`または`xintaowang@tencent.com`までお願いします。

<!---------------------------------- Projects that use Real-ESRGAN --------------------------->
## 🧩 Real-ESRGANを使用するプロジェクト

あなたのプロジェクトでReal-ESRGANを開発・使用しているのであれば、ぜひ教えてください。

- NCNN-Android: [RealSR-NCNN-Android](https://github.com/tumuyan/RealSR-NCNN-Android) by [tumuyan](https://github.com/tumuyan)
- VapourSynth: [vs-realesrgan](https://github.com/HolyWu/vs-realesrgan) by [HolyWu](https://github.com/HolyWu)
- NCNN: [Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)

&nbsp;&nbsp;&nbsp;&nbsp;**GUI**

- [Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) by [AaronFeng753](https://github.com/AaronFeng753)
- [Squirrel-RIFE](https://github.com/Justin62628/Squirrel-RIFE) by [Justin62628](https://github.com/Justin62628)
- [Real-GUI](https://github.com/scifx/Real-GUI) by [scifx](https://github.com/scifx)
- [Real-ESRGAN_GUI](https://github.com/net2cn/Real-ESRGAN_GUI) by [net2cn](https://github.com/net2cn)
- [Real-ESRGAN-EGUI](https://github.com/WGzeyu/Real-ESRGAN-EGUI) by [WGzeyu](https://github.com/WGzeyu)
- [anime_upscaler](https://github.com/shangar21/anime_upscaler) by [shangar21](https://github.com/shangar21)
- [Upscayl](https://github.com/upscayl/upscayl) by [Nayam Amarshe](https://github.com/NayamAmarshe) and [TGS963](https://github.com/TGS963)

## 🤗 謝辞

すべてのコントリビュータに感謝します。

- [AK391](https://github.com/AK391): RealESRGANを[Huggingface Spaces](https://huggingface.co/spaces)と[Gradio](https://github.com/gradio-app/gradio)に統合しました。[Gradio Web Demo](https://huggingface.co/spaces/akhaliq/Real-ESRGAN)を参照。
- [Asiimoviet](https://github.com/Asiimoviet): README.mdを中国語(中文)に翻訳しました。
- [2ji3150](https://github.com/2ji3150): [詳細で貴重なご意見/ご提案](https://github.com/xinntao/Real-ESRGAN/issues/131)をありがとうございました。
- [Jared-02](https://github.com/Jared-02): Training.mdを中国語(中文)に翻訳しました。
