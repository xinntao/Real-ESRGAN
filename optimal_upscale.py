import os
import argparse
import subprocess
from PIL import Image
import time
import math

# PILのDecompressionBombWarningとDecompressionBombErrorを無効化
Image.MAX_IMAGE_PIXELS = None


def get_optimal_scale_factor(img_width, img_height, max_size=10000):
    """最適なスケールファクターを計算する"""
    # 長辺を基準にスケールファクターを計算
    longer_dimension = max(img_width, img_height)
    scale_factor = max_size / longer_dimension

    return scale_factor


def run_realesrgan(input_path, output_path, model="RealESRGAN_x4plus", outscale=4, gpu_id=0, tile=0):
    """Real-ESRGANを実行して画像を拡大（タイル処理を追加）"""
    tile_option = f"--tile {tile}" if tile > 0 else ""
    cmd = f"python inference_realesrgan.py -n {model} -i {input_path} -o {output_path} --outscale {outscale} --gpu-id {gpu_id} {tile_option} --alpha_upsampler realesrgan --ext png"
    print(f"実行コマンド: {cmd}")
    subprocess.call(cmd, shell=True)

    # 出力ファイル名の形式を推測
    base_name = os.path.basename(input_path).split('.')[0]
    output_file = f"{base_name}_out.png"
    return os.path.join(output_path, output_file)


def resize_to_max_dimension(image_path, output_path, max_dimension=10000):
    """画像を最大サイズに制限（アスペクト比維持）"""
    with Image.open(image_path) as img:
        width, height = img.size

        # 現在のサイズが既に最大サイズ以下なら何もしない
        if width <= max_dimension and height <= max_dimension:
            img.save(output_path)
            print(f"最終サイズ: {width}x{height}")
            return output_path

        # 縦横比を維持しながら最大サイズに収める
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        # リサイズ（Lanczosフィルターを使用）
        print(f"最大サイズに調整: {width}x{height} → {new_width}x{new_height}")
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_img.save(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='入力画像のパス')
    parser.add_argument('-o', '--output', type=str, default='results', help='出力フォルダ')
    parser.add_argument('-m', '--model', type=str, default='RealESRGAN_x4plus', help='使用するモデル')
    parser.add_argument('--max_size', type=int, default=10000, help='目標サイズ（ピクセル）')
    parser.add_argument('--gpu-id', type=int, default=0, help='使用するGPU ID')
    parser.add_argument('--tile', type=int, default=0, help='タイルサイズ（0=タイルなし、推奨値：512）')
    parser.add_argument('--overscale', type=float, default=1.5,
                        help='オーバースケール係数（現在は使用していません）')
    args = parser.parse_args()

    # 出力フォルダがない場合は作成
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 入力画像のサイズを取得
    with Image.open(args.input) as original_img:
        img_width, img_height = original_img.size
    print(f"元の画像サイズ: {img_width}x{img_height}")

    # 最適なスケールファクターを計算
    scale_factor = get_optimal_scale_factor(
        img_width, img_height,
        max_size=args.max_size
    )
    print(f"計算された最適スケールファクター: {scale_factor:.2f}x")

    # タイルサイズがデフォルト（0）なら、GPUメモリを考慮して適切なタイルサイズを設定
    tile_size = args.tile
    if tile_size == 0:
        # 画像サイズが大きい場合や複数回の拡大が必要な場合は、タイルサイズを設定
        if max(img_width, img_height) > 1000 or scale_factor > 4.0:
            tile_size = 512  # 標準的なタイルサイズ
            print(f"メモリ不足を防ぐため、タイルサイズを{tile_size}に設定しました")

    # 何回拡大する必要があるか対数計算で正確に求める
    if scale_factor <= 4.0:
        iterations = 1
    else:
        iterations = min(3, math.ceil(math.log(scale_factor) / math.log(4.0)))

    print(f"{iterations}回の拡大処理が必要です。合計スケール: {4.0 ** iterations:.2f}x")

    # 最初の拡大処理
    current_input = args.input
    current_output_path = None

    # 一時フォルダの作成
    for i in range(iterations):
        temp_dir = f"temp_output_{i}"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

    # 段階的に拡大処理
    for i in range(iterations):
        temp_dir = f"temp_output_{i}"
        print(f"\n拡大処理 {i + 1}/{iterations} - 入力: {current_input}")

        # タイルサイズを適応的に調整（後の反復では大きなタイルが必要）
        current_tile = tile_size
        if i > 0 and tile_size > 0:
            # 画像サイズが大きくなるにつれてタイルサイズを大きくする
            current_tile = min(1024, tile_size * (i + 1))
            print(f"反復 {i + 1} のタイルサイズ: {current_tile}")

        try:
            output_path = run_realesrgan(
                current_input, temp_dir,
                model=args.model,
                outscale=4,
                gpu_id=args.gpu_id,
                tile=current_tile
            )

            # 出力が存在するか確認
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"出力ファイルが生成されませんでした: {output_path}")

            current_output_path = output_path

            # 前回の入力が一時ファイルだった場合は削除（最初の入力は除く）
            if i > 0 and current_input != args.input:
                try:
                    time.sleep(0.5)  # ファイルハンドルが開放されるのを待つ
                    if os.path.exists(current_input):
                        os.remove(current_input)
                        print(f"一時ファイル削除: {current_input}")
                except Exception as e:
                    print(f"警告: 一時ファイル削除に失敗: {e}")

            # 次の反復の入力として設定
            current_input = output_path

        except Exception as e:
            print(f"エラー: 拡大処理 {i + 1} で問題が発生しました: {e}")
            print("タイルサイズを大きくして再試行してください: --tile 512 または --tile 1024")
            # 既存の一時ディレクトリを削除
            for j in range(iterations):
                temp_dir = f"temp_output_{j}"
                try:
                    if os.path.exists(temp_dir):
                        for file in os.listdir(temp_dir):
                            try:
                                os.remove(os.path.join(temp_dir, file))
                            except:
                                pass
                        os.rmdir(temp_dir)
                except:
                    pass
            return

    # 拡大後のサイズを確認（DecompressionBombErrorを避けるためにPILではなくファイルパスのみを扱う）
    if current_output_path and os.path.exists(current_output_path):
        try:
            # 直接ファイルを開く代わりに、メタデータのみを取得する方法を使用する
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            with Image.open(current_output_path) as upscaled_img:
                upscaled_width, upscaled_height = upscaled_img.size
                print(f"\n拡大後の画像サイズ: {upscaled_width}x{upscaled_height}")
        except Exception as e:
            print(f"警告: 画像サイズの取得に失敗しました: {e}")
            print("処理は続行します。")

        # 最終処理：max_sizeを超えていたら縮小
        output_filename = os.path.basename(args.input).split('.')[0] + f'_max{args.max_size}.png'
        output_path = os.path.join(args.output, output_filename)

        try:
            final_path = resize_to_max_dimension(current_output_path, output_path, args.max_size)
            print(f'\n処理完了: {final_path}')
        except Exception as e:
            print(f"エラー: 最終リサイズ処理に失敗しました: {e}")
            print("最終出力を一時ファイルからコピーします。")
            import shutil
            shutil.copy2(current_output_path, output_path)
            print(f"処理完了（リサイズなし）: {output_path}")
    else:
        print("エラー: 最終出力ファイルが見つかりません。処理に失敗しました。")

    # 一時ファイルを削除する前に少し待機
    time.sleep(0.5)

    # 一時ファイルとディレクトリを削除
    try:
        for i in range(iterations):
            temp_dir = f"temp_output_{i}"
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        file_path = os.path.join(temp_dir, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"警告: ファイル削除に失敗: {e}")
                try:
                    os.rmdir(temp_dir)
                except Exception as e:
                    print(f"警告: ディレクトリ削除に失敗: {e}")
    except Exception as e:
        print(f"一時ファイルの削除中にエラーが発生しました: {e}")
        print("プログラムは正常に終了しました。一時ファイルは手動で削除してください。")


if __name__ == '__main__':
    main()