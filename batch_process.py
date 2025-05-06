import os
import subprocess
import time


def process_multiple_files(start_num, end_num, input_dir, output_dir, max_size=10000, tile=512):
    """複数のファイルを連続処理する"""
    total_files = end_num - start_num + 1
    current = 0

    for i in range(start_num, end_num + 1):
        file_name = f"{i:03d}.png"  # 001.png, 002.png 形式
        input_path = os.path.join(input_dir, file_name)

        if not os.path.exists(input_path):
            print(f"警告: ファイル {input_path} が見つかりません。スキップします。")
            continue

        current += 1
        print(f"\n===== ファイル {current}/{total_files} 処理中: {file_name} =====")

        # optimal_upscale.pyを実行
        cmd = f"python optimal_upscale.py -i {input_path} -o {output_dir} --max_size {max_size} --tile {tile}"
        subprocess.call(cmd, shell=True)

        # 各ファイル処理後に少し待機（システムの負荷軽減のため）
        time.sleep(1)

    print(f"\n処理完了！ {current} ファイルが処理されました。")


if __name__ == "__main__":
    # パラメータ設定
    input_directory = "line-stamps"  # 入力ファイルのディレクトリ
    output_directory = "results"  # 出力先ディレクトリ
    max_size = 11000  # 目標サイズ
    tile_size = 512  # タイルサイズ

    # 処理実行（001.pngから011.pngまで）
    process_multiple_files(1, 12, input_directory, output_directory, max_size, tile_size)