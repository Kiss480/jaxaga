import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# 設定
input_folder = r"ここにファイルの入力する画像のパスをいれる"
output_path = r"ここに出力する画像パスをいれる"
grid_size = 260  # グリッド数 (適切な値に調整)
target_width = 267
target_height = 327
threshold_value = 100

# 画像ファイルの取得
image_files = glob.glob(os.path.join(input_folder, '*.png'))

# 画像が存在しない場合のエラー処理
if not image_files:
    print(f"エラー: {input_folder} に画像ファイルが見つかりません。")
    exit()

cloud_ratios_list = []

num_images = len(image_files) # 画像の総数を取得

for image_index, image_file in enumerate(image_files):
    print(f"処理中の画像: {image_file} ({image_index + 1}/{num_images})")
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"エラー: 画像 {image_file} の読み込みに失敗しました。スキップします。")
        continue

    resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
    _, cloud_mask = cv2.threshold(resized_img, threshold_value, 255, cv2.THRESH_BINARY)

    height, width = cloud_mask.shape
    cloud_ratios = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            y_start = int(i * height / grid_size)
            y_end = min(int((i + 1) * height / grid_size), height)
            x_start = int(j * width / grid_size)
            x_end = min(int((j + 1) * width / grid_size), width)

            if (x_end - x_start) > 0 and (y_end - y_start) > 0:
                cell = cloud_mask[y_start:y_end, x_start:x_end]
                cloud_ratio = np.mean(cell == 255)
                cloud_ratios[i, j] = cloud_ratio
            else:
                cloud_ratios[i, j] = 0.0

    cloud_ratios_list.append(cloud_ratios)

if cloud_ratios_list:
    average_cloud_distribution = np.mean(cloud_ratios_list, axis=0)

    np.save("average_cloud_distribution.npy", average_cloud_distribution)

    # 上下反転の修正を適用
    flipped_distribution = np.flipud(average_cloud_distribution)

    # ヒートマップの描画 (上下反転修正済み)
    plt.figure()
    plt.imshow(flipped_distribution, cmap='Blues', origin='lower')
    plt.colorbar()
    plt.title("Average Cloud Distribution")
    plt.savefig("raw_average_cloud_distribution.png")
    plt.show()

    # ヒートマップの描画 (拡張範囲指定)
    plt.figure(figsize=(8, 8))
    plt.imshow(flipped_distribution, cmap='Blues', extent=[0, target_width, 0, target_height], origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Cloud Coverage Ratio')
    plt.title("Average Cloud Distribution")
    plt.xlabel("X [pixel]")
    plt.ylabel("Y [pixel]")
    plt.savefig(output_path)
    print(f"結果のヒートマップが '{output_path}' として保存されました。")
    plt.show()
else:
    print("エラー: 処理する画像がありませんでした。")
