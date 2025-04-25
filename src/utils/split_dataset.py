import os
import random
import shutil
import argparse
from pathlib import Path

def split_dataset(dataset_path: str, val_ratio: float = 1/3):
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images'
    labels_dir = dataset_path / 'labels'

    images_train = images_dir / 'train'
    images_val = images_dir / 'val'
    labels_train = labels_dir / 'train'
    labels_val = labels_dir / 'val'

    # 建立資料夾
    images_train.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    # 抓出所有圖片
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    print(f"總共有 {len(image_files)} 張圖片")

    # 按 val_ratio 計算驗證集數量
    num_val = int(len(image_files) * val_ratio)
    val_images = random.sample(image_files, num_val)
    val_image_stems = {img.stem for img in val_images}

    print(f"按照 val_ratio={val_ratio} 抽取 {num_val} 張作為驗證集")

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + '.txt')

        if img_path.stem in val_image_stems:
            target_img_dir = images_val
            target_label_dir = labels_val
        else:
            target_img_dir = images_train
            target_label_dir = labels_train

        if label_path.exists():
            shutil.move(str(img_path), str(target_img_dir / img_path.name))
            shutil.move(str(label_path), str(target_label_dir / label_path.name))
        else:
            print(f"警告：找不到標籤 {label_path}，跳過")

    print("資料集整理完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='資料集根目錄 (含images/labels)')
    parser.add_argument('--val-ratio', type=float, default=1/3, help='驗證集比例 (預設1/3)')
    args = parser.parse_args()

    split_dataset(args.dataset, args.val_ratio)
