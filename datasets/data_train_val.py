import os
import random
import shutil
from pathlib import Path

def split_yolo_data(data_root, train_ratio=0.8):
    # 1. 設定原始路徑與目標路徑
    raw_img_dir = Path(data_root) / "images"
    raw_lab_dir = Path(data_root) / "labels"
    
    # 建立 YOLO 要求的子資料夾
    target_structure = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    for folder in target_structure:
        (Path(data_root) / folder).mkdir(parents=True, exist_ok=True)

    # 2. 找出對應成功的檔案 (對齊)
    # 取得所有圖片的檔名 (不含副檔名)
    all_images = {f.stem: f.suffix for f in raw_img_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    all_labels = {f.stem: f.suffix for f in raw_lab_dir.glob("*.txt") if f.is_file()}

    # 取交集：只有圖片跟標籤都有的才要
    matched_names = list(all_images.keys() & all_labels.keys())
    
    print(f"找到圖片: {len(all_images)} 張")
    print(f"找到標籤: {len(all_labels)} 個")
    print(f"成功配對: {len(matched_names)} 組")

    # 3. 隨機洗牌並分配
    random.shuffle(matched_names)
    split_idx = int(len(matched_names) * train_ratio)
    
    train_list = matched_names[:split_idx]
    val_list = matched_names[split_idx:]

    def move_files(name_list, subset):
        for name in name_list:
            # 搬移圖片
            img_ext = all_images[name]
            shutil.move(raw_img_dir / f"{name}{img_ext}", Path(data_root) / f"images/{subset}/{name}{img_ext}")
            # 搬移標籤
            shutil.move(raw_lab_dir / f"{name}.txt", Path(data_root) / f"labels/{subset}/{name}.txt")

    # 4. 執行搬移
    print(f"正在分配檔案到 {data_root} ...")
    move_files(train_list, "train")
    move_files(val_list, "val")
    
    print(f"完成！")
    print(f"  - Train: {len(train_list)} 組")
    print(f"  - Val:   {len(val_list)} 組")

if __name__ == "__main__":
    # 這裡填入你的資料夾路徑，例如 'datasets/my_pose_data'
    # 執行前請確保 images/ 和 labels/ 就在這個路徑下
    DATA_PATH = "datasets/player_pose" 
    split_yolo_data(DATA_PATH)