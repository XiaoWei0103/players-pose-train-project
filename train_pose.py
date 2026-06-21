# train_pose.py
# train_pose.py
import argparse
import os
from ultralytics import YOLO
from ultralytics.utils import SETTINGS  # ⬅ 新增

# 設定 Ultralytics 資料集根目錄為「這個專案底下的 datasets 資料夾」
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "datasets")
SETTINGS.update({"datasets_dir": DATASET_ROOT})
# 這樣之後 YAML 裡的 path 就會是相對於 DATASET_ROOT


#python train_pose.py --data pose_dataset.yaml --model yolov8s-pose.pt --epochs 100 --imgsz 640 --batch 16 --name exp1
# python train_pose.py --data pose_dataset.yaml --model yolov8s-pose.pt --epochs 100 --imgsz 640 --batch 16 --project runs/pose --name player_from_video1 --device 0

# python train_pose.py `
#   --data pose_dataset.yaml `
#   --model yolov8s-pose.pt `
#   --epochs 100 `
#   --imgsz 640 `
#   --batch 16 `
#   --project runs/pose `
#   --name player_from_video1 `
#   --device 0

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Training Tool")
    parser.add_argument("--data", type=str, required=True,
                        help="dataset yaml 路徑，例如 pose_dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt",
                        help="預訓練模型（yolov8n-pose.pt / yolov8s-pose.pt 等）")
    parser.add_argument("--epochs", type=int, default=100,
                        help="訓練輪數")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="輸入影像尺寸 (imgsz x imgsz)")
    parser.add_argument("--batch", type=int, default=16,
                        help="batch size")
    parser.add_argument("--project", type=str, default="runs/pose",
                        help="輸出結果根目錄")
    parser.add_argument("--name", type=str, default="exp",
                        help="此次實驗名稱（會變成 runs/pose/name）")
    parser.add_argument("--device", type=str, default="",
                        help="使用裝置：''=自動, '0'=GPU0, 'cpu'=用CPU")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== YOLOv8 Pose 訓練設定 ===")
    print(f"data   : {args.data}")
    print(f"model  : {args.model}")
    print(f"epochs : {args.epochs}")
    print(f"imgsz  : {args.imgsz}")
    print(f"batch  : {args.batch}")
    print(f"project: {args.project}")
    print(f"name   : {args.name}")
    print(f"device : {args.device}")

    # 載入預訓練 pose 模型
    model = YOLO(args.model)

    # 開始訓練
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,
    )

    print("訓練結束，模型與 log 已存於:", f"{args.project}/{args.name}")


if __name__ == "__main__":
    main()
