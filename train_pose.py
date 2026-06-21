# train_pose.py
# train_pose.py
import argparse
import os
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# 設定 Ultralytics 資料集根目錄為「這個專案底下的 datasets 資料夾」
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "datasets")
SETTINGS.update({"datasets_dir": DATASET_ROOT})

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
    #Add 20260621
    parser.add_argument("--degrees", type=float, default=0.0,
                        help="隨機旋轉角度範圍，例如 15.0 (預設 0.0)")
    parser.add_argument("--hsv_v", type=float, default=0.0,
                        help="隨機明度/對比度調整幅度，例如 0.4 (預設 0.0)")
    parser.add_argument("--fliplr", type=float, default=0.0,
                        help="隨機左右翻轉機率，例如 0.5 (預設 0.0)")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="微量的圖層混疊增強機率，例如 0.1 (預設 0.0)")
    #
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
    print(f"degrees: {args.degrees}")
    print(f"hsv_v  : {args.hsv_v}")
    print(f"fliplr : {args.fliplr}")
    print(f"mixup  : {args.mixup}")

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
        degrees=args.degrees,
        hsv_v=args.hsv_v,
        fliplr=args.fliplr,
        mixup=args.mixup,
    )

    print("訓練結束，模型與 log 已存於:", f"{args.project}/{args.name}")


if __name__ == "__main__":
    main()
