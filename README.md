# YOLOv8 姿勢估計 (Pose Estimation) 訓練工具

此專案提供一個基於 Ultralytics YOLOv8 的姿勢估計訓練腳本 `train_pose.py`。腳本內建了自動更新 `datasets_dir` 的機制，讓你可以直接將資料集放置於專案目錄下，免去手動修改 Ultralytics 全域設定的麻煩。

---

## 📁 建議專案結構

為了讓腳本順利運作，建議將資料集與設定檔放置於以下位置：

```text
your_project/
├── datasets/
│   └── player_pose/        # 你的姿勢估計資料集（包含 images 與 labels）
│       ├── images/         # 你的姿勢影像資料集
│       │   ├── train/      # 姿勢影像拿來訓練的資料集
│       │   └── val/        # 姿勢影像拿來驗證的資料集
│       └── labels/         # 你的姿勢標記資料集
│           ├── train/      # 姿勢標記拿來訓練的資料集
│           └── val/        # 姿勢標記拿來驗證的資料集
├── pose_dataset.yaml       # 資料集設定檔
├── train_pose.py           # 訓練腳本
└── README.md               # 本說明文件
```
# 🛠️ 環境準備

請確保已安裝 ultralytics 核心套件以及對應的深度學習環境：

```bash
pip install ultralytics
```
如需使用 GPU 訓練，請確保已正確安裝支援 CUDA 的 PyTorch 版本。

# 🚀 使用方法

你可以透過命令列（Terminal / PowerShell）帶入參數來執行訓練。
## 1. 基本訓練指令 (預設參數)

```bash
python train_pose.py --data pose_dataset.yaml
```

## 2. 進階訓練指令 (指定 GPU 與實驗名稱)

```bash
python train_pose.py --data pose_dataset.yaml --model yolov8s-pose.pt --epochs 100 --imgsz 640 --batch 16 --project runs/pose --name player_from_video1 --device 0
```

## 3. Windows PowerShell 多行輸入格式
如果你使用的是 Windows PowerShell，可以使用 ` 符號換行以利閱讀：

```bash
python train_pose.py `
  --data pose_dataset.yaml `
  --model yolov8s-pose.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 16 `
  --project runs/pose `
  --name player_from_video1 `
  --device 0
```
## 4.model如果要填入預訓練模組
```bash
python train_pose.py --data pose_dataset.yaml --model runs/pose/player_from_video1/weights/best.pt --name player_from_video2
```

# 📊 參數說明

| 參數名稱 | 類型 | 預設值 | 說明 |
| :--- | :--- | :--- | :--- |
|`--data`  | `str` |(必填)|資料集YAML 設定檔的路徑 (例如：`pose_dataset.yaml`)|
|`--model`|`str`|`yolov8s-pose.pt`|預訓練模型名稱，可選 `yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt` 等|
|`--epochs`|`int`|`100`|訓練的總輪數 (Epochs)|
|`--imgsz`|`int`|`640`|輸入影像的尺寸 (會調整為 `imgsz x imgsz`)|
|`--batch`|`int`|`16`|每次訓練的批次大小 (Batch Size)，若顯存不足可調小|
|`--project`|`str`|`runs/pose`|訓練結果與日誌儲存的根目錄|
|`--name`|`str`|`exp`|此次訓練的實驗名稱，最終結果會存在 `[project]/[name]`|
|`--device`|`str`|`""`|使用的硬體裝置：`""`(自動判定), `"0"`(GPU 0), `"cpu"`(強迫使用 CPU)|

# 🎯 訓練結果

訓練完成後，模型權重、訓練日誌（Logs）以及評估圖表將會自動儲存至：
`runs/pose/[你的實驗名稱]/`

最佳權重：`runs/pose/[你的實驗名稱]/weights/best.pt`

最後權重：`runs/pose/[你的實驗名稱]/weights/last.pt`