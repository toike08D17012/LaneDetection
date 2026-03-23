# CLRerNet 白線検知環境

[English](README_en.md) | 日本語

CLRerNet（WACV 2024）を使った白線検知推論環境です。  
Docker + CUDA 12.1 + PyTorch 2.1 + mmlab スタックで構成されています。

## 環境要件

| 項目 | 要件 |
|------|------|
| OS | Linux (Ubuntu 22.04 推奨) |
| GPU | NVIDIA CUDA 12.1 対応 GPU |
| Docker | Docker Engine + NVIDIA Container Toolkit |

> **CPU専用環境での注意**: カスタムNMS拡張はCUDA必須です。CPUのみの環境では `--device cpu` で一部機能が制限されます。

## セットアップ手順

### 1. Dockerイメージのビルド

```bash
cd docker
bash build-docker.sh
```

ビルドには時間がかかります（主にPyTorch/mmlab のダウンロード）。

### 2. モデルウェイトのダウンロード

コンテナの外（ホスト）、またはコンテナ内でダウンロードします。

```bash
# ホストで実行（コンテナ起動後にマウントされる）
bash scripts/download_weights.sh
```

ダウンロードされるファイル:
- `weights/clrernet_culane_dla34_ema.pth` (推奨・高精度 F1=81.55)
- `weights/clrernet_culane_dla34.pth` (標準 F1=81.11)

### 3. コンテナの起動

```bash
cd docker
bash run-docker.sh
```

### 4. 推論の実行

コンテナ内で以下を実行します。

```bash
# Python ラッパーを使用したシンプルな推論
python demo/run_inference.py \
    <input_image.jpg> \
    weights/clrernet_culane_dla34_ema.pth \
    --output result.png

# CLRerNet のデモスクリプトを直接使用する場合
python /opt/CLRerNet/demo/image_demo.py \
    <input_image.jpg> \
    /opt/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_ema.py \
    weights/clrernet_culane_dla34_ema.pth \
    --out-file result.png
```

### 5. Python API（コンテナ内）

```python
from lane_detection import LaneDetector

detector = LaneDetector(
    checkpoint="weights/clrernet_culane_dla34_ema.pth",
    device="cuda:0",
)

result = detector.detect_and_visualize("input.jpg", save_path="result.png")
```

## ディレクトリ構成

```
LaneDetection/
├── docker/
│   ├── Dockerfile              # CUDA 12.1 + Python 3.11 + CLRerNet
│   ├── docker-compose.yml
│   └── docker-compose.gpu.yml
├── src/
│   └── lane_detection/
│       ├── __init__.py
│       └── inference.py        # LaneDetector クラス
├── demo/
│   └── run_inference.py        # デモ推論スクリプト
├── scripts/
│   └── download_weights.sh     # モデルウェイトダウンロード
├── weights/                    # ダウンロードしたモデル (.pth) を配置
└── tests/
```

## 品質管理コマンド（コンテナ内）

```bash
ruff format && ruff check --fix && mypy . && pytest
```

## Docker ベースイメージ

本プロジェクトは `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` をベースイメージとして使用します。  
CLRerNetのカスタムNMS拡張コンパイルには `devel` イメージ（nvcc付き）が必須です。

## ディレクトリ構成

- `.devcontainer/`: Dev Container 設定 (VS Code用)
- `docker/`: Docker関連ファイル
- `src/lane_detection/`: 白線検知ラッパーモジュール
- `demo/`: 推論デモスクリプト
- `scripts/`: ユーティリティスクリプト
- `AGENTS.md`: コーディング規約 (Google Style, Ruff設定など)

