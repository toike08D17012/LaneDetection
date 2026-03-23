---
name: "docker-runner"
description: "dockerコンテナの起動スクリプト"
---

# Instructions

`docker/run-docker.sh`はdockerコンテナを起動するスクリプトです。以下のように使用できます：

## 基本的な使い方

### 1. 引数なし（デフォルト動作）
```bash
bash docker/run-docker.sh
```
実行すると、dockerコンテナ内で対話的なbashシェルが起動します。
その中で必要なコマンドを実行できます。

### 2. 引数あり（コマンド実行）
```bash
bash docker/run-docker.sh <command>
```
引数で指定したコマンドをdockerコンテナ内で実行します。

#### 使用例
```bash
# Pythonスクリプトの実行
bash docker/run-docker.sh python script.py

# Ruffでのフォーマット
bash docker/run-docker.sh ruff format

# 型チェック
bash docker/run-docker.sh mypy .
```
