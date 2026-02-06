# Python Workspace Template

[English Versions](README_en.md) | 日本語

Pythonプロジェクト開発用のテンプレートリポジトリです。
Dev Container、uv、Ruff、Mypyを用いたモダンな開発環境を提供します。

## 最初にやること
以下の指示をCoding Agentにお願いしてください
なお、〇〇の部分は適宜置き換えてください
```
このレポジトリは〇〇を目的としたレポジトリです。
以下の箇所を適切に修正してください
- `README.md` / `README_en.md`の「最初にやること」/「Inital Setup」の章を削除
- `README.md` / `README_en.md`のレポジトリの説明を修正
- `src/python_workspace_template`を`src/{repositroy_name}`に修正
- `docker/docker-compose.yml`の以下の内容をレポジトリ名に合わせて修正
    - image名
    - サービス名
    - volumes
    - working_dir
- `docker/docker-compose.gpu.yml`の以下の内容をレポジトリ名に合わせて修正
    - サービス名
- `docker/run-docker.sh`の起動するサービス名をレポジトリ名に合わせて修正
- `.devcontainer/devcontainer.json`の以下の内容をレポジトリ名に合わせて修正
    - name
    - service
    - workSpaceForder
```

## 機能・特徴

- **パッケージ管理**: `uv` を使用した高速な依存関係解決
- **開発環境**: Dev Container (`.devcontainer`) による一貫した環境
- **静的解析・フォーマット**: `ruff` による高速なLint/Format
- **型チェック**: `mypy` による静的型チェック
- **機械学習対応**: PyTorch (CPU/CUDA) の動的なインストール設定済み

## 使い方

### 1. Dev Container の起動
VS Code で本リポジトリを開き、推奨される拡張機能「Dev Containers」を使用してコンテナを起動してください。
`postCreateCommand` により、自動的に `uv sync` が実行され、環境がセットアップされます。

### 2. 依存関係の追加
パッケージを追加する場合は `uv` を使用します。

```bash
uv add <package_name>
```

### 3. 品質管理コマンド
本プロジェクトでは `AGENTS.md` に基づき、以下のコマンドでの品質チェックを推奨しています。

```bash
# フォーマット、Lint自動修正、型チェックを一括実行
ruff format && ruff check --fix && mypy .
```

## ディレクトリ構成

- `.devcontainer/`: Dev Container 設定 (VS Code用)
- `docker/`: docker関連ファイル
- `AGENTS.md`: コーディング規約 (Google Style, Ruff設定など)
