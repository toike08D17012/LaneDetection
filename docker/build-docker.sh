#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# ==== 設定（必要なら変更）====
export USER_NAME="${USER_NAME:-kujira}"
export GROUP_NAME="${GROUP_NAME:-$USER_NAME}"

CONTEXT=".."   # レポジトリルート
export REPO_NAME="$(basename "$(cd "${CONTEXT}" && pwd)")"

# bash の組込み $UID をそのまま使う（上書きしない）
export USER_UID="${UID}"
export USER_GID="$(id -g)"
export USER_HOME="/home/${USER_NAME}"

# ==== 実行 ====
echo "Build args:"
echo "  REPO_NAME=${REPO_NAME}"
echo "  USER_NAME=${USER_NAME}"
echo "  GROUP_NAME=${GROUP_NAME}"
echo "  USER_UID=${USER_UID}"
echo "  USER_GID=${USER_GID}"
echo "  HOME=${USER_HOME}"

docker compose build
