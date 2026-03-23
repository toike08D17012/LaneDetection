#!/usr/bin/env bash
# Download CLRerNet pre-trained model weights from the official GitHub release.
#
# Usage:
#   bash scripts/download_weights.sh [output_directory]
#
# The weights are saved to the specified directory (default: ./weights).
# Two variants are available:
#   - clrernet_culane_dla34_ema.pth  : EMA model, F1=81.55 (recommended)
#   - clrernet_culane_dla34.pth      : Standard model, F1=81.11

set -euo pipefail

WEIGHTS_DIR="${1:-./weights}"
BASE_URL="https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0"

mkdir -p "${WEIGHTS_DIR}"

echo "Downloading CLRerNet model weights to '${WEIGHTS_DIR}'..."

# EMA model (recommended for highest accuracy)
echo "  → clrernet_culane_dla34_ema.pth (EMA, F1=81.55)"
wget --no-clobber --show-progress \
    -P "${WEIGHTS_DIR}" \
    "${BASE_URL}/clrernet_culane_dla34_ema.pth"

# Standard model
echo "  → clrernet_culane_dla34.pth (standard, F1=81.11)"
wget --no-clobber --show-progress \
    -P "${WEIGHTS_DIR}" \
    "${BASE_URL}/clrernet_culane_dla34.pth"

echo ""
echo "Done. Weights saved to: ${WEIGHTS_DIR}"
echo ""
echo "Run inference with:"
echo "  python demo/run_inference.py <image.jpg> ${WEIGHTS_DIR}/clrernet_culane_dla34_ema.pth"
