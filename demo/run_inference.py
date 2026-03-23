#!/usr/bin/env python3
r"""Demo script for CLRerNet lane detection inference.

Usage:
    # With default EMA config (recommended):
    python demo/run_inference.py input.jpg weights/clrernet_culane_dla34_ema.pth

    # With explicit config path:
    python demo/run_inference.py input.jpg weights/clrernet_culane_dla34_ema.pth \\
        --config /opt/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_ema.py

    # CPU-only inference:
    python demo/run_inference.py input.jpg weights/clrernet_culane_dla34.pth --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="CLRerNet lane detection demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image", type=Path, help="Path to the input image file")
    parser.add_argument("checkpoint", type=Path, help="Path to the model checkpoint (.pth file)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=("Path to the CLRerNet config file. Defaults to clrernet_culane_dla34_ema.py in $CLRERNET_ROOT."),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result.png"),
        help="Output path for the visualized result image (default: result.png)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device string: 'cuda:0' (default) or 'cpu'",
    )
    return parser


def main() -> int:
    """Run inference and save the result image.

    Returns:
        0 on success, 1 on error.
    """
    parser = build_parser()
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: Input image not found: {args.image}", file=sys.stderr)
        return 1

    if not args.checkpoint.exists():
        print(
            f"Error: Checkpoint not found: {args.checkpoint}\n"
            "Download model weights with: bash scripts/download_weights.sh",
            file=sys.stderr,
        )
        return 1

    print(f"Loading CLRerNet model from {args.checkpoint} on {args.device}...")

    try:
        from lane_detection import LaneDetector
    except ImportError:
        print(
            "Error: failed to import lane_detection package. "
            "Set PYTHONPATH to include your project src directory, e.g.\n"
            "  export PYTHONPATH=/opt/CLRerNet:${WORK_DIR}/src\n"
            "or run inside the configured Docker environment.",
            file=sys.stderr,
        )
        return 1

    detector = LaneDetector(
        checkpoint=args.checkpoint,
        config=args.config,
        device=args.device,
    )

    print(f"Running inference on {args.image}...")
    detector.detect_and_visualize(args.image, save_path=args.output)

    print(f"Done. Result saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
