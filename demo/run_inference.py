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
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _to_json_serializable(obj: Any) -> Any:
    """Convert nested prediction payloads into JSON-serializable objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    return obj


def _lane_point_count(lane_pred: Any) -> int:
    """Estimate point count from one prediction entry."""
    if isinstance(lane_pred, dict):
        for key in ("points", "keypoints", "coords", "polyline", "lane"):
            value = lane_pred.get(key)
            if isinstance(value, (list, tuple, np.ndarray)):
                return int(len(value))
        if "x" in lane_pred and isinstance(lane_pred["x"], (list, tuple, np.ndarray)):
            return int(len(lane_pred["x"]))
        return 0

    if isinstance(lane_pred, (list, tuple, np.ndarray)):
        return int(len(lane_pred))
    return 0


def _lane_y_range(lane_pred: Any) -> tuple[float, float] | None:
    """Estimate y-range from one prediction entry when available."""
    points: np.ndarray | None = None
    if isinstance(lane_pred, dict):
        for key in ("points", "keypoints", "coords", "polyline", "lane"):
            value = lane_pred.get(key)
            if value is None:
                continue
            candidate = np.asarray(value)
            if candidate.ndim == 2 and candidate.shape[1] >= 2:
                points = candidate[:, :2]
                break
        if points is None and "y" in lane_pred:
            y_vals = np.asarray(lane_pred["y"]) if lane_pred["y"] is not None else np.array([])
            if y_vals.size > 0:
                return float(np.min(y_vals)), float(np.max(y_vals))
    else:
        candidate = np.asarray(lane_pred)
        if candidate.ndim == 2 and candidate.shape[1] >= 2:
            points = candidate[:, :2]

    if points is None or points.size == 0:
        return None
    y_vals = points[:, 1].astype(np.float64)
    return float(np.min(y_vals)), float(np.max(y_vals))


def _build_prediction_summary(preds: list[Any]) -> list[dict[str, Any]]:
    """Create compact lane-by-lane summary for diagnostics."""
    summary: list[dict[str, Any]] = []
    for idx, lane_pred in enumerate(preds):
        y_range = _lane_y_range(lane_pred)
        lane_info: dict[str, Any] = {
            "index": idx,
            "point_count": _lane_point_count(lane_pred),
        }
        if y_range is not None:
            lane_info["y_min"] = y_range[0]
            lane_info["y_max"] = y_range[1]
        summary.append(lane_info)
    return summary


def _save_predictions_json(path: Path, image_path: Path, preds: list[Any]) -> None:
    """Write raw predictions and compact summary to a JSON file."""
    payload = {
        "image": str(image_path),
        "prediction_count": len(preds),
        "summary": _build_prediction_summary(preds),
        "predictions": _to_json_serializable(preds),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


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
    parser.add_argument(
        "--preds-json",
        type=Path,
        default=None,
        help="Optional path to save raw lane predictions as JSON for debugging",
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
    source_bgr, preds = detector.detect(args.image)
    detector.detect_and_visualize(source_bgr, save_path=args.output)

    summary = _build_prediction_summary(preds)
    print(f"Detected lane predictions: {len(preds)}")
    for lane_info in summary:
        y_text = "n/a"
        if "y_min" in lane_info and "y_max" in lane_info:
            y_text = f"{lane_info['y_min']:.1f}..{lane_info['y_max']:.1f}"
        print(
            f"  lane[{lane_info['index']}]: points={lane_info['point_count']}, "
            f"y_range={y_text}"
        )

    if args.preds_json is not None:
        _save_predictions_json(path=args.preds_json, image_path=args.image, preds=preds)
        print(f"Saved raw predictions JSON to: {args.preds_json}")

    print(f"Done. Result saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
