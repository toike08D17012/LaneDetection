"""CLRerNet lane detection inference wrapper.

Requires CLRerNet dependencies to be installed and CLRERNET_ROOT environment
variable to be set (default: /opt/CLRerNet inside the Docker container).

Example:
    detector = LaneDetector(checkpoint="weights/clrernet_culane_dla34_ema.pth")
    result_image = detector.detect_and_visualize("input.jpg", save_path="result.png")
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Default CLRerNet installation root inside the Docker container
_CLRERNET_ROOT = os.environ.get("CLRERNET_ROOT", "/opt/CLRerNet")

# Default config relative to CLRerNet root (EMA model recommended for higher F1)
_DEFAULT_CONFIG = "configs/clrernet/culane/clrernet_culane_dla34_ema.py"

# Wide threshold to include bright whites under typical road illumination.
_WHITE_BGR_LOW = np.array([160, 160, 160], dtype=np.uint8)
_WHITE_BGR_HIGH = np.array([255, 255, 255], dtype=np.uint8)


@dataclass(frozen=True)
class CenterlineSampleResult:
    """Container for centerline and distance-based sampling outputs.

    Attributes:
        centerline_xy: Centerline polyline in image coordinates (x, y).
        sampled_xy: Points sampled from centerline at fixed meter spacing.
        cumulative_distance_m: Cumulative arc-length for centerline points in meters.
        sample_spacing_m: Requested fixed sampling spacing in meters.
    """

    centerline_xy: np.ndarray
    sampled_xy: np.ndarray
    cumulative_distance_m: np.ndarray
    sample_spacing_m: float


class LaneDetector:
    """CLRerNet-based lane detection model.

    Wraps CLRerNet's inference API to provide a simple interface for single-image
    lane detection. The underlying model (DLA34 backbone) was trained on the CULane
    dataset and achieves 81.55 F1 with the EMA variant.

    Attributes:
        device: PyTorch device string used for inference.
        model: Loaded MMDetection model instance.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path | None = None,
        device: str = "cuda:0",
    ) -> None:
        """Initialize the lane detector and load model weights.

        Args:
            checkpoint: Path to the CLRerNet model checkpoint (.pth file).
                Download with scripts/download_weights.sh.
            config: Path to the CLRerNet config file. If None, uses the default
                EMA config at ``$CLRERNET_ROOT/configs/clrernet/culane/
                clrernet_culane_dla34_ema.py``.
            device: PyTorch device string. Use ``"cuda:0"`` for GPU inference or
                ``"cpu"`` for CPU-only mode (significantly slower).

        Raises:
            ImportError: If mmdet or CLRerNet ``libs`` package are not installed.
            FileNotFoundError: If the config or checkpoint file does not exist.
        """
        try:
            from mmdet.apis import init_detector  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mmdet is not installed. Build and run the Docker environment first:\n"
                "  cd docker && bash build-docker.sh && bash run-docker.sh"
            ) from exc

        clrernet_root = Path(_CLRERNET_ROOT)
        resolved_config = Path(config) if config is not None else clrernet_root / _DEFAULT_CONFIG
        resolved_checkpoint = Path(checkpoint)

        if not resolved_config.exists():
            raise FileNotFoundError(f"CLRerNet config not found: {resolved_config}")
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

        from mmdet.apis import init_detector

        self.device = device
        self.model = init_detector(str(resolved_config), str(resolved_checkpoint), device=device)

    @staticmethod
    def _extract_lane_points(lane_pred: Any) -> np.ndarray | None:
        """Extract lane points from heterogeneous prediction objects.

        Supported input variants include dictionary-like predictions with keys
        such as ``points``/``keypoints``/``coords`` and sequence-like lanes
        containing ``(x, y)`` pairs.
        """
        candidate: Any = None
        if isinstance(lane_pred, dict):
            for key in ("points", "keypoints", "coords", "polyline", "lane"):
                if key in lane_pred:
                    candidate = lane_pred[key]
                    break
            if candidate is None and "x" in lane_pred and "y" in lane_pred:
                x_vals = np.asarray(lane_pred["x"], dtype=np.float64)
                y_vals = np.asarray(lane_pred["y"], dtype=np.float64)
                if x_vals.shape != y_vals.shape:
                    return None
                candidate = np.column_stack((x_vals, y_vals))
        else:
            candidate = lane_pred

        if candidate is None:
            return None

        points = np.asarray(candidate, dtype=np.float64)
        if points.ndim != 2:
            return None
        if points.shape[1] >= 2:
            points = points[:, :2]
        else:
            return None
        if not np.isfinite(points).all():
            return None
        return points.astype(np.float32)

    @staticmethod
    def _deduplicate_polyline(points_xy: np.ndarray, min_points: int = 2) -> np.ndarray | None:
        """Drop repeated points and reject zero-length polylines."""
        if points_xy.ndim != 2 or points_xy.shape[0] < min_points:
            return None

        rounded = np.round(points_xy.astype(np.float64), 4)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        unique_points = points_xy[np.sort(unique_indices)]
        if unique_points.shape[0] < min_points:
            return None

        seg = np.diff(unique_points, axis=0)
        lengths = np.linalg.norm(seg, axis=1)
        if not np.any(lengths > 0.0):
            return None
        return unique_points

    @staticmethod
    def _is_white_lane(
        lane_pred: Any,
        source_bgr: np.ndarray | None,
        points_xy: np.ndarray,
        min_white_ratio: float = 0.5,
    ) -> bool:
        """Check white-lane class first, then fallback to image-based color test."""
        if isinstance(lane_pred, dict):
            label_candidate = lane_pred.get("class_name") or lane_pred.get("label") or lane_pred.get("type")
            if isinstance(label_candidate, str):
                lower_label = label_candidate.lower()
                if "white" in lower_label:
                    return True
                if "yellow" in lower_label:
                    return False

            class_id = lane_pred.get("class_id")
            class_name_map = lane_pred.get("class_name_map")
            if class_id is not None and isinstance(class_name_map, dict):
                class_name = class_name_map.get(class_id)
                if isinstance(class_name, str):
                    lower_name = class_name.lower()
                    if "white" in lower_name:
                        return True
                    if "yellow" in lower_name:
                        return False

        if source_bgr is None or points_xy.size == 0:
            return False

        h, w = source_bgr.shape[:2]
        xy = np.rint(points_xy).astype(np.int32)
        valid_mask = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        if not np.any(valid_mask):
            return False

        valid_xy = xy[valid_mask]
        sampled = source_bgr[valid_xy[:, 1], valid_xy[:, 0]]
        white_mask = np.all((sampled >= _WHITE_BGR_LOW) & (sampled <= _WHITE_BGR_HIGH), axis=1)
        white_ratio = float(np.mean(white_mask))
        return white_ratio >= min_white_ratio

    def _normalize_lane_points(
        self,
        preds: list,
        source_bgr: np.ndarray | None = None,
        require_white: bool = True,
        min_points: int = 4,
    ) -> list[np.ndarray]:
        """Normalize raw predictions into cleaned lane polylines."""
        normalized: list[np.ndarray] = []
        for lane_pred in preds:
            points = self._extract_lane_points(lane_pred)
            if points is None:
                continue

            points = self._deduplicate_polyline(points, min_points=min_points)
            if points is None:
                continue

            # Interpolation helpers expect monotonic y-ordering.
            y_order = np.argsort(points[:, 1])
            points = points[y_order]
            points = self._deduplicate_polyline(points, min_points=min_points)
            if points is None:
                continue

            if require_white and not self._is_white_lane(lane_pred, source_bgr, points):
                continue

            normalized.append(points.astype(np.float32))
        return normalized

    @staticmethod
    def _interpolate_x_at_y(lane_xy: np.ndarray, y: float) -> float | None:
        """Interpolate x-coordinate at a specific y-coordinate."""
        if lane_xy.shape[0] < 2:
            return None
        ys = lane_xy[:, 1].astype(np.float64)
        xs = lane_xy[:, 0].astype(np.float64)

        order = np.argsort(ys)
        ys = ys[order]
        xs = xs[order]
        unique_ys, unique_indices = np.unique(ys, return_index=True)
        unique_xs = xs[unique_indices]
        if unique_ys.shape[0] < 2:
            return None
        if y < float(unique_ys[0]) or y > float(unique_ys[-1]):
            return None
        x_interp = np.interp(y, unique_ys, unique_xs)
        return float(x_interp)

    def _select_ego_lane_pair(
        self,
        lanes_xy: list[np.ndarray],
        image_shape: tuple[int, int],
        reference_y_ratio: float = 0.9,
        max_lane_width_ratio: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Pick nearest left/right lanes around the image center near the bottom."""
        if len(lanes_xy) < 2:
            return None

        height, width = image_shape
        center_x = width / 2.0
        reference_y = (height - 1) * reference_y_ratio

        left_candidates: list[tuple[float, np.ndarray]] = []
        right_candidates: list[tuple[float, np.ndarray]] = []
        for lane_xy in lanes_xy:
            x_at_ref = self._interpolate_x_at_y(lane_xy, reference_y)
            if x_at_ref is None:
                continue
            distance_to_center = abs(x_at_ref - center_x)
            if x_at_ref < center_x:
                left_candidates.append((distance_to_center, lane_xy))
            elif x_at_ref > center_x:
                right_candidates.append((distance_to_center, lane_xy))

        if not left_candidates or not right_candidates:
            return None

        left_lane = min(left_candidates, key=lambda item: item[0])[1]
        right_lane = min(right_candidates, key=lambda item: item[0])[1]

        left_x = self._interpolate_x_at_y(left_lane, reference_y)
        right_x = self._interpolate_x_at_y(right_lane, reference_y)
        if left_x is None or right_x is None or left_x >= right_x:
            return None

        lane_width = right_x - left_x
        if lane_width <= 0.0 or lane_width > width * max_lane_width_ratio:
            return None
        return left_lane, right_lane

    @staticmethod
    def _compute_centerline_polyline(
        left_lane_xy: np.ndarray,
        right_lane_xy: np.ndarray,
        min_overlap_points: int = 4,
    ) -> np.ndarray | None:
        """Generate centerline as midpoint of interpolated left and right lanes."""
        if left_lane_xy.shape[0] < 2 or right_lane_xy.shape[0] < 2:
            return None

        left_y_min, left_y_max = float(np.min(left_lane_xy[:, 1])), float(np.max(left_lane_xy[:, 1]))
        right_y_min, right_y_max = float(np.min(right_lane_xy[:, 1])), float(np.max(right_lane_xy[:, 1]))
        overlap_y_min = max(left_y_min, right_y_min)
        overlap_y_max = min(left_y_max, right_y_max)
        if overlap_y_max <= overlap_y_min:
            return None

        shared_count = min(left_lane_xy.shape[0], right_lane_xy.shape[0])
        n_samples = max(min_overlap_points, shared_count)
        y_samples = np.linspace(overlap_y_min, overlap_y_max, num=n_samples, dtype=np.float64)

        left_x = np.array([LaneDetector._interpolate_x_at_y(left_lane_xy, y) for y in y_samples], dtype=np.float64)
        right_x = np.array([LaneDetector._interpolate_x_at_y(right_lane_xy, y) for y in y_samples], dtype=np.float64)
        valid = np.isfinite(left_x) & np.isfinite(right_x)
        if int(np.count_nonzero(valid)) < min_overlap_points:
            return None

        center_x = (left_x[valid] + right_x[valid]) / 2.0
        center_y = y_samples[valid]
        centerline = np.column_stack((center_x, center_y)).astype(np.float32)
        return LaneDetector._deduplicate_polyline(centerline, min_points=min_overlap_points)

    @staticmethod
    def _sample_polyline_by_distance_px(
        polyline_xy: np.ndarray,
        sample_spacing_px: float,
    ) -> np.ndarray | None:
        """Sample a polyline at fixed pixel intervals along arc length."""
        if sample_spacing_px <= 0:
            raise ValueError("sample_spacing_px must be positive")

        points = LaneDetector._deduplicate_polyline(polyline_xy, min_points=2)
        if points is None:
            return None

        deltas = np.diff(points, axis=0)
        seg_lengths = np.linalg.norm(deltas, axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        total_length = float(cumulative[-1])
        if total_length <= 0.0:
            return None

        sample_distances = np.arange(0.0, total_length + sample_spacing_px, sample_spacing_px, dtype=np.float64)
        if sample_distances[-1] > total_length:
            sample_distances[-1] = total_length

        sampled_x = np.interp(sample_distances, cumulative, points[:, 0])
        sampled_y = np.interp(sample_distances, cumulative, points[:, 1])
        sampled = np.column_stack((sampled_x, sampled_y)).astype(np.float32)
        return LaneDetector._deduplicate_polyline(sampled, min_points=2)

    @staticmethod
    def _sample_polyline_at_meter_spacing(
        polyline_xy: np.ndarray,
        sample_spacing_m: float,
        pixels_per_meter: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Sample polyline using meter spacing and return cumulative distance in meters."""
        if sample_spacing_m <= 0:
            raise ValueError("sample_spacing_m must be positive")
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be positive")

        points = LaneDetector._deduplicate_polyline(polyline_xy, min_points=2)
        if points is None:
            return None

        seg_lengths_px = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_px = np.concatenate(([0.0], np.cumsum(seg_lengths_px)))
        if float(cumulative_px[-1]) <= 0.0:
            return None

        cumulative_m = cumulative_px / pixels_per_meter
        sampled = LaneDetector._sample_polyline_by_distance_px(points, sample_spacing_m * pixels_per_meter)
        if sampled is None:
            return None
        return sampled, cumulative_m.astype(np.float32)

    @staticmethod
    def _repair_shadowed_mmdet_module() -> bool:
        """Remove non-package ``mmdet`` module from ``sys.modules`` when shadowed."""
        mmdet_module = sys.modules.get("mmdet")
        if mmdet_module is None:
            return False

        is_package_by_path = hasattr(mmdet_module, "__path__")
        module_spec = getattr(mmdet_module, "__spec__", None)
        has_package_spec = bool(
            module_spec is not None and getattr(module_spec, "submodule_search_locations", None)
        )
        if is_package_by_path or has_package_spec:
            return False

        sys.modules.pop("mmdet", None)
        importlib.invalidate_caches()
        return True

    def detect(self, image: str | Path | np.ndarray) -> tuple[np.ndarray, list]:
        """Run lane detection on a single image.

        Args:
            image: Input image. Either a file path (str or Path) or a BGR numpy
                array with shape (H, W, 3) in OpenCV format.

        Returns:
            A tuple ``(source_bgr, predictions)`` where ``source_bgr`` is the
            original image as a BGR numpy array and ``predictions`` is a list of
            detected lane keypoints.

        Raises:
            FileNotFoundError: If the provided image path does not exist.
        """
        inference_one_image = None
        try:
            from libs.api.inference import inference_one_image
        except ImportError as exc:
            err = str(exc)
            is_shadowed_mmdet_error = "mmdet.registry" in err and "'mmdet' is not a package" in err
            if is_shadowed_mmdet_error and self._repair_shadowed_mmdet_module():
                try:
                    from libs.api.inference import inference_one_image
                except ImportError as retry_exc:
                    raise ImportError(
                        "Failed to import CLRerNet inference because a non-package 'mmdet' module was "
                        "detected in the Python runtime. Restart the process or verify PYTHONPATH does "
                        "not shadow the official mmdet package."
                    ) from retry_exc
            else:
                raise ImportError(
                    "CLRerNet 'libs' package not found. "
                    f"Ensure CLRERNET_ROOT points to a valid CLRerNet checkout: {_CLRERNET_ROOT}."
                ) from exc

        if inference_one_image is None:
            raise RuntimeError("Failed to load CLRerNet inference function.")

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Input image not found: {image_path}")
            src, preds = inference_one_image(self.model, str(image_path))
        else:
            # Accept numpy array input by writing to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                cv2.imwrite(tmp_path, image)
                src, preds = inference_one_image(self.model, tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        return src, preds

    def detect_centerline_samples(
        self,
        image: str | Path | np.ndarray,
        *,
        sample_spacing_m: float = 1.0,
        pixels_per_meter: float,
    ) -> tuple[np.ndarray, list, CenterlineSampleResult | None]:
        """Detect lanes and optionally return sampled ego-lane centerline.

        Args:
            image: Input image. Either a file path or a BGR numpy array.
            sample_spacing_m: Sampling interval in meters for centerline points.
            pixels_per_meter: Pixel-to-meter scale used for arc-length sampling.

        Returns:
            Tuple ``(source_bgr, preds, result_or_none)`` where ``result_or_none``
            is ``None`` when valid ego-lane pairing or centerline sampling is not
            possible.

        Raises:
            ValueError: If ``sample_spacing_m`` or ``pixels_per_meter`` is not
                a positive finite value.
        """
        spacing = float(sample_spacing_m)
        ppm = float(pixels_per_meter)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("sample_spacing_m must be a positive finite value")
        if not np.isfinite(ppm) or ppm <= 0.0:
            raise ValueError("pixels_per_meter must be a positive finite value")

        source_bgr, preds = self.detect(image)
        lanes_xy = self._normalize_lane_points(preds=preds, source_bgr=source_bgr, require_white=True)
        lane_pair = self._select_ego_lane_pair(lanes_xy=lanes_xy, image_shape=source_bgr.shape[:2])
        if lane_pair is None:
            return source_bgr, preds, None

        centerline_xy = self._compute_centerline_polyline(
            left_lane_xy=lane_pair[0],
            right_lane_xy=lane_pair[1],
        )
        if centerline_xy is None:
            return source_bgr, preds, None

        sampled_payload = self._sample_polyline_at_meter_spacing(
            polyline_xy=centerline_xy,
            sample_spacing_m=spacing,
            pixels_per_meter=ppm,
        )
        if sampled_payload is None:
            return source_bgr, preds, None

        sampled_xy, cumulative_distance_m = sampled_payload
        result = CenterlineSampleResult(
            centerline_xy=centerline_xy,
            sampled_xy=sampled_xy,
            cumulative_distance_m=cumulative_distance_m,
            sample_spacing_m=spacing,
        )
        return source_bgr, preds, result

    def detect_and_visualize(
        self,
        image: str | Path | np.ndarray,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Run lane detection and return a visualization with lane overlays.

        Args:
            image: Input image. Either a file path or a BGR numpy array.
            save_path: If provided, the visualized result is written to this path.
                Supports any format accepted by ``cv2.imwrite`` (e.g. ``.png``, ``.jpg``).

        Returns:
            BGR numpy array of the same size as the input with detected lanes
            drawn as colored polylines.
        """
        try:
            from libs.utils.visualizer import visualize_lanes
        except ImportError as exc:
            raise ImportError(
                "CLRerNet 'libs' package not found. "
                f"Ensure CLRERNET_ROOT points to a valid CLRerNet checkout: {_CLRERNET_ROOT}."
            ) from exc

        src, preds = self.detect(image)

        kwargs: dict = {}
        if save_path is not None:
            kwargs["save_path"] = str(save_path)

        dst: np.ndarray = visualize_lanes(src, preds, **kwargs)
        return dst
