"""Tests for the LaneDetector inference wrapper.

These tests verify the behavior of LaneDetector without requiring a real model
or CLRerNet dependencies (mmdet/libs are mocked where necessary).
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_repair_noncallable_clrernet_nms_rebinds_callable() -> None:
    """Repair helper rewires module-like nms symbols to their callable nms attribute."""
    from lane_detection.inference import LaneDetector

    rebound_callable = MagicMock()
    clrernet_head = types.SimpleNamespace(
        nms=types.SimpleNamespace(nms=rebound_callable),
    )

    with patch("lane_detection.inference.importlib.import_module", return_value=clrernet_head):
        repaired = LaneDetector._repair_noncallable_clrernet_nms()

    assert repaired is True
    assert clrernet_head.nms is rebound_callable
    assert callable(clrernet_head.nms)


def test_repair_noncallable_clrernet_nms_noop_when_already_callable() -> None:
    """Repair helper leaves callable nms symbol unchanged."""
    from lane_detection.inference import LaneDetector

    callable_nms = MagicMock()
    clrernet_head = types.SimpleNamespace(nms=callable_nms)

    with patch("lane_detection.inference.importlib.import_module", return_value=clrernet_head):
        repaired = LaneDetector._repair_noncallable_clrernet_nms()

    assert repaired is False
    assert clrernet_head.nms is callable_nms


def test_lane_detector_class_importable() -> None:
    """Verify that LaneDetector is importable from the lane_detection package."""
    from lane_detection import LaneDetector  # noqa: F401


def test_lane_detector_missing_checkpoint(tmp_path: Path) -> None:
    """LaneDetector raises FileNotFoundError for a non-existent checkpoint."""
    fake_config = tmp_path / "config.py"
    fake_config.write_text("# dummy config")

    # Patch mmdet so the import check passes even without real installation
    with patch.dict("sys.modules", {"mmdet": MagicMock(), "mmdet.apis": MagicMock()}):
        from lane_detection.inference import LaneDetector

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            LaneDetector(
                checkpoint=tmp_path / "nonexistent.pth",
                config=fake_config,
            )


def test_lane_detector_missing_config(tmp_path: Path) -> None:
    """LaneDetector raises FileNotFoundError for a non-existent config."""
    fake_checkpoint = tmp_path / "model.pth"
    fake_checkpoint.write_bytes(b"")  # create empty file so checkpoint check passes first

    with patch.dict("sys.modules", {"mmdet": MagicMock(), "mmdet.apis": MagicMock()}):
        from lane_detection.inference import LaneDetector

        with pytest.raises(FileNotFoundError, match="CLRerNet config not found"):
            LaneDetector(
                checkpoint=fake_checkpoint,
                config=tmp_path / "nonexistent_config.py",
            )


def test_lane_detector_missing_image(tmp_path: Path) -> None:
    """detect() raises FileNotFoundError when the input image path does not exist."""
    fake_config = tmp_path / "config.py"
    fake_config.write_text("# dummy config")
    fake_checkpoint = tmp_path / "model.pth"
    fake_checkpoint.write_bytes(b"")

    mock_model = MagicMock()

    mmdet_mock = MagicMock()
    mmdet_mock.apis.init_detector.return_value = mock_model

    with patch.dict(
        "sys.modules",
        {"mmdet": mmdet_mock, "mmdet.apis": mmdet_mock.apis},
    ):
        from lane_detection.inference import LaneDetector

        detector = LaneDetector(checkpoint=fake_checkpoint, config=fake_config)

        with pytest.raises(FileNotFoundError, match="Input image not found"):
            detector.detect(tmp_path / "nonexistent.jpg")


def test_lane_detector_init_uses_random_palette(tmp_path: Path) -> None:
    """LaneDetector initialization forwards palette override to init_detector."""
    fake_config = tmp_path / "config.py"
    fake_config.write_text("# dummy config")
    fake_checkpoint = tmp_path / "model.pth"
    fake_checkpoint.write_bytes(b"")

    mmdet_mock = MagicMock()
    mmdet_mock.apis.init_detector.return_value = MagicMock()

    with patch.dict(
        "sys.modules",
        {"mmdet": mmdet_mock, "mmdet.apis": mmdet_mock.apis},
    ):
        from lane_detection.inference import LaneDetector

        LaneDetector(checkpoint=fake_checkpoint, config=fake_config)

    mmdet_mock.apis.init_detector.assert_called_once_with(
        str(fake_config),
        str(fake_checkpoint),
        device="cuda:0",
        palette="random",
    )


def test_detect_with_numpy_input(tmp_path: Path) -> None:
    """detect() accepts a numpy array and passes it through a temporary file."""
    fake_config = tmp_path / "config.py"
    fake_config.write_text("# dummy config")
    fake_checkpoint = tmp_path / "model.pth"
    fake_checkpoint.write_bytes(b"")

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_src = dummy_image.copy()
    mock_preds: list = []

    mmdet_mock = MagicMock()
    mmdet_mock.apis.init_detector.return_value = MagicMock()
    libs_mock = MagicMock()
    libs_mock.api.inference.inference_one_image.return_value = (mock_src, mock_preds)

    with patch.dict(
        "sys.modules",
        {
            "mmdet": mmdet_mock,
            "mmdet.apis": mmdet_mock.apis,
            "libs": libs_mock,
            "libs.api": libs_mock.api,
            "libs.api.inference": libs_mock.api.inference,
        },
    ):
        from lane_detection.inference import LaneDetector

        detector = LaneDetector(checkpoint=fake_checkpoint, config=fake_config)
        src, preds = detector.detect(dummy_image)

    assert src is dummy_image
    assert preds == mock_preds


def test_detect_resizes_non_culane_image_to_model_input(tmp_path: Path) -> None:
    """detect() resizes images that do not match the CULane model input size (590×1640)."""
    import cv2 as _cv2

    fake_config = tmp_path / "config.py"
    fake_config.write_text("# dummy config")
    fake_checkpoint = tmp_path / "model.pth"
    fake_checkpoint.write_bytes(b"")

    # CULane resolution is 590×1640; pass a typical 540×960 image instead.
    input_image = np.zeros((540, 960, 3), dtype=np.uint8)
    mock_src = np.zeros((590, 1640, 3), dtype=np.uint8)
    mock_preds: list = []

    written_shapes: list[tuple[int, int]] = []

    original_imwrite = _cv2.imwrite

    def capturing_imwrite(path: str, img: np.ndarray, *args: object) -> bool:
        written_shapes.append((img.shape[0], img.shape[1]))
        return original_imwrite(path, img, *args)

    mmdet_mock = MagicMock()
    mmdet_mock.apis.init_detector.return_value = MagicMock()
    libs_mock = MagicMock()
    libs_mock.api.inference.inference_one_image.return_value = (mock_src, mock_preds)

    with patch.dict(
        "sys.modules",
        {
            "mmdet": mmdet_mock,
            "mmdet.apis": mmdet_mock.apis,
            "libs": libs_mock,
            "libs.api": libs_mock.api,
            "libs.api.inference": libs_mock.api.inference,
        },
    ), patch("lane_detection.inference.cv2.imwrite", side_effect=capturing_imwrite):
        from lane_detection.inference import LaneDetector

        detector = LaneDetector(checkpoint=fake_checkpoint, config=fake_config)
        src, preds = detector.detect(input_image)

    # The image written to the temp file must be the model input size.
    assert len(written_shapes) == 1
    assert written_shapes[0] == (590, 1640), f"Expected (590, 1640), got {written_shapes[0]}"
    assert src is input_image
    assert preds == mock_preds


def test_detect_inverse_transforms_predictions_after_top_crop(tmp_path: Path) -> None:
    """detect() maps model-space lane points back to original image coordinates."""
    fake_config = tmp_path / "config.py"
    fake_config.write_text("# dummy config")
    fake_checkpoint = tmp_path / "model.pth"
    fake_checkpoint.write_bytes(b"")

    input_image = np.zeros((540, 960, 3), dtype=np.uint8)
    model_src = np.zeros((590, 1640, 3), dtype=np.uint8)
    model_preds = [{"points": [[0.0, 0.0], [1640.0, 590.0]]}]

    mmdet_mock = MagicMock()
    mmdet_mock.apis.init_detector.return_value = MagicMock()
    libs_mock = MagicMock()
    libs_mock.api.inference.inference_one_image.return_value = (model_src, model_preds)

    with patch.dict(
        "sys.modules",
        {
            "mmdet": mmdet_mock,
            "mmdet.apis": mmdet_mock.apis,
            "libs": libs_mock,
            "libs.api": libs_mock.api,
            "libs.api.inference": libs_mock.api.inference,
        },
    ):
        from lane_detection.inference import LaneDetector

        detector = LaneDetector(checkpoint=fake_checkpoint, config=fake_config)
        _, preds = detector.detect(input_image)

    points = np.asarray(preds[0]["points"], dtype=np.float32)
    assert points[0, 0] == pytest.approx(0.0, abs=1.0)
    assert points[0, 1] == pytest.approx(195.0, abs=1.0)
    assert points[1, 0] == pytest.approx(960.0, abs=1.0)
    assert points[1, 1] == pytest.approx(540.0, abs=1.0)


def test_detect_and_visualize_restores_original_resolution(tmp_path: Path) -> None:
    """detect_and_visualize() returns an image matching the original input resolution."""
    fake_config = tmp_path / "config.py"
    fake_config.write_text("# dummy config")
    fake_checkpoint = tmp_path / "model.pth"
    fake_checkpoint.write_bytes(b"")

    # Simulate 540×960 input (smaller than model's 590×1640).
    input_image = np.zeros((540, 960, 3), dtype=np.uint8)
    model_src = np.zeros((590, 1640, 3), dtype=np.uint8)

    def mock_visualize(src: np.ndarray, _preds: list, **_kwargs: object) -> np.ndarray:
        return src.copy()

    mmdet_mock = MagicMock()
    mmdet_mock.apis.init_detector.return_value = MagicMock()
    libs_mock = MagicMock()
    libs_mock.api.inference.inference_one_image.return_value = (model_src, [])
    libs_mock.utils.visualizer.visualize_lanes.side_effect = mock_visualize

    with patch.dict(
        "sys.modules",
        {
            "mmdet": mmdet_mock,
            "mmdet.apis": mmdet_mock.apis,
            "libs": libs_mock,
            "libs.api": libs_mock.api,
            "libs.api.inference": libs_mock.api.inference,
            "libs.utils": libs_mock.utils,
            "libs.utils.visualizer": libs_mock.utils.visualizer,
        },
    ):
        from lane_detection.inference import LaneDetector

        detector = LaneDetector(checkpoint=fake_checkpoint, config=fake_config)
        result = detector.detect_and_visualize(input_image)

    assert result.shape[0] == 540
    assert result.shape[1] == 960


def test_detect_keeps_array_type_for_sequence_lane_predictions(tmp_path: Path) -> None:
    """Non-dict predictions remain numpy arrays after inverse transform mapping."""
    fake_config = tmp_path / "config.py"
    fake_config.write_text("# dummy config")
    fake_checkpoint = tmp_path / "model.pth"
    fake_checkpoint.write_bytes(b"")

    input_image = np.zeros((540, 960, 3), dtype=np.uint8)
    model_src = np.zeros((590, 1640, 3), dtype=np.uint8)
    model_preds = [[[0.0, 0.0], [1640.0, 590.0]]]

    mmdet_mock = MagicMock()
    mmdet_mock.apis.init_detector.return_value = MagicMock()
    libs_mock = MagicMock()
    libs_mock.api.inference.inference_one_image.return_value = (model_src, model_preds)

    with patch.dict(
        "sys.modules",
        {
            "mmdet": mmdet_mock,
            "mmdet.apis": mmdet_mock.apis,
            "libs": libs_mock,
            "libs.api": libs_mock.api,
            "libs.api.inference": libs_mock.api.inference,
        },
    ):
        from lane_detection.inference import LaneDetector

        detector = LaneDetector(checkpoint=fake_checkpoint, config=fake_config)
        _, preds = detector.detect(input_image)

    assert isinstance(preds[0], np.ndarray)
    assert preds[0].shape == (2, 2)


def _make_vertical_lane(x: float, y_start: float = 10.0, y_end: float = 110.0) -> list[list[float]]:
    """Create a simple vertical lane polyline for synthetic tests."""
    return [
        [x, y_start],
        [x, y_start + 25.0],
        [x, y_start + 50.0],
        [x, y_start + 75.0],
        [x, y_end],
    ]


def test_detect_centerline_samples_success_with_synthetic_lanes() -> None:
    """Centerline samples are produced from valid synthetic left/right white lanes."""
    from lane_detection.inference import LaneDetector

    detector = LaneDetector.__new__(LaneDetector)
    source_bgr = np.full((120, 400, 3), 255, dtype=np.uint8)
    preds = [
        {"points": _make_vertical_lane(120.0), "class_name": "white"},
        {"points": _make_vertical_lane(280.0), "class_name": "white"},
    ]

    detector.detect = MagicMock(return_value=(source_bgr, preds))
    src, raw_preds, result = detector.detect_centerline_samples(
        source_bgr,
        sample_spacing_m=1.0,
        pixels_per_meter=10.0,
    )

    assert src is source_bgr
    assert raw_preds is preds
    assert result is not None
    assert result.sample_spacing_m == pytest.approx(1.0)
    assert result.sampled_xy.shape[0] == 11
    assert np.allclose(result.sampled_xy[:, 0], 200.0, atol=1e-3)
    assert float(result.sampled_xy[0, 1]) == pytest.approx(10.0, abs=1e-3)
    assert float(result.sampled_xy[-1, 1]) == pytest.approx(110.0, abs=1e-3)


def test_detect_centerline_samples_excludes_non_white_candidates() -> None:
    """Non-white lane candidates are filtered out and cannot form a lane pair."""
    from lane_detection.inference import LaneDetector

    detector = LaneDetector.__new__(LaneDetector)
    source_bgr = np.full((120, 400, 3), 255, dtype=np.uint8)
    preds = [
        {"points": _make_vertical_lane(120.0), "class_name": "white"},
        {"points": _make_vertical_lane(280.0), "class_name": "yellow"},
    ]

    detector.detect = MagicMock(return_value=(source_bgr, preds))
    _, _, result = detector.detect_centerline_samples(
        source_bgr,
        sample_spacing_m=1.0,
        pixels_per_meter=10.0,
    )

    assert result is None


def test_detect_centerline_samples_configurable_spacing_changes_sample_density() -> None:
    """Smaller spacing produces more samples and sampled distances stay monotonic."""
    from lane_detection.inference import LaneDetector

    detector = LaneDetector.__new__(LaneDetector)
    source_bgr = np.full((120, 400, 3), 255, dtype=np.uint8)
    preds = [
        {"points": _make_vertical_lane(120.0), "class_name": "white"},
        {"points": _make_vertical_lane(280.0), "class_name": "white"},
    ]
    detector.detect = MagicMock(return_value=(source_bgr, preds))

    _, _, spacing_1m = detector.detect_centerline_samples(
        source_bgr,
        sample_spacing_m=1.0,
        pixels_per_meter=10.0,
    )
    _, _, spacing_2m = detector.detect_centerline_samples(
        source_bgr,
        sample_spacing_m=2.0,
        pixels_per_meter=10.0,
    )

    assert spacing_1m is not None
    assert spacing_2m is not None
    assert spacing_1m.sampled_xy.shape[0] > spacing_2m.sampled_xy.shape[0]

    spacing_1m_step = np.linalg.norm(np.diff(spacing_1m.sampled_xy, axis=0), axis=1) / 10.0
    spacing_2m_step = np.linalg.norm(np.diff(spacing_2m.sampled_xy, axis=0), axis=1) / 10.0
    assert np.all(spacing_1m_step > 0.0)
    assert np.all(spacing_2m_step > 0.0)
    assert np.allclose(spacing_1m_step, 1.0, atol=1e-3)
    assert np.allclose(spacing_2m_step, 2.0, atol=1e-3)


@pytest.mark.parametrize("bad_spacing", [0.0, -1.0, float("nan")])
def test_detect_centerline_samples_rejects_invalid_sample_spacing_m(bad_spacing: float) -> None:
    """Invalid meter spacing raises ValueError."""
    from lane_detection.inference import LaneDetector

    detector = LaneDetector.__new__(LaneDetector)
    with pytest.raises(ValueError, match="sample_spacing_m"):
        detector.detect_centerline_samples(
            np.zeros((10, 10, 3), dtype=np.uint8),
            sample_spacing_m=bad_spacing,
            pixels_per_meter=10.0,
        )


@pytest.mark.parametrize("bad_ppm", [0.0, -0.1, float("inf")])
def test_detect_centerline_samples_rejects_invalid_pixels_per_meter(bad_ppm: float) -> None:
    """Invalid pixel-per-meter scale raises ValueError."""
    from lane_detection.inference import LaneDetector

    detector = LaneDetector.__new__(LaneDetector)
    with pytest.raises(ValueError, match="pixels_per_meter"):
        detector.detect_centerline_samples(
            np.zeros((10, 10, 3), dtype=np.uint8),
            sample_spacing_m=1.0,
            pixels_per_meter=bad_ppm,
        )


def test_detect_centerline_samples_returns_none_when_pairing_is_impossible() -> None:
    """No left/right split around image center results in fallback None."""
    from lane_detection.inference import LaneDetector

    detector = LaneDetector.__new__(LaneDetector)
    source_bgr = np.full((120, 400, 3), 255, dtype=np.uint8)
    preds = [
        {"points": _make_vertical_lane(60.0), "class_name": "white"},
        {"points": _make_vertical_lane(100.0), "class_name": "white"},
    ]

    detector.detect = MagicMock(return_value=(source_bgr, preds))
    _, _, result = detector.detect_centerline_samples(
        source_bgr,
        sample_spacing_m=1.0,
        pixels_per_meter=10.0,
    )

    assert result is None


def test_detect_centerline_samples_returns_none_for_zero_length_duplicate_polyline() -> None:
    """Degenerate duplicate-only lanes are rejected and return fallback None."""
    from lane_detection.inference import LaneDetector

    detector = LaneDetector.__new__(LaneDetector)
    source_bgr = np.full((120, 400, 3), 255, dtype=np.uint8)
    degenerate_left = [[120.0, 20.0], [120.0, 20.0], [120.0, 20.0], [120.0, 20.0]]
    valid_right = _make_vertical_lane(280.0)
    preds = [
        {"points": degenerate_left, "class_name": "white"},
        {"points": valid_right, "class_name": "white"},
    ]

    detector.detect = MagicMock(return_value=(source_bgr, preds))
    _, _, result = detector.detect_centerline_samples(
        source_bgr,
        sample_spacing_m=1.0,
        pixels_per_meter=10.0,
    )

    assert result is None


def test_repair_shadowed_mmdet_module_removes_non_package_module() -> None:
    """Non-package mmdet entry is removed from sys.modules."""
    from lane_detection.inference import LaneDetector

    with patch.dict(sys.modules, {"mmdet": object()}):
        assert LaneDetector._repair_shadowed_mmdet_module() is True
        assert "mmdet" not in sys.modules


def test_repair_shadowed_mmdet_module_keeps_real_package() -> None:
    """Package-like mmdet module is preserved."""
    from lane_detection.inference import LaneDetector

    package_like_mmdet = MagicMock()
    package_like_mmdet.__path__ = ["/tmp/mmdet"]

    with patch.dict(sys.modules, {"mmdet": package_like_mmdet}):
        assert LaneDetector._repair_shadowed_mmdet_module() is False
        assert sys.modules["mmdet"] is package_like_mmdet
