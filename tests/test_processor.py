"""
Unit tests for loader and processor.
Run with: pytest tests/ -v
"""
import numpy as np
import pytest
from pathlib import Path

from scanflow.config import AppConfig
from scanflow.loader import ScanData, ScanLoader, LoadError, InvalidShapeError
from scanflow.processor import ScanProcessor, ProcessedScan


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


@pytest.fixture
def good_scan() -> ScanData:
    rng = np.random.default_rng(42)
    return ScanData(
        path=Path("fake/good/scan_001.png"),
        array=rng.uniform(0.3, 0.7, (224, 224)).astype(np.float32),
        label="good",
        split="test",
        augment_group=None,
        scan_id="scan_001",
    )


@pytest.fixture
def defect_scan() -> ScanData:
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.3, 0.6, (224, 224)).astype(np.float32)
    arr[50:80, 100:140] = 0.95
    return ScanData(
        path=Path("fake/defect/scan_002.png"),
        array=arr,
        label="defect",
        split="test",
        augment_group=None,
        scan_id="scan_002",
    )


@pytest.fixture
def processor(cfg) -> ScanProcessor:
    return ScanProcessor(cfg)


class TestScanData:
    def test_is_defect_true(self, defect_scan):
        assert defect_scan.is_defect is True

    def test_is_defect_false(self, good_scan):
        assert good_scan.is_defect is False

    def test_shape_property(self, good_scan):
        assert good_scan.shape == (224, 224)


class TestScanLoader:
    def test_load_single_validates_shape(self, cfg, tmp_path):
        import cv2
        img_path = tmp_path / "bad.png"
        bad_img = np.zeros((100, 100), dtype=np.uint8)
        cv2.imwrite(str(img_path), bad_img)

        loader = ScanLoader(cfg)
        with pytest.raises(InvalidShapeError):
            loader.load_single(img_path, label="good", split="test")

    def test_load_single_missing_file(self, cfg):
        loader = ScanLoader(cfg)
        with pytest.raises(LoadError):
            loader.load_single(Path("nonexistent/path.png"), label="good", split="test")

    def test_load_single_correct_dtype(self, cfg, tmp_path):
        import cv2
        img_path = tmp_path / "test.png"
        img = np.full((224, 224), 128, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        loader = ScanLoader(cfg)
        scan = loader.load_single(img_path, label="good", split="test")

        assert scan.array.dtype == np.float32
        assert 0.0 <= scan.array.min()
        assert scan.array.max() <= 1.0


class TestScanProcessor:
    def test_output_is_processed_scan(self, processor, good_scan):
        result = processor.process(good_scan)
        assert isinstance(result, ProcessedScan)

    def test_feature_vector_is_1d(self, processor, good_scan):
        result = processor.process(good_scan)
        assert result.features.ndim == 1

    def test_feature_vector_same_length_for_all(
        self, processor, good_scan, defect_scan
    ):
        r1 = processor.process(good_scan)
        r2 = processor.process(defect_scan)
        assert r1.n_features == r2.n_features

    def test_feature_dtype_float32(self, processor, good_scan):
        result = processor.process(good_scan)
        assert result.features.dtype == np.float32

    def test_no_nan_in_features(self, processor, good_scan, defect_scan):
        for scan in [good_scan, defect_scan]:
            result = processor.process(scan)
            assert not np.any(np.isnan(result.features))

    def test_normalisation_clamps_to_0_1(self, processor, good_scan):
        result = processor.process(good_scan)
        assert result.array.min() >= 0.0
        assert result.array.max() <= 1.0

    def test_stats_keys_present(self, processor, good_scan):
        result = processor.process(good_scan)
        for key in ["mean", "std", "min", "max", "p50", "p95", "energy"]:
            assert key in result.stats

    def test_feature_matrix_shape(self, processor, good_scan, defect_scan):
        processed = processor.process_batch([good_scan, defect_scan])
        X = processor.to_feature_matrix(processed)
        assert X.shape[0] == 2
        assert X.ndim == 2

    def test_flat_image_no_division_error(self, processor):
        flat = ScanData(
            path=Path("fake/flat.png"),
            array=np.full((224, 224), 0.5, dtype=np.float32),
            label="good", split="test",
            augment_group=None, scan_id="flat",
        )
        result = processor.process(flat)
        assert not np.any(np.isnan(result.features))


class TestLoaderProcessorIntegration:
    def test_round_trip(self, cfg, tmp_path):
        import cv2
        img = (np.random.rand(224, 224) * 255).astype(np.uint8)
        img_path = tmp_path / "round_trip.png"
        cv2.imwrite(str(img_path), img)

        loader = ScanLoader(cfg)
        processor = ScanProcessor(cfg)

        scan = loader.load_single(img_path, label="good", split="test")
        result = processor.process(scan)

        assert result.n_features > 0
        assert not np.any(np.isnan(result.features))
