"""
Image loader for the scan dataset.

Loads images from the directory structure and returns typed ScanData
dataclasses ready for the pipeline.

Supported structures:
  train/{1,2,3}/{0_good,1_defect}/*.png
  final_test/{good*,defect*}/*.png
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scanflow.config import AppConfig

log = logging.getLogger(__name__)


@dataclass
class ScanData:
    path: Path
    array: np.ndarray
    label: str
    split: str
    augment_group: str | None
    scan_id: str

    @property
    def is_defect(self) -> bool:
        return self.label == "defect"

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape


class LoadError(Exception):
    """Raised when an image cannot be loaded."""


class InvalidShapeError(LoadError):
    """Raised when array shape is wrong."""


class ScanLoader:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._target_size = (cfg.pipeline.image_size, cfg.pipeline.image_size)

    def _read_image(self, path: Path) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if self.cfg.pipeline.grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            raise LoadError(f"cv2 could not read: {path}")

        if img.shape[:2] != self._target_size:
            img = cv2.resize(img, self._target_size)

        return img.astype(np.float32) / 255.0

    def _validate(self, array: np.ndarray, path: Path) -> None:
        if array.shape != self._target_size:
            raise InvalidShapeError(
                f"{path.name}: expected {self._target_size}, got {array.shape}")
        if not (0.0 <= array.min() and array.max() <= 1.0):
            raise LoadError(f"{path.name}: pixel values out of [0,1] range")

    def load_single(self, path: Path, label: str, split: str,
                    augment_group: str | None = None) -> ScanData:
        array = self._read_image(path)
        self._validate(array, path)
        return ScanData(
            path=path,
            array=array,
            label=label,
            split=split,
            augment_group=augment_group,
            scan_id=path.stem,
        )

    def _get_label(self, folder_name: str) -> str | None:
        name = folder_name.lower()
        if "good" in name:
            return "good"
        if "defect" in name:
            return "defect"
        return None

    def _iter_test_paths(self, split_dir: Path):
        for label_folder in sorted(split_dir.iterdir()):
            label = self._get_label(label_folder.name)
            if label is None:
                continue
            for img_path in sorted(label_folder.glob("*.png")):
                yield img_path, label, None

    def _iter_train_paths(self, split_dir: Path):
        for aug_folder in sorted(split_dir.iterdir()):
            if not aug_folder.is_dir():
                continue
            aug_group = aug_folder.name
            for label_folder in sorted(aug_folder.iterdir()):
                label = self._get_label(label_folder.name)
                if label is None:
                    continue
                for img_path in sorted(label_folder.glob("*.png")):
                    yield img_path, label, aug_group

    def load_split(self, split: str, data_dir: Path):
        split_dir = data_dir / split
        if split == "train":
            paths_iter = self._iter_train_paths(split_dir)
        else:
            paths_iter = self._iter_test_paths(split_dir)

        scans = []
        for img_path, label, aug_group in paths_iter:
            try:
                scan = self.load_single(img_path, label, split, aug_group)
                scans.append(scan)
            except LoadError as e:
                log.warning(f"Skipped {img_path.name}: {e}")
        return scans
