"""
Feature extraction from 2D scan arrays.

Two modes:
  - "cnn": pretrained ResNet18 -> 512-dim feature vector (default)
  - "handcrafted": row/col/global stats + texture

The CNN features capture shapes, edges, and textures at multiple scales,
making them much better at spotting small, localised defects.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from scanflow.config import AppConfig
from scanflow.loader import ScanData

log = logging.getLogger(__name__)


@dataclass
class ProcessedScan:
    scan_id: str
    label: str
    features: np.ndarray
    array: np.ndarray
    stats: dict

    @property
    def n_features(self) -> int:
        return len(self.features)

    def __repr__(self) -> str:
        return (
            f"ProcessedScan(id={self.scan_id!r}, label={self.label!r}, "
            f"n_features={self.n_features})"
        )


class ScanProcessor:
    """ScanData -> ProcessedScan with feature extraction."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._cnn_model = None

        if cfg.features.method == "cnn":
            self._init_cnn()

    def _init_cnn(self) -> None:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        self._cnn_model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self._cnn_model.eval()

        self._cnn_transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        log.info("CNN feature extractor loaded (ResNet18, 512-dim)")

    def process(self, scan: ScanData) -> ProcessedScan:
        arr = self._normalise(scan.array)

        if self.cfg.features.method == "cnn":
            features = self._extract_cnn_features(arr)
        else:
            features = self._extract_features(arr)

        stats = self._compute_stats(arr)
        log.debug(f"Processed {scan.scan_id}: {len(features)} features")

        return ProcessedScan(
            scan_id=scan.scan_id,
            label=scan.label,
            features=features,
            array=arr,
            stats=stats,
        )

    def process_batch(self, scans: list[ScanData]) -> list[ProcessedScan]:
        log.info(f"Processing batch of {len(scans)} scans...")
        results = [self.process(s) for s in scans]
        log.info(f"Batch processing complete — {len(results)} processed")
        return results

    def to_feature_matrix(self, processed: list[ProcessedScan]) -> np.ndarray:
        X = np.stack([p.features for p in processed], axis=0)
        log.info(f"Feature matrix shape: {X.shape}")
        return X

    # ------------------------------------------------------------------
    # CNN features
    # ------------------------------------------------------------------

    def _extract_cnn_features(self, arr: np.ndarray) -> np.ndarray:
        import torch

        if arr.ndim == 2:
            img_rgb = np.stack([arr, arr, arr], axis=-1)
        else:
            img_rgb = arr

        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)

        tensor = self._cnn_transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            feat = self._cnn_model(tensor)

        return feat.squeeze().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Handcrafted features
    """Scan processing module - updated from main."""
    # a
    # ------------------------------------------------------------------

    def _normalise(self, array: np.ndarray) -> np.ndarray:
        if not self.cfg.pipeline.normalise:
            return array.copy()
        mn, mx = array.min(), array.max()
        return (array - mn) / (mx - mn + 1e-8)

    def _extract_features(self, arr: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []

        if self.cfg.features.row_stats:
            parts.append(self._row_stats(arr))
        if self.cfg.features.col_stats:
            parts.append(self._col_stats(arr))
        if self.cfg.features.global_stats:
            parts.append(self._global_stats(arr))
        if self.cfg.features.texture:
            parts.append(self._texture_features(arr))

        return np.concatenate(parts).astype(np.float32)

    def _row_stats(self, arr: np.ndarray) -> np.ndarray:
        row_mean = arr.mean(axis=1)
        row_std = arr.std(axis=1)
        row_max = arr.max(axis=1)
        row_energy = (arr ** 2).mean(axis=1)
        return np.concatenate([row_mean, row_std, row_max, row_energy])

    def _col_stats(self, arr: np.ndarray) -> np.ndarray:
        col_mean = arr.mean(axis=0)
        col_std = arr.std(axis=0)
        col_max = arr.max(axis=0)
        return np.concatenate([col_mean, col_std, col_max])

    def _global_stats(self, arr: np.ndarray) -> np.ndarray:
        percentiles = np.percentile(arr, [5, 25, 50, 75, 95])
        return np.array([
            arr.mean(),
            arr.std(),
            arr.min(),
            arr.max(),
            *percentiles,
            arr.max() - arr.min(),
            float(np.median(np.abs(arr - np.median(arr)))),
        ], dtype=np.float32)

    def _texture_features(self, arr: np.ndarray) -> np.ndarray:
        gy = np.diff(arr, axis=0)
        gx = np.diff(arr, axis=1)

        h = min(gy.shape[0], gx.shape[0])
        w = min(gy.shape[1], gx.shape[1])
        grad_mag = np.sqrt(gy[:h, :w] ** 2 + gx[:h, :w] ** 2)

        local_mean = uniform_filter(arr, size=8)
        local_sq_mean = uniform_filter(arr ** 2, size=8)
        local_var = np.clip(local_sq_mean - local_mean ** 2, 0, None)
        local_std = np.sqrt(local_var)

        return np.array([
            grad_mag.mean(),
            grad_mag.std(),
            grad_mag.max(),
            np.percentile(grad_mag, 90),
            local_std.mean(),
            local_std.std(),
            local_std.max(),
        ], dtype=np.float32)

    def _compute_stats(self, arr: np.ndarray) -> dict:
        return {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "p50":    float(np.percentile(arr, 50)),
            "p95":    float(np.percentile(arr, 95)),
            "energy": float((arr ** 2).mean()),
        }
