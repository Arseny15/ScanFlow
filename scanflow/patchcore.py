"""
PatchCore anomaly detection for industrial defect detection.

How it works:
  1. Use ResNet18 to extract PATCH-level features from intermediate layers
     (layer2 + layer3), not the final global vector.
     For a 224x224 image this gives a grid of 28x28 patches, each with
     a rich feature vector.

  2. During fit(): collect all patch features from GOOD training images
     into a "memory bank" of what normal looks like.
     Subsample to keep it fast.

  3. During predict(): for each test image, extract patches and find the
     nearest neighbor in the memory bank. The distance = anomaly score.
     If ANY patch is far from normal, the image is flagged.

Why this beats whole-image features:
  A tiny scratch covers maybe 1% of pixels. Whole-image stats average it away.
  PatchCore checks each local region independently — one abnormal patch is enough.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from scipy.spatial.distance import cdist

from scanflow.config import AppConfig

log = logging.getLogger(__name__)


class PatchCoreExtractor:
    """
    Extracts patch-level features from ResNet18 intermediate layers.

    Hooks into layer2 (128-dim) and layer3 (256-dim) to get multi-scale
    features. These are concatenated per spatial position to form
    384-dim patch descriptors on a 28x28 grid.
    """

    def __init__(self) -> None:
        weights = models.ResNet18_Weights.DEFAULT
        self._model = models.resnet18(weights=weights)
        self._model.eval()

        self._features: dict[str, torch.Tensor] = {}

        self._model.layer2.register_forward_hook(self._hook("layer2"))
        self._model.layer3.register_forward_hook(self._hook("layer3"))

        self._transform = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _hook(self, name: str):
        def fn(_module, _input, output):
            self._features[name] = output
        return fn

    def extract(self, img: np.ndarray) -> np.ndarray:
        """
        Extract patch features from a single image.

        Returns:
            (n_patches, 384) array where n_patches = 28*28 = 784
        """
        if img.ndim == 2:
            img_rgb = np.stack([img, img, img], axis=-1)
        else:
            img_rgb = img

        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)

        tensor = self._transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            self._model(tensor)

        feat2 = self._features["layer2"]  # (1, 128, 28, 28)
        feat3 = self._features["layer3"]  # (1, 256, 14, 14)

        feat3_up = F.interpolate(feat3, size=feat2.shape[-2:],
                                 mode="bilinear", align_corners=False)

        combined = torch.cat([feat2, feat3_up], dim=1)  # (1, 384, 28, 28)

        n_channels = combined.shape[1]
        patches = combined.squeeze(0).permute(1, 2, 0)  # (28, 28, 384)
        patches = patches.reshape(-1, n_channels)        # (784, 384)

        return patches.numpy().astype(np.float32)


@dataclass
class PatchCoreResult:
    scan_id: str
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    true_label: str

    @property
    def correct(self) -> bool | None:
        if self.true_label == "unknown":
            return None
        predicted = "defect" if self.is_anomaly else "good"
        return predicted == self.true_label


@dataclass
class PatchCoreReport:
    results: list[PatchCoreResult]
    metrics: dict = field(default_factory=dict)
    threshold: float = 0.0

    @property
    def n_total(self) -> int:
        return len(self.results)

    @property
    def n_flagged(self) -> int:
        return sum(1 for r in self.results if r.is_anomaly)

    @property
    def flag_rate(self) -> float:
        return self.n_flagged / self.n_total if self.n_total > 0 else 0.0

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "threshold": self.threshold,
                "total": self.n_total,
                "flagged": self.n_flagged,
                "results": [r.__dict__ for r in self.results],
            }, f, indent=2)
        log.info(f"Report saved -> {path}")


class PatchCoreDetector:
    """
    PatchCore anomaly detector.

    fit():    extract patches from good images -> build memory bank
    predict(): score test images by nearest-neighbor distance to memory bank
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._extractor = PatchCoreExtractor()
        self._memory_bank: np.ndarray | None = None
        self._threshold = 0.0
        self._is_fitted = False
        self._coreset_size = 5000

    def fit(self, good_scans) -> "PatchCoreDetector":
        log.info(f"PatchCore: extracting patches from {len(good_scans)} good scans...")

        all_patches = []
        for i, scan in enumerate(good_scans):
            patches = self._extractor.extract(scan.array)
            all_patches.append(patches)
            if (i + 1) % 100 == 0:
                log.info(f"  extracted {i + 1}/{len(good_scans)}")

        self._memory_bank = np.vstack(all_patches)
        log.info(f"Raw memory bank: {self._memory_bank.shape[0]} patches")

        if self._memory_bank.shape[0] > self._coreset_size:
            self._memory_bank = self._subsample(
                self._memory_bank, self._coreset_size
            )
            log.info(f"After subsampling: {self._memory_bank.shape[0]} patches")

        # Set threshold from training scores
        log.info("PatchCore: computing threshold from training scores...")
        train_scores = []
        for scan in good_scans:
            score = self._score_single(scan.array)
            train_scores.append(score)

        train_scores = np.array(train_scores)
        percentile = self.cfg.detector.threshold_percentile
        self._threshold = np.percentile(train_scores, 100 - percentile)
        log.info(f"Threshold set at p{100 - percentile}: {self._threshold:.4f}")
        log.info(f"Training score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")

        self._is_fitted = True
        return self

    def _score_single(self, img: np.ndarray) -> float:
        patches = self._extractor.extract(img)
        dists = cdist(patches, self._memory_bank, metric="euclidean")
        min_dists = dists.min(axis=1)
        return float(min_dists.max())

    def predict(self, scans) -> PatchCoreReport:
        if not self._is_fitted:
            raise RuntimeError("PatchCore not fitted -- call .fit() first")

        log.info(f"PatchCore: scoring {len(scans)} scans...")
        results = []
        for i, scan in enumerate(scans):
            score = self._score_single(scan.array)
            is_anomaly = score > self._threshold

            results.append(PatchCoreResult(
                scan_id=scan.scan_id,
                anomaly_score=float(score),
                is_anomaly=bool(is_anomaly),
                confidence=abs(score - self._threshold) / (self._threshold + 1e-8),
                true_label=scan.label,
            ))

            if (i + 1) % 50 == 0:
                log.info(f"  scored {i + 1}/{len(scans)}")

        return PatchCoreReport(results=results, threshold=self._threshold)

    def _subsample(self, features: np.ndarray, target_size: int) -> np.ndarray:
        n = features.shape[0]
        if n <= target_size:
            return features
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=target_size, replace=False)
        return features[indices]

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "patchcore.pkl", "wb") as f:
            pickle.dump({
                "memory_bank": self._memory_bank,
                "threshold": self._threshold,
            }, f)
        log.info(f"PatchCore model saved -> {model_dir / 'patchcore.pkl'}")

    def load(self, model_dir: Path) -> "PatchCoreDetector":
        model_path = Path(model_dir) / "patchcore.pkl"
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self._memory_bank = bundle["memory_bank"]
        self._threshold = bundle["threshold"]
        self._is_fitted = True
        log.info(f"PatchCore model loaded from {model_path}")
        return self
