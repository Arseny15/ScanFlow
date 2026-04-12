"""
IsolationForest anomaly detector.

Workflow:
  1. Train on GOOD images only (unsupervised — model learns "normal")
  2. Score ALL images -> anomaly_score in [-1, +1] (lower = more anomalous)
  3. Threshold scores -> binary prediction
  4. Evaluate against ground truth labels
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from scanflow.config import AppConfig
from scanflow.processor import ProcessedScan

log = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
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
class DetectionReport:
    results: list[AnomalyResult]
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


class AnomalyDetector:

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._model = IsolationForest(
            n_estimators=cfg.detector.n_estimators,
            contamination=cfg.detector.contamination,
            random_state=cfg.detector.random_state,
        )
        self._scaler = StandardScaler()
        self._threshold = 0.0
        self._is_fitted = False

    def fit(self, good_scans: list[ProcessedScan]) -> "AnomalyDetector":
        X = np.stack([s.features for s in good_scans])
        X_scaled = self._scaler.fit_transform(X)

        self._model.fit(X_scaled)

        train_scores = self._model.score_samples(X_scaled)
        self._threshold = np.percentile(
            train_scores, self.cfg.detector.threshold_percentile
        )

        self._is_fitted = True
        return self

    def predict(self, scans: list[ProcessedScan]) -> DetectionReport:
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted — call .fit() first")

        X = np.stack([s.features for s in scans])
        X_scaled = self._scaler.transform(X)
        raw_scores = self._model.score_samples(X_scaled)

        results = []
        for scan, score in zip(scans, raw_scores):
            results.append(AnomalyResult(
                scan_id=scan.scan_id,
                anomaly_score=float(score),
                is_anomaly=bool(score < self._threshold),
                confidence=0.0,
                true_label=scan.label,
            ))

        return DetectionReport(results=results, threshold=self._threshold)

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump({
                "model": self._model,
                "scaler": self._scaler,
                "threshold": self._threshold,
            }, f)

    def load(self, model_dir: Path) -> "AnomalyDetector":
        model_path = Path(model_dir) / "model.pkl"
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self._threshold = bundle["threshold"]
        self._is_fitted = True
        return self
