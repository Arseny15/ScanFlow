"""Debug script: train detector and print score distributions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scanflow.config import AppConfig
from scanflow.loader import ScanLoader
from scanflow.processor import ScanProcessor
from scanflow.detector import AnomalyDetector

cfg = AppConfig.from_yaml(Path(__file__).resolve().parent.parent / "config.yaml")
loader = ScanLoader(cfg)
processor = ScanProcessor(cfg)
detector = AnomalyDetector(cfg)

data_dir = Path(__file__).resolve().parent.parent / "data_scan"

train_scans = loader.load_split("train", data_dir)
processed_train = processor.process_batch(train_scans)
good_only = [p for p in processed_train if p.label == "good"]
detector.fit(good_only)

test_scans = loader.load_split("final_test", data_dir)
processed_test = processor.process_batch(test_scans)

X = np.stack([s.features for s in processed_test])
X_scaled = detector._scaler.transform(X)
scores = detector._model.score_samples(X_scaled)

good_scores = [s for s, p in zip(scores, processed_test) if p.label == "good"]
defect_scores = [s for s, p in zip(scores, processed_test) if p.label == "defect"]

print(f"Threshold: {detector._threshold:.4f}")
print(f"\nGood scores   — min: {min(good_scores):.4f}, max: {max(good_scores):.4f}, "
      f"mean: {np.mean(good_scores):.4f}")
print(f"Defect scores — min: {min(defect_scores):.4f}, max: {max(defect_scores):.4f}, "
      f"mean: {np.mean(defect_scores):.4f}")
print(f"\nBelow threshold: {sum(1 for s in scores if s < detector._threshold)}")
