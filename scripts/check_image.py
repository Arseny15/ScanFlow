"""Quick visual sanity check: load one good + one defect image, print stats."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from scanflow.config import AppConfig
from scanflow.loader import ScanLoader

cfg = AppConfig.from_yaml(Path(__file__).resolve().parent.parent / "config.yaml")
loader = ScanLoader(cfg)

data_dir = Path(__file__).resolve().parent.parent / "data_scan"
scans = loader.load_split("train", data_dir)
good_imgs = [s for s in scans if s.label == "good"]
defect_imgs = [s for s in scans if s.label == "defect"]

print(f"Good: {len(good_imgs)}, Defect: {len(defect_imgs)}")

g = good_imgs[0].array
d = defect_imgs[0].array
print(f"\nGood   — mean: {g.mean():.4f}, std: {g.std():.4f}, max: {g.max():.4f}")
print(f"Defect — mean: {d.mean():.4f}, std: {d.std():.4f}, max: {d.max():.4f}")

cv2.imwrite("good_sample.png", (g * 255).astype(np.uint8))
cv2.imwrite("defect_sample.png", (d * 255).astype(np.uint8))
print("\nSaved good_sample.png and defect_sample.png")
