# ScanFlow

Anomaly detection pipeline for industrial scan images. Built to catch surface defects (scratches, broken teeth, fabric tears) in grayscale scans using unsupervised learning — no labeled defect data needed for training.

## How it works

The pipeline trains exclusively on images of **good** parts. At inference time, it scores how "different" each test image looks compared to normal, and flags outliers as potential defects.

Two detection backends are available:

- **PatchCore** (default) — Extracts patch-level features from a pretrained ResNet18 and builds a memory bank of normal patches. Each test image is scored by how far its worst patch deviates from anything seen during training. This catches small, localized defects that whole-image methods miss.

- **IsolationForest** — Uses CNN features (ResNet18, 512-dim) or handcrafted statistical features with sklearn's IsolationForest. Faster to train, but less sensitive to small defects.

## Project structure

```
scanflow/
  config.py       # Typed dataclass config, loaded from YAML
  loader.py       # Image loading + validation
  processor.py    # Feature extraction (CNN / handcrafted)
  detector.py     # IsolationForest anomaly detector
  patchcore.py    # PatchCore anomaly detector
  cli.py          # CLI entry point
tests/
  test_processor.py
scripts/
  check_image.py    # Quick visual sanity check
  debug_scores.py   # Print score distributions
config.yaml         # Pipeline configuration
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Everything is driven through the CLI. You need a dataset structured as:

```
data_scan/
  train/{1,2,3}/{0_good,1_defect}/*.png
  final_test/{good*,defect*}/*.png
```

**Train and predict in one shot:**

```bash
python -m scanflow.cli run --data-dir data_scan
```

**Train only (saves model to outputs/models/):**

```bash
python -m scanflow.cli train --data-dir data_scan
```

**Predict only (loads saved model):**

```bash
python -m scanflow.cli predict --data-dir data_scan
```

**Override config on the fly:**

```bash
python -m scanflow.cli run --data-dir data_scan --contamination 0.05
```

Results are saved to `outputs/reports/` as timestamped JSON files.

## Configuration

All settings live in `config.yaml`:

```yaml
detector:
  algorithm: patchcore    # or "isolation_forest"
  threshold_percentile: 50

pipeline:
  image_size: 224
  grayscale: true

features:
  method: cnn             # or "handcrafted" (only used with isolation_forest)
```

`threshold_percentile` controls sensitivity — higher values flag more images (catches more defects, but more false alarms). Lower values are more conservative.

## Results

On the zipper scan dataset (140 good, 19 defect test images):

| Backend |          Precision | Recall | False alarms |
| IsolationForest + CNN | 55% | 58% | 9 |
| PatchCore |             67% | 42% | 4 |

PatchCore produces fewer false alarms with higher precision. Tuning `threshold_percentile` trades off between recall and false alarm rate.

## Tests

```bash
pytest tests/ -v
```

## Tech stack

- Python 3.10+
- PyTorch + torchvision (pretrained ResNet18)
- OpenCV (image I/O)
- scikit-learn (IsolationForest, StandardScaler)
- NumPy, SciPy

## License

MIT
