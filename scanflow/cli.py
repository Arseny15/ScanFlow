"""
CLI entry point for ScanFlow.

  train   — load train split, extract features, fit detector, save model
  predict — load test split, score images, save detection report
  run     — train then predict in one shot (default workflow)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from scanflow.config import AppConfig
from scanflow.loader import ScanLoader
from scanflow.processor import ScanProcessor
from scanflow.detector import AnomalyDetector, DetectionReport
from scanflow.patchcore import PatchCoreDetector, PatchCoreReport

log = logging.getLogger(__name__)


def timed(fn):
    """Log how long each major step takes."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info(f"{fn.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def _use_patchcore(cfg: AppConfig) -> bool:
    return cfg.detector.algorithm == "patchcore"


@timed
def step_train(cfg: AppConfig, data_dir: Path):
    """Load train data -> extract features -> fit detector -> save."""
    loader = ScanLoader(cfg)

    log.info("=== STEP: Load training data ===")
    all_train = loader.load_split("train", data_dir)

    if _use_patchcore(cfg):
        detector = PatchCoreDetector(cfg)
        good_scans = [s for s in all_train if s.label == "good"]
        log.info(f"Training PatchCore on {len(good_scans)} good scans "
                 f"(of {len(all_train)} total)")
        detector.fit(good_scans)
    else:
        processor = ScanProcessor(cfg)
        detector = AnomalyDetector(cfg)
        log.info("=== STEP: Extract features ===")
        processed = processor.process_batch(all_train)
        good_scans = [p for p in processed if p.label == "good"]
        log.info(f"Training IsolationForest on {len(good_scans)} good scans "
                 f"(of {len(processed)} total)")
        detector.fit(good_scans)

    detector.save(cfg.paths.model_dir)
    return detector


@timed
def step_predict(cfg: AppConfig, data_dir: Path, detector, dry_run: bool = False):
    """Load test data -> score -> save report JSON."""
    loader = ScanLoader(cfg)

    log.info("=== STEP: Load test data ===")
    test_scans = loader.load_split("final_test", data_dir)
    log.info(f"Test split: {len(test_scans)} images loaded")

    if dry_run:
        log.info(f"[DRY RUN] Would score {len(test_scans)} images -- stopping here")
        return None

    log.info("=== STEP: Score scans ===")
    if _use_patchcore(cfg):
        report = detector.predict(test_scans)
    else:
        processor = ScanProcessor(cfg)
        processed = processor.process_batch(test_scans)
        report = detector.predict(processed)

    # Save report with timestamp and algorithm name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algo = cfg.detector.algorithm
    report_path = cfg.paths.reports_dir / f"report_{algo}_{timestamp}.json"
    report.save(report_path)

    _print_summary(report)
    return report


def _print_summary(report) -> None:
    print("\n" + "=" * 60)
    print("  ScanFlow Detection Summary")
    print("=" * 60)
    print(f"  Total scans scored : {report.n_total}")
    print(f"  Anomalies flagged  : {report.n_flagged} ({report.flag_rate:.1%})")

    if report.metrics:
        m = report.metrics
        print(f"  Accuracy           : {m.get('accuracy', 'N/A')}")
        print(f"  Recall (defect)    : {m.get('recall_defect', 'N/A')}")
        print(f"  F1 (defect)        : {m.get('f1_defect', 'N/A')}")
        print(f"  ROC-AUC            : {m.get('roc_auc', 'N/A')}")

    if report.n_flagged > 0:
        print(f"\n  Top 5 most anomalous:")
        flagged = sorted(report.results, key=lambda r: r.anomaly_score,
                         reverse=True)[:5]
        for r in flagged:
            status = "+" if r.correct else "-" if r.correct is False else "?"
            print(f"    [{status}] {r.scan_id:<40} score={r.anomaly_score:.4f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scanflow",
        description="ScanFlow — Industrial Scan Anomaly Detection Pipeline",
    )
    p.add_argument(
        "command",
        choices=["train", "predict", "run"],
        help="train: fit model | predict: score test set | run: do both",
    )
    p.add_argument(
        "--data-dir", type=Path, required=True,
        help="Root folder of the dataset (contains train/ and final_test/ dirs)",
    )
    p.add_argument(
        "--config", type=Path, default=Path("config.yaml"),
        help="Path to config.yaml (default: ./config.yaml)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory from config",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List files and steps without running the pipeline",
    )
    p.add_argument(
        "--contamination", type=float, default=None,
        help="Override contamination rate for IsolationForest (0.0-0.5)",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)
    cfg.setup_logging()

    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
    if args.contamination is not None:
        cfg.detector.contamination = args.contamination

    cfg.paths.create_dirs()

    log.info(f"ScanFlow starting — command: {args.command}")
    log.info(f"Data dir : {args.data_dir}")
    log.info(f"Output   : {cfg.paths.output_dir}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error(f"Data directory not found: {data_dir}")
        return 1

    try:
        if args.command == "train":
            step_train(cfg, data_dir)

        elif args.command == "predict":
            if _use_patchcore(cfg):
                detector = PatchCoreDetector(cfg).load(cfg.paths.model_dir)
            else:
                detector = AnomalyDetector(cfg).load(cfg.paths.model_dir)
            step_predict(cfg, data_dir, detector, dry_run=args.dry_run)

        elif args.command == "run":
            detector = step_train(cfg, data_dir)
            step_predict(cfg, data_dir, detector, dry_run=args.dry_run)

    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
        return 1
    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
        return 1

    log.info("ScanFlow finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
