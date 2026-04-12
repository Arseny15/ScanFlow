"""
Typed configuration loaded from config.yaml.
Uses dataclasses so every field is explicit, typed, and IDE-friendly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PipelineConfig:
    image_size: int = 224
    grayscale: bool = True
    normalise: bool = True


@dataclass
class DetectorConfig:
    algorithm: str = "isolation_forest"
    contamination: float = 0.1
    n_estimators: int = 200
    random_state: int = 42
    threshold_percentile: int = 10


@dataclass
class FeatureConfig:
    method: str = "cnn"
    cnn_model: str = "resnet18"
    row_stats: bool = True
    col_stats: bool = True
    global_stats: bool = True
    texture: bool = True


@dataclass
class PathConfig:
    data_dir: Path = Path("data_scan")
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("outputs/models")
    reports_dir: Path = Path("outputs/reports")

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.model_dir = Path(self.model_dir)
        self.reports_dir = Path(self.reports_dir)

    def create_dirs(self) -> None:
        for p in [self.output_dir, self.model_dir, self.reports_dir]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "outputs/scanflow.log"


@dataclass
class AppConfig:
    """Top-level config — single object passed through the whole pipeline."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "AppConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return cls(
            pipeline=PipelineConfig(**raw.get("pipeline", {})),
            detector=DetectorConfig(**raw.get("detector", {})),
            features=FeatureConfig(**raw.get("features", {})),
            paths=PathConfig(**raw.get("paths", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )

    def setup_logging(self) -> None:
        Path(self.logging.file).parent.mkdir(parents=True, exist_ok=True)
        level = getattr(logging, self.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.logging.file),
            ],
        )
