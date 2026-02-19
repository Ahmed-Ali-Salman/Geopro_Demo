"""GeoPro configuration loader.

Reads config/default.yaml and exposes typed settings via Pydantic models.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ── Sub-models ──────────────────────────────────────────────────────────────
class PathsConfig(BaseModel):
    lidar_dir: str = "data/lidar"
    images_dir: str = "data/images"
    metadata_csv: str = "data/metadata.csv"
    basemap: str = "data/basemap/middle_east.mbtiles"
    output_db: str = "output/features.gpkg"
    export_dir: str = "output/exports"
    models_dir: str = "models"


class LidarConfig(BaseModel):
    chunk_size: int = 500_000
    target_crs: str = "EPSG:4326"
    voxel_size: float = 0.1
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0


class GroundRemovalConfig(BaseModel):
    method: Literal["ransac", "csf"] = "ransac"
    ransac_distance_threshold: float = 0.3
    ransac_n: int = 3
    ransac_num_iterations: int = 1000


class SegmentationConfig(BaseModel):
    algorithm: Literal["dbscan", "hdbscan"] = "dbscan"
    eps: float = 0.8
    min_samples: int = 15
    min_cluster_points: int = 10


class DetectionConfig(BaseModel):
    model: str = "models/yolo26n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    image_size: int = 1280
    batch_size: int = 4
    device: str = "auto"


class OCRConfig(BaseModel):
    engine: Literal["paddleocr", "surya"] = "paddleocr"
    languages: List[str] = Field(default_factory=lambda: ["ar", "en"])
    use_gpu: bool = False
    confidence_threshold: float = 0.5
    deskew: bool = True
    enhance_contrast: bool = True


class FusionConfig(BaseModel):
    iou_match_threshold: float = 0.3
    max_projection_distance: float = 100.0


class StorageConfig(BaseModel):
    analytics_backend: Literal["none", "duckdb"] = "none"


class UIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False


# ── Root config ─────────────────────────────────────────────────────────────
class PipelineConfig(BaseModel):
    """Full pipeline configuration, loaded from YAML."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    lidar: LidarConfig = Field(default_factory=LidarConfig)
    ground_removal: GroundRemovalConfig = Field(default_factory=GroundRemovalConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/geopro -> Geopro_Demo


def load_config(config_path: Optional[str | Path] = None) -> PipelineConfig:
    """Load pipeline config from a YAML file, falling back to defaults."""
    if config_path is None:
        config_path = _PROJECT_ROOT / "config" / "default.yaml"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return PipelineConfig(**raw)

    return PipelineConfig()
