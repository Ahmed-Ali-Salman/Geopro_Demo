"""Feature class mapping and attribute schema.

Maps YOLO model class IDs to GeoPro feature classes and defines
the expected attribute schema for each class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class FeatureClass:
    """Definition of a single feature class."""

    name: str
    label: str
    yolo_class_ids: List[int]
    attributes: Dict[str, Dict[str, Any]]


class FeatureClassRegistry:
    """Registry that maps YOLO class IDs to GeoPro feature classes."""

    def __init__(self, config_path: Optional[str | Path] = None):
        if config_path is None:
            config_path = _PROJECT_ROOT / "config" / "feature_classes.yaml"

        self._classes: Dict[str, FeatureClass] = {}
        self._yolo_to_feature: Dict[int, str] = {}
        self._load(Path(config_path))

    def _load(self, path: Path) -> None:
        """Load feature class definitions from YAML."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        for name, defn in raw.get("feature_classes", {}).items():
            fc = FeatureClass(
                name=name,
                label=defn.get("label", name),
                yolo_class_ids=defn.get("yolo_class_ids", []),
                attributes=defn.get("attributes", {}),
            )
            self._classes[name] = fc
            for yolo_id in fc.yolo_class_ids:
                self._yolo_to_feature[yolo_id] = name

        logger.info("Loaded %d feature classes", len(self._classes))

    def get_by_name(self, name: str) -> Optional[FeatureClass]:
        return self._classes.get(name)

    def get_by_yolo_id(self, yolo_id: int) -> Optional[FeatureClass]:
        name = self._yolo_to_feature.get(yolo_id)
        if name:
            return self._classes[name]
        return None

    def map_yolo_class(self, yolo_id: int) -> str:
        """Map a YOLO class ID to a GeoPro feature class name."""
        return self._yolo_to_feature.get(yolo_id, "unknown")

    @property
    def all_classes(self) -> Dict[str, FeatureClass]:
        return dict(self._classes)
