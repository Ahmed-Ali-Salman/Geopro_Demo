"""SQLite / GeoPackage database operations.

CRUD operations for detected features with spatial indexing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import create_engine, select, update, delete
from sqlalchemy.engine import Engine

from geopro.storage.models import (
    FeatureCreate,
    FeatureRead,
    FeatureUpdate,
    features_table,
    metadata_obj,
)

logger = logging.getLogger(__name__)


class FeatureDatabase:
    """SQLite database for storing detected features."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
        )
        self._create_tables()

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        metadata_obj.create_all(self.engine)
        logger.info("Database ready at %s", self.db_path)

    def insert_feature(self, feature: FeatureCreate) -> int:
        """Insert a single feature and return its ID."""
        with self.engine.connect() as conn:
            result = conn.execute(
                features_table.insert().values(
                    feature_class=feature.feature_class,
                    label=feature.label,
                    latitude=feature.latitude,
                    longitude=feature.longitude,
                    altitude=feature.altitude,
                    confidence=feature.confidence,
                    source_image=feature.source_image,
                    source_lidar=feature.source_lidar,
                    attributes_json=json.dumps(feature.attributes),
                    ocr_text=feature.ocr_text,
                    ocr_text_ar=feature.ocr_text_ar,
                    ocr_text_en=feature.ocr_text_en,
                    review_status=feature.review_status,
                    notes=feature.notes,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            conn.commit()
            return result.inserted_primary_key[0]

    def insert_features_bulk(self, features: List[FeatureCreate]) -> int:
        """Insert multiple features at once. Returns the count inserted."""
        rows = [
            {
                "feature_class": f.feature_class,
                "label": f.label,
                "latitude": f.latitude,
                "longitude": f.longitude,
                "altitude": f.altitude,
                "confidence": f.confidence,
                "source_image": f.source_image,
                "source_lidar": f.source_lidar,
                "attributes_json": json.dumps(f.attributes),
                "ocr_text": f.ocr_text,
                "ocr_text_ar": f.ocr_text_ar,
                "ocr_text_en": f.ocr_text_en,
                "review_status": f.review_status,
                "notes": f.notes,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            for f in features
        ]
        with self.engine.connect() as conn:
            conn.execute(features_table.insert(), rows)
            conn.commit()

        logger.info("Inserted %d features", len(rows))
        return len(rows)

    def get_feature(self, feature_id: int) -> Optional[FeatureRead]:
        """Get a single feature by ID."""
        with self.engine.connect() as conn:
            row = conn.execute(
                select(features_table).where(features_table.c.id == feature_id)
            ).first()

        if row is None:
            return None
        return _row_to_feature_read(row)

    def get_all_features(
        self,
        feature_class: Optional[str] = None,
        review_status: Optional[str] = None,
        limit: int = 10000,
    ) -> List[FeatureRead]:
        """Get all features, optionally filtered."""
        stmt = select(features_table)
        if feature_class:
            stmt = stmt.where(features_table.c.feature_class == feature_class)
        if review_status:
            stmt = stmt.where(features_table.c.review_status == review_status)
        stmt = stmt.limit(limit)

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        return [_row_to_feature_read(r) for r in rows]

    def update_feature(self, feature_id: int, updates: FeatureUpdate) -> bool:
        """Update a feature's attributes."""
        values = {}
        update_data = updates.model_dump(exclude_none=True)

        for key, val in update_data.items():
            if key == "attributes":
                values["attributes_json"] = json.dumps(val)
            else:
                values[key] = val

        if not values:
            return False

        values["updated_at"] = datetime.utcnow()

        with self.engine.connect() as conn:
            result = conn.execute(
                update(features_table)
                .where(features_table.c.id == feature_id)
                .values(**values)
            )
            conn.commit()
            return result.rowcount > 0

    def delete_feature(self, feature_id: int) -> bool:
        """Delete a feature by ID."""
        with self.engine.connect() as conn:
            result = conn.execute(
                delete(features_table).where(features_table.c.id == feature_id)
            )
            conn.commit()
            return result.rowcount > 0

    def count_features(self, feature_class: Optional[str] = None) -> int:
        """Count features, optionally by class."""
        from sqlalchemy import func
        stmt = select(func.count()).select_from(features_table)
        if feature_class:
            stmt = stmt.where(features_table.c.feature_class == feature_class)
        with self.engine.connect() as conn:
            return conn.execute(stmt).scalar() or 0


def _row_to_feature_read(row) -> FeatureRead:
    """Convert a database row to a FeatureRead model."""
    attrs = {}
    try:
        attrs = json.loads(row.attributes_json) if row.attributes_json else {}
    except (json.JSONDecodeError, TypeError):
        pass

    return FeatureRead(
        id=row.id,
        feature_class=row.feature_class,
        label=row.label or "",
        latitude=row.latitude,
        longitude=row.longitude,
        altitude=row.altitude or 0.0,
        confidence=row.confidence or 0.0,
        source_image=row.source_image or "",
        source_lidar=row.source_lidar or "",
        attributes=attrs,
        ocr_text=row.ocr_text or "",
        ocr_text_ar=row.ocr_text_ar or "",
        ocr_text_en=row.ocr_text_en or "",
        review_status=row.review_status or "pending",
        notes=row.notes or "",
        created_at=str(row.created_at) if row.created_at else "",
        updated_at=str(row.updated_at) if row.updated_at else "",
    )
