"""Data models for detected features.

Pydantic models for runtime validation and SQLAlchemy ORM models
for database persistence.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ── Pydantic models (runtime / API / export) ────────────────────────────────


class FeatureBase(BaseModel):
    """Base model for a detected roadside feature."""

    feature_class: str
    label: str = ""
    latitude: float
    longitude: float
    altitude: float = 0.0
    confidence: float = 0.0
    source_image: str = ""
    source_lidar: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    ocr_text: str = ""
    ocr_text_ar: str = ""
    ocr_text_en: str = ""
    review_status: str = "pending"  # pending | accepted | rejected | edited
    notes: str = ""


class FeatureCreate(FeatureBase):
    """Model for creating a new feature."""
    pass


class FeatureUpdate(BaseModel):
    """Model for partially updating a feature."""
    label: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    ocr_text: Optional[str] = None
    ocr_text_ar: Optional[str] = None
    ocr_text_en: Optional[str] = None
    review_status: Optional[str] = None
    notes: Optional[str] = None


class FeatureRead(FeatureBase):
    """Model returned by the API."""
    id: int
    created_at: str = ""
    updated_at: str = ""

    class Config:
        from_attributes = True


# ── SQLAlchemy ORM model ────────────────────────────────────────────────────
# We define the table using raw SQLAlchemy Core to avoid heavy ORM setup
# while keeping things simple for a SQLite / GeoPackage backend.

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)

metadata_obj = MetaData()

features_table = Table(
    "features",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("feature_class", String(50), nullable=False, index=True),
    Column("label", String(100), default=""),
    Column("latitude", Float, nullable=False),
    Column("longitude", Float, nullable=False),
    Column("altitude", Float, default=0.0),
    Column("confidence", Float, default=0.0),
    Column("source_image", String(255), default=""),
    Column("source_lidar", String(255), default=""),
    Column("attributes_json", Text, default="{}"),  # JSON string
    Column("ocr_text", Text, default=""),
    Column("ocr_text_ar", Text, default=""),
    Column("ocr_text_en", Text, default=""),
    Column("review_status", String(20), default="pending"),
    Column("notes", Text, default=""),
    Column("created_at", DateTime, default=datetime.utcnow),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)
