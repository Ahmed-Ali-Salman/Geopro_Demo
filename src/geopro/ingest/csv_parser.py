"""Metadata CSV parser.

Parses the project metadata CSV that links each image to its camera model
(GPS position, intrinsic parameters, extrinsic orientation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum required columns in the CSV
REQUIRED_COLUMNS = {
    "image_filename",
    "latitude",
    "longitude",
}

# Additional expected columns (not strictly required)
EXPECTED_COLUMNS = {
    "altitude",
    "fx", "fy", "cx", "cy",       # Intrinsic parameters
    "k1", "k2", "p1", "p2",       # Distortion coefficients
    "r11", "r12", "r13",           # Rotation matrix row 1
    "r21", "r22", "r23",           # Rotation matrix row 2
    "r31", "r32", "r33",           # Rotation matrix row 3
    "tx", "ty", "tz",              # Translation vector
    "timestamp",
}


@dataclass
class CameraModel:
    """Camera model for a single panoramic image."""

    image_filename: str
    latitude: float
    longitude: float
    altitude: float = 0.0

    # Intrinsic parameters
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    distortion: np.ndarray = None  # [k1, k2, p1, p2]

    # Extrinsic parameters
    rotation: np.ndarray = None    # 3x3 rotation matrix
    translation: np.ndarray = None  # 3x1 translation vector

    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.distortion is None:
            self.distortion = np.zeros(4, dtype=np.float64)
        if self.rotation is None:
            self.rotation = np.eye(3, dtype=np.float64)
        if self.translation is None:
            self.translation = np.zeros(3, dtype=np.float64)

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix K."""
        return np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ],
        ], dtype=np.float64)

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """3x4 extrinsic matrix [R | t]."""
        return np.hstack([self.rotation, self.translation.reshape(3, 1)])

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 projection matrix P = K @ [R | t]."""
        return self.intrinsic_matrix @ self.extrinsic_matrix

    @property
    def gps_position(self) -> tuple:
        """(latitude, longitude, altitude)."""
        return (self.latitude, self.longitude, self.altitude)


def parse_metadata_csv(
    csv_path: str | Path,
    *,
    validate: bool = True,
) -> Dict[str, CameraModel]:
    """Parse the metadata CSV and return a dict mapping filename → CameraModel.

    Parameters
    ----------
    csv_path : path to the CSV file
    validate : if True, check for required columns and log warnings

    Returns
    -------
    Dict mapping image_filename → CameraModel
    """
    csv_path = Path(csv_path)
    logger.info("Parsing metadata CSV: %s", csv_path.name)

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Validate
    if validate:
        _validate_columns(df)

    result: Dict[str, CameraModel] = {}
    for _, row in df.iterrows():
        cam = _row_to_camera_model(row)
        if cam:
            result[cam.image_filename] = cam

    logger.info("Parsed %d camera models from CSV", len(result))
    return result


def _validate_columns(df: pd.DataFrame) -> None:
    """Check that required columns are present."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata CSV missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Warn about missing optional columns
    optional_missing = EXPECTED_COLUMNS - set(df.columns)
    if optional_missing:
        logger.warning(
            "Metadata CSV missing optional columns (defaults will be used): %s",
            optional_missing,
        )


def _row_to_camera_model(row: pd.Series) -> Optional[CameraModel]:
    """Convert a single CSV row to a CameraModel."""
    try:
        filename = str(row["image_filename"]).strip()
        lat = float(row["latitude"])
        lon = float(row["longitude"])
    except (KeyError, ValueError) as exc:
        logger.warning("Skipping invalid row: %s", exc)
        return None

    cam = CameraModel(
        image_filename=filename,
        latitude=lat,
        longitude=lon,
        altitude=_safe_float(row, "altitude"),
    )

    # Intrinsics
    cam.fx = _safe_float(row, "fx")
    cam.fy = _safe_float(row, "fy")
    cam.cx = _safe_float(row, "cx")
    cam.cy = _safe_float(row, "cy")
    cam.distortion = np.array([
        _safe_float(row, "k1"),
        _safe_float(row, "k2"),
        _safe_float(row, "p1"),
        _safe_float(row, "p2"),
    ], dtype=np.float64)

    # Extrinsics — rotation
    r_vals = [_safe_float(row, f"r{i}{j}") for i in range(1, 4) for j in range(1, 4)]
    if any(v != 0.0 for v in r_vals):
        cam.rotation = np.array(r_vals, dtype=np.float64).reshape(3, 3)

    # Extrinsics — translation
    t_vals = [_safe_float(row, k) for k in ("tx", "ty", "tz")]
    if any(v != 0.0 for v in t_vals):
        cam.translation = np.array(t_vals, dtype=np.float64)

    # Timestamp
    cam.timestamp = _safe_float(row, "timestamp") or None

    return cam


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    """Safely extract a float value from a pandas Series."""
    try:
        val = row[key]
        if pd.isna(val):
            return default
        return float(val)
    except (KeyError, ValueError, TypeError):
        return default
