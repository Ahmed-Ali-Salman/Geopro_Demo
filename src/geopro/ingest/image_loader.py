"""Panoramic image loader.

Loads 360° equirectangular images, validates dimensions,
and extracts EXIF GPS as fallback positioning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class PanoramicImage:
    """Container for a loaded panoramic image with metadata."""

    filepath: Path
    image: np.ndarray  # BGR (OpenCV format), shape (H, W, 3)
    width: int
    height: int
    is_equirectangular: bool = False
    exif_gps: Optional[dict] = field(default=None)


def load_image(filepath: str | Path) -> PanoramicImage:
    """Load a single panoramic image."""
    filepath = Path(filepath)
    if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {filepath.suffix}")

    # Load via OpenCV for processing, PIL for EXIF
    img_bgr = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Failed to load image: {filepath}")

    h, w = img_bgr.shape[:2]
    is_equirect = _check_equirectangular(w, h)

    if not is_equirect:
        logger.warning(
            "%s does not appear to be equirectangular (aspect %.2f, expected ~2.0)",
            filepath.name,
            w / h,
        )

    # Extract EXIF GPS
    gps = _extract_exif_gps(filepath)

    return PanoramicImage(
        filepath=filepath,
        image=img_bgr,
        width=w,
        height=h,
        is_equirectangular=is_equirect,
        exif_gps=gps,
    )


def load_images_from_dir(
    directory: str | Path,
    extensions: Optional[set] = None,
) -> List[PanoramicImage]:
    """Load all panoramic images from a directory."""
    directory = Path(directory)
    exts = extensions or SUPPORTED_EXTENSIONS
    files = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in exts
    )
    logger.info("Found %d image files in %s", len(files), directory)
    return [load_image(f) for f in files]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _check_equirectangular(w: int, h: int, tolerance: float = 0.15) -> bool:
    """Check if the image has a 2:1 aspect ratio (equirectangular)."""
    return abs((w / h) - 2.0) < tolerance


def _extract_exif_gps(filepath: Path) -> Optional[dict]:
    """Extract GPS coordinates from EXIF data."""
    try:
        pil_img = Image.open(filepath)
        exif_data = pil_img._getexif()
        if exif_data is None:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value

        if not gps_info:
            return None

        lat = _dms_to_decimal(
            gps_info.get("GPSLatitude"),
            gps_info.get("GPSLatitudeRef", "N"),
        )
        lon = _dms_to_decimal(
            gps_info.get("GPSLongitude"),
            gps_info.get("GPSLongitudeRef", "E"),
        )

        if lat is not None and lon is not None:
            return {"latitude": lat, "longitude": lon}
    except Exception as exc:
        logger.debug("Could not extract EXIF GPS from %s: %s", filepath.name, exc)

    return None


def _dms_to_decimal(dms, ref: str) -> Optional[float]:
    """Convert EXIF DMS (degrees, minutes, seconds) to decimal degrees."""
    if dms is None:
        return None
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, IndexError, ValueError):
        return None
