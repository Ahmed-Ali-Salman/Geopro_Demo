"""Unified LiDAR point cloud reader.

Supports .las, .laz (via laspy), .ply, .pcd (via Open3D).
Outputs a consistent numpy array of shape (N, C) with at least X,Y,Z columns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Column layout for the unified output array
COLUMNS = ["x", "y", "z", "intensity", "return_number", "classification"]


def read_point_cloud(
    filepath: str | Path,
    *,
    chunk_size: Optional[int] = None,
) -> np.ndarray | Generator[np.ndarray, None, None]:
    """Read a point cloud file and return an (N, C) numpy array.

    Parameters
    ----------
    filepath : path to LiDAR file (.las, .laz, .ply, .pcd)
    chunk_size : if set, yield chunks of this size (only for .las/.laz)

    Returns
    -------
    np.ndarray of shape (N, C) or a generator of such arrays.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in (".las", ".laz"):
        if chunk_size:
            return _read_las_chunked(filepath, chunk_size)
        return _read_las(filepath)
    elif suffix == ".ply":
        return _read_open3d(filepath)
    elif suffix == ".pcd":
        return _read_open3d(filepath)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")


def get_las_crs(filepath: str | Path) -> Optional[str]:
    """Extract CRS / EPSG from a LAS/LAZ file header VLRs."""
    import laspy

    filepath = Path(filepath)
    with laspy.open(str(filepath)) as reader:
        for vlr in reader.header.vlrs:
            # GeoTIFF GeoKeyDirectoryTag
            if vlr.record_id == 34735:
                # Simplified: look for EPSG in the record data
                pass
        # Try to read CRS from WKT VLR (record_id 2112)
        for vlr in reader.header.vlrs:
            if vlr.record_id == 2112:
                try:
                    wkt = vlr.record_data.decode("utf-8", errors="ignore").strip("\x00")
                    return wkt
                except Exception:
                    pass
    logger.warning("No CRS information found in %s", filepath.name)
    return None


# ── Private readers ─────────────────────────────────────────────────────────

def _read_las(filepath: Path) -> np.ndarray:
    """Read an entire LAS/LAZ file into memory."""
    import laspy

    logger.info("Reading LAS file: %s", filepath.name)
    las = laspy.read(str(filepath))
    return _las_to_array(las)


def _read_las_chunked(
    filepath: Path, chunk_size: int
) -> Generator[np.ndarray, None, None]:
    """Stream a LAS/LAZ file in chunks to manage memory."""
    import laspy

    logger.info("Streaming LAS file in chunks of %d: %s", chunk_size, filepath.name)
    with laspy.open(str(filepath)) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            yield _las_to_array(chunk)


def _las_to_array(las) -> np.ndarray:
    """Convert a laspy PointRecord to a numpy array with standard columns."""
    n = len(las.x)
    arr = np.zeros((n, len(COLUMNS)), dtype=np.float64)
    arr[:, 0] = las.x
    arr[:, 1] = las.y
    arr[:, 2] = las.z

    # Optional fields — not all LAS files have these
    try:
        arr[:, 3] = las.intensity
    except AttributeError:
        pass
    try:
        arr[:, 4] = las.return_number
    except AttributeError:
        pass
    try:
        arr[:, 5] = las.classification
    except AttributeError:
        pass

    logger.debug("Loaded %d points from LAS", n)
    return arr


def _read_open3d(filepath: Path) -> np.ndarray:
    """Read a PLY or PCD file via Open3D."""
    import open3d as o3d

    logger.info("Reading %s file: %s", filepath.suffix.upper(), filepath.name)
    pcd = o3d.io.read_point_cloud(str(filepath))
    points = np.asarray(pcd.points)  # (N, 3)

    # Pad to standard column count
    n = points.shape[0]
    arr = np.zeros((n, len(COLUMNS)), dtype=np.float64)
    arr[:, :3] = points

    # If colors are present, we could store them but they don't map to our columns
    logger.debug("Loaded %d points from %s", n, filepath.suffix)
    return arr
