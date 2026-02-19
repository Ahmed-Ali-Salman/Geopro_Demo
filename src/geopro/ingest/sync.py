"""LiDAR ↔ Panoramic Image synchronization.

Associates each panoramic image with the nearest LiDAR scan segment
based on GPS proximity or timestamp matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from geopro.ingest.csv_parser import CameraModel

logger = logging.getLogger(__name__)


@dataclass
class SyncPair:
    """A synchronized LiDAR–image pair."""

    image_filename: str
    lidar_filepath: Path
    camera_model: CameraModel
    distance_m: float  # spatial proximity in metres
    timestamp_delta: Optional[float] = None  # seconds, if available


def synchronize_by_proximity(
    camera_models: Dict[str, CameraModel],
    lidar_files: List[Path],
    lidar_centroids: Optional[Dict[str, np.ndarray]] = None,
    max_distance_m: float = 50.0,
) -> List[SyncPair]:
    """Match each image to the nearest LiDAR file by GPS proximity.

    Parameters
    ----------
    camera_models : filename → CameraModel mapping from CSV
    lidar_files : list of LiDAR file paths
    lidar_centroids : optional pre-computed centroids for each LiDAR file
        (dict mapping filepath.name → np.array([lat, lon]))
        If not provided, centroids are estimated from filenames or file headers.
    max_distance_m : maximum matching distance

    Returns
    -------
    List of SyncPair objects
    """
    if lidar_centroids is None:
        lidar_centroids = _estimate_lidar_centroids(lidar_files)

    pairs: List[SyncPair] = []
    lidar_names = list(lidar_centroids.keys())
    lidar_coords = np.array([lidar_centroids[n] for n in lidar_names])

    for img_name, cam in camera_models.items():
        img_coord = np.array([cam.latitude, cam.longitude])
        # Approximate distance in metres using haversine
        distances = _haversine_batch(img_coord, lidar_coords)
        best_idx = int(np.argmin(distances))
        best_dist = distances[best_idx]

        if best_dist <= max_distance_m:
            # Find the actual file path
            lidar_name = lidar_names[best_idx]
            lidar_path = _find_lidar_path(lidar_name, lidar_files)
            if lidar_path:
                pairs.append(SyncPair(
                    image_filename=img_name,
                    lidar_filepath=lidar_path,
                    camera_model=cam,
                    distance_m=best_dist,
                ))
        else:
            logger.warning(
                "No LiDAR match within %.0fm for image %s (nearest: %.0fm)",
                max_distance_m, img_name, best_dist,
            )

    logger.info("Synchronized %d image↔LiDAR pairs", len(pairs))
    return pairs


def synchronize_by_timestamp(
    camera_models: Dict[str, CameraModel],
    lidar_files: List[Path],
    lidar_timestamps: Dict[str, float],
    max_delta_s: float = 1.0,
) -> List[SyncPair]:
    """Match each image to the nearest LiDAR file by timestamp.

    Parameters
    ----------
    camera_models : filename → CameraModel (must have .timestamp set)
    lidar_files : list of LiDAR file paths
    lidar_timestamps : dict mapping lidar filename → epoch timestamp
    max_delta_s : maximum time difference in seconds

    Returns
    -------
    List of SyncPair objects
    """
    pairs: List[SyncPair] = []
    lidar_names = list(lidar_timestamps.keys())
    lidar_ts = np.array([lidar_timestamps[n] for n in lidar_names])

    for img_name, cam in camera_models.items():
        if cam.timestamp is None:
            logger.debug("No timestamp for image %s, skipping", img_name)
            continue

        deltas = np.abs(lidar_ts - cam.timestamp)
        best_idx = int(np.argmin(deltas))
        best_delta = deltas[best_idx]

        if best_delta <= max_delta_s:
            lidar_name = lidar_names[best_idx]
            lidar_path = _find_lidar_path(lidar_name, lidar_files)
            if lidar_path:
                pairs.append(SyncPair(
                    image_filename=img_name,
                    lidar_filepath=lidar_path,
                    camera_model=cam,
                    distance_m=0.0,
                    timestamp_delta=best_delta,
                ))

    logger.info("Synchronized %d pairs by timestamp", len(pairs))
    return pairs


# ── Helpers ─────────────────────────────────────────────────────────────────

def _haversine_batch(
    point: np.ndarray,  # (2,) — [lat, lon]
    coords: np.ndarray,  # (N, 2) — [[lat, lon], ...]
) -> np.ndarray:
    """Vectorised haversine distance in metres."""
    R = 6_371_000  # Earth radius in metres
    lat1, lon1 = np.radians(point)
    lat2 = np.radians(coords[:, 0])
    lon2 = np.radians(coords[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _estimate_lidar_centroids(lidar_files: List[Path]) -> Dict[str, np.ndarray]:
    """Estimate LiDAR file centroids by reading the first few points."""
    from geopro.ingest.lidar_reader import read_point_cloud

    centroids: Dict[str, np.ndarray] = {}
    for fp in lidar_files:
        try:
            pts = read_point_cloud(fp)
            if isinstance(pts, np.ndarray) and len(pts) > 0:
                # Use mean of X,Y as centroid (assuming projected or geographic coords)
                centroids[fp.name] = np.array([
                    np.mean(pts[:, 1]),  # Y → lat (approximate)
                    np.mean(pts[:, 0]),  # X → lon (approximate)
                ])
        except Exception as exc:
            logger.warning("Could not read centroid for %s: %s", fp.name, exc)

    return centroids


def _find_lidar_path(name: str, files: List[Path]) -> Optional[Path]:
    """Find a LiDAR file path by its basename."""
    for fp in files:
        if fp.name == name:
            return fp
    return None
