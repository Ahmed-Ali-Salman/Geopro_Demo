"""WGS84 georeferencing for detected features.

Assigns geographic coordinates to features using LiDAR coordinates
or camera GPS + bearing estimation as fallback.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from pyproj import Transformer

from geopro.ingest.csv_parser import CameraModel
from geopro.pointcloud.segmentation import PointCluster

logger = logging.getLogger(__name__)

# Transformer for projected → WGS84 conversions
_transformers: dict = {}


def get_transformer(source_crs: str, target_crs: str = "EPSG:4326") -> Transformer:
    """Get or create a CRS transformer."""
    key = (source_crs, target_crs)
    if key not in _transformers:
        _transformers[key] = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return _transformers[key]


def cluster_to_wgs84(
    cluster: PointCluster,
    source_crs: str = "EPSG:4326",
) -> Tuple[float, float, float]:
    """Convert a cluster's centroid to WGS84 (lat, lon, alt).

    Parameters
    ----------
    cluster : point cloud cluster with centroid in the source CRS
    source_crs : EPSG code of the LiDAR data

    Returns
    -------
    (latitude, longitude, altitude)
    """
    if source_crs == "EPSG:4326":
        # Already in WGS84 — X=lon, Y=lat
        return (
            float(cluster.centroid[1]),  # lat
            float(cluster.centroid[0]),  # lon
            float(cluster.centroid[2]),  # alt
        )

    transformer = get_transformer(source_crs)
    lon, lat = transformer.transform(cluster.centroid[0], cluster.centroid[1])
    return (float(lat), float(lon), float(cluster.centroid[2]))


def camera_gps_fallback(
    camera: CameraModel,
    bbox_center_px: np.ndarray,
    image_width: int,
    estimated_distance_m: float = 10.0,
) -> Tuple[float, float, float]:
    """Estimate a feature's GPS position from camera position + bearing.

    Uses the horizontal position of the detection in the panoramic image
    to estimate the bearing from the camera.

    Parameters
    ----------
    camera : camera model with GPS position
    bbox_center_px : (u, v) pixel center of the detection
    image_width : panoramic image width
    estimated_distance_m : rough distance from camera to object

    Returns
    -------
    (latitude, longitude, altitude)
    """
    # In an equirectangular image, horizontal position → azimuth
    u = bbox_center_px[0]
    bearing_rad = (u / image_width) * 2.0 * np.pi  # 0 → 2π

    # Offset from camera GPS
    R = 6_371_000  # Earth radius
    lat1 = np.radians(camera.latitude)
    lon1 = np.radians(camera.longitude)
    d = estimated_distance_m

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d / R) +
        np.cos(lat1) * np.sin(d / R) * np.cos(bearing_rad)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(d / R) * np.cos(lat1),
        np.cos(d / R) - np.sin(lat1) * np.sin(lat2),
    )

    return (
        float(np.degrees(lat2)),
        float(np.degrees(lon2)),
        camera.altitude,
    )


def geolocate_feature(
    cluster: Optional[PointCluster],
    camera: Optional[CameraModel],
    bbox_center_px: Optional[np.ndarray] = None,
    image_width: int = 0,
    source_crs: str = "EPSG:4326",
    estimated_distance_m: float = 10.0,
) -> Tuple[float, float, float]:
    """Assign WGS84 coordinates to a detected feature.

    Strategy:
    1. If a matched 3D cluster exists → use its centroid
    2. Else, use camera GPS + bearing fallback

    Returns
    -------
    (latitude, longitude, altitude)
    """
    if cluster is not None:
        return cluster_to_wgs84(cluster, source_crs)

    if camera is not None and bbox_center_px is not None and image_width > 0:
        return camera_gps_fallback(camera, bbox_center_px, image_width, estimated_distance_m)

    logger.warning("Cannot geolocate feature: no cluster or camera available")
    return (0.0, 0.0, 0.0)
