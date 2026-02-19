"""3D ↔ 2D projection utilities.

Projects LiDAR points onto equirectangular panoramic images and
back-projects 2D bounding boxes into 3D frustums.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from geopro.ingest.csv_parser import CameraModel

logger = logging.getLogger(__name__)


def project_3d_to_equirectangular(
    points_3d: np.ndarray,
    camera: CameraModel,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Project 3D world points onto an equirectangular (360°) image.

    Parameters
    ----------
    points_3d : (N, 3) XYZ array in world / LiDAR coordinates
    camera : camera model with extrinsics
    image_width, image_height : panoramic image dimensions

    Returns
    -------
    (N, 2) array of [u, v] pixel coordinates. Points behind the camera
    or outside the image will have NaN values.
    """
    N = len(points_3d)
    uv = np.full((N, 2), np.nan, dtype=np.float64)

    # Transform to camera coordinate system
    R = camera.rotation
    t = camera.translation
    pts_cam = (R @ points_3d.T).T + t  # (N, 3)

    # Convert to spherical coordinates
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Avoid division by zero
    valid = r > 1e-6
    theta = np.full(N, np.nan)
    phi = np.full(N, np.nan)

    # Azimuth (longitude): atan2(x, z) → [−π, π]
    theta[valid] = np.arctan2(x[valid], z[valid])
    # Elevation (latitude): asin(y / r) → [−π/2, π/2]
    phi[valid] = np.arcsin(np.clip(y[valid] / r[valid], -1, 1))

    # Map to pixel coordinates
    # u = (θ / π + 1) / 2 * width    → 0..width
    # v = (1 - φ / (π/2)) / 2 * height → 0..height
    u = (theta / np.pi + 1.0) / 2.0 * image_width
    v = (1.0 - phi / (np.pi / 2.0)) / 2.0 * image_height

    # Mark valid projections
    in_bounds = valid & (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    uv[in_bounds, 0] = u[in_bounds]
    uv[in_bounds, 1] = v[in_bounds]

    return uv


def project_3d_to_pinhole(
    points_3d: np.ndarray,
    camera: CameraModel,
) -> np.ndarray:
    """Project 3D world points using a standard pinhole camera model.

    Parameters
    ----------
    points_3d : (N, 3) XYZ
    camera : camera model with intrinsics and extrinsics

    Returns
    -------
    (N, 2) pixel coordinates. Invalid projections have NaN.
    """
    N = len(points_3d)
    uv = np.full((N, 2), np.nan, dtype=np.float64)

    # Homogeneous coordinates (N, 4)
    pts_h = np.hstack([points_3d, np.ones((N, 1))])

    # Project: P = K @ [R | t] @ X
    P = camera.projection_matrix  # (3, 4)
    projected = (P @ pts_h.T).T  # (N, 3)

    # Normalise
    valid = projected[:, 2] > 1e-6  # in front of camera
    uv[valid, 0] = projected[valid, 0] / projected[valid, 2]
    uv[valid, 1] = projected[valid, 1] / projected[valid, 2]

    return uv


def backproject_bbox_to_frustum(
    bbox: np.ndarray,
    camera: CameraModel,
    image_width: int,
    image_height: int,
    depth_range: Tuple[float, float] = (1.0, 100.0),
) -> np.ndarray:
    """Back-project a 2D bounding box into a 3D frustum.

    Returns 8 corners of the frustum in world coordinates.

    Parameters
    ----------
    bbox : [x1, y1, x2, y2] pixel coordinates
    camera : camera model
    image_width, image_height : image dimensions
    depth_range : (near, far) in metres

    Returns
    -------
    (8, 3) array of frustum corner points in world coords
    """
    x1, y1, x2, y2 = bbox[:4]
    near, far = depth_range

    # 4 corner pixels
    corners_2d = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype=np.float64)

    # Back-project to normalised camera rays
    K_inv = np.linalg.inv(camera.intrinsic_matrix)
    corners_h = np.hstack([corners_2d, np.ones((4, 1))])
    rays_cam = (K_inv @ corners_h.T).T  # (4, 3)

    # Normalise ray directions
    ray_norms = np.linalg.norm(rays_cam, axis=1, keepdims=True)
    rays_cam = rays_cam / ray_norms

    # Generate near and far plane points
    R_inv = camera.rotation.T
    t = camera.translation

    frustum_points = np.zeros((8, 3), dtype=np.float64)
    for i, ray in enumerate(rays_cam):
        # Near point
        pt_near_cam = ray * near
        pt_near_world = R_inv @ (pt_near_cam - t)
        frustum_points[i] = pt_near_world

        # Far point
        pt_far_cam = ray * far
        pt_far_world = R_inv @ (pt_far_cam - t)
        frustum_points[i + 4] = pt_far_world

    return frustum_points
