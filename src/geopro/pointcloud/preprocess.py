"""Point cloud preprocessing.

- Statistical outlier removal
- Voxel downsampling
- Ground plane removal (RANSAC or CSF)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def numpy_to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert an (N, C) numpy array (first 3 cols = XYZ) to an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


def o3d_to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Convert an Open3D PointCloud back to (N, 3) numpy array."""
    return np.asarray(pcd.points)


def remove_outliers(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Remove statistical outliers from a point cloud.

    Parameters
    ----------
    points : (N, C) array, first 3 columns are XYZ
    nb_neighbors : number of neighbours for mean distance calculation
    std_ratio : standard deviation multiplier threshold

    Returns
    -------
    Filtered (M, C) array
    """
    pcd = numpy_to_o3d(points)
    filtered, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    inlier_idx = list(inlier_idx)
    logger.info(
        "Outlier removal: %d → %d points (removed %d)",
        len(points), len(inlier_idx), len(points) - len(inlier_idx),
    )
    return points[inlier_idx]


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """Downsample a point cloud using a voxel grid.

    Parameters
    ----------
    points : (N, C) array
    voxel_size : voxel cube edge length in metres

    Returns
    -------
    Downsampled (M, 3) array  (attributes beyond XYZ are lost)
    """
    pcd = numpy_to_o3d(points)
    downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
    result = o3d_to_numpy(downsampled)
    logger.info(
        "Voxel downsample (%.2fm): %d → %d points",
        voxel_size, len(points), len(result),
    )
    return result


def remove_ground_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.3,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the ground plane using RANSAC.

    Parameters
    ----------
    points : (N, C) array
    distance_threshold : max distance from the plane to be considered inlier
    ransac_n : number of points to sample per iteration
    num_iterations : number of RANSAC iterations

    Returns
    -------
    (non_ground, ground) — both as numpy arrays
    """
    pcd = numpy_to_o3d(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    logger.info(
        "Ground plane: %.3fx + %.3fy + %.3fz + %.3f = 0  (%d inliers)",
        a, b, c, d, len(inliers),
    )

    inlier_set = set(inliers)
    all_indices = np.arange(len(points))
    ground_mask = np.isin(all_indices, list(inlier_set))

    ground = points[ground_mask]
    non_ground = points[~ground_mask]

    logger.info(
        "Ground removal: %d ground, %d non-ground points",
        len(ground), len(non_ground),
    )
    return non_ground, ground


def preprocess_pipeline(
    points: np.ndarray,
    *,
    voxel_size: float = 0.1,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
    ground_distance_threshold: float = 0.3,
    ground_ransac_n: int = 3,
    ground_num_iterations: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: outlier removal → downsample → ground removal.

    Returns
    -------
    (non_ground_points, ground_points)
    """
    logger.info("Starting preprocessing pipeline on %d points", len(points))

    # 1. Outlier removal
    points = remove_outliers(points, outlier_nb_neighbors, outlier_std_ratio)

    # 2. Voxel downsampling
    points = voxel_downsample(points, voxel_size)

    # 3. Ground removal
    non_ground, ground = remove_ground_ransac(
        points,
        distance_threshold=ground_distance_threshold,
        ransac_n=ground_ransac_n,
        num_iterations=ground_num_iterations,
    )

    return non_ground, ground
