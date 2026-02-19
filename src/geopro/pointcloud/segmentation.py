"""Point cloud segmentation via clustering.

Segments non-ground points into individual object clusters
using DBSCAN or HDBSCAN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


@dataclass
class PointCluster:
    """A single segmented point cluster representing a potential object."""

    cluster_id: int
    points: np.ndarray            # (M, 3) XYZ
    centroid: np.ndarray           # (3,) XYZ
    bbox_min: np.ndarray           # (3,) XYZ
    bbox_max: np.ndarray           # (3,) XYZ
    point_count: int
    height: float                  # max_z - min_z
    footprint_area: float          # approximate 2D footprint area
    aspect_ratio: float            # height / max(width, depth)
    label: str = "unknown"        # assigned later by classifier


def segment_clusters(
    points: np.ndarray,
    *,
    algorithm: Literal["dbscan", "hdbscan"] = "dbscan",
    eps: float = 0.8,
    min_samples: int = 15,
    min_cluster_points: int = 10,
) -> List[PointCluster]:
    """Segment a non-ground point cloud into object clusters.

    Parameters
    ----------
    points : (N, 3+) array — non-ground points
    algorithm : clustering algorithm
    eps : DBSCAN neighbourhood radius (metres)
    min_samples : minimum points to form a cluster core
    min_cluster_points : discard clusters smaller than this

    Returns
    -------
    List of PointCluster objects
    """
    xyz = points[:, :3]
    logger.info("Segmenting %d points with %s (eps=%.2f)", len(xyz), algorithm, eps)

    if algorithm == "hdbscan":
        try:
            from hdbscan import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=min_samples)
        except ImportError:
            logger.warning("hdbscan not installed, falling back to DBSCAN")
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    labels = clusterer.fit_predict(xyz)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # noise

    clusters: List[PointCluster] = []
    for label_id in sorted(unique_labels):
        mask = labels == label_id
        cluster_pts = xyz[mask]

        if len(cluster_pts) < min_cluster_points:
            continue

        centroid = cluster_pts.mean(axis=0)
        bbox_min = cluster_pts.min(axis=0)
        bbox_max = cluster_pts.max(axis=0)
        dims = bbox_max - bbox_min
        height = dims[2]  # Z extent
        width = dims[0]
        depth = dims[1]
        footprint = width * depth
        max_horiz = max(width, depth, 1e-6)
        aspect = height / max_horiz

        clusters.append(PointCluster(
            cluster_id=int(label_id),
            points=cluster_pts,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            point_count=len(cluster_pts),
            height=height,
            footprint_area=footprint,
            aspect_ratio=aspect,
        ))

    n_noise = int(np.sum(labels == -1))
    logger.info(
        "Segmentation result: %d clusters, %d noise points",
        len(clusters), n_noise,
    )
    return clusters


def classify_clusters_heuristic(
    clusters: List[PointCluster],
    ground_z: float = 0.0,
) -> List[PointCluster]:
    """Apply simple heuristic rules to pre-classify clusters.

    This is a coarse initial classification; it will be refined
    when fused with 2D detection results.

    Rules:
    - Tall + narrow (aspect > 3) → pole-like
    - Tall + wide → tree-like
    - Low + small footprint → ground_object (manhole, etc.)
    """
    for c in clusters:
        abs_height = c.bbox_max[2] - ground_z

        if abs_height > 2.0 and c.aspect_ratio > 3.0:
            c.label = "pole_like"
        elif abs_height > 2.0 and c.aspect_ratio <= 3.0:
            c.label = "vegetation_like"
        elif abs_height < 0.5 and c.footprint_area < 2.0:
            c.label = "ground_object"
        else:
            c.label = "unknown"

    return clusters
