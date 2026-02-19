"""2D detection ↔ 3D point cloud cluster matcher.

Matches YOLOv8 2D bounding box detections to 3D clusters by
projecting cluster centroids onto the image plane and computing IoU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from geopro.detection.yolo_detector import Detection
from geopro.fusion.projector import project_3d_to_equirectangular
from geopro.ingest.csv_parser import CameraModel
from geopro.pointcloud.segmentation import PointCluster

logger = logging.getLogger(__name__)


@dataclass
class MatchedFeature:
    """A feature matched between 2D detection and 3D cluster."""

    detection: Optional[Detection]
    cluster: Optional[PointCluster]
    match_score: float  # IoU or proximity score
    class_name: str
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    attributes: Dict[str, object] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


def match_detections_to_clusters(
    detections: List[Detection],
    clusters: List[PointCluster],
    camera: CameraModel,
    image_width: int,
    image_height: int,
    iou_threshold: float = 0.3,
) -> List[MatchedFeature]:
    """Match 2D detections to 3D clusters.

    Algorithm:
    1. Project each cluster centroid onto the 2D image
    2. Create a synthetic bbox around the projected centroid (based on cluster size)
    3. Compute IoU with each detection bbox
    4. Apply Hungarian matching or greedy best-IoU assignment

    Parameters
    ----------
    detections : list of 2D YOLO detections for one image
    clusters : list of 3D point cloud clusters
    camera : camera model for this image
    image_width, image_height : image dimensions
    iou_threshold : minimum IoU to consider a match

    Returns
    -------
    List of MatchedFeature objects
    """
    if not detections or not clusters:
        # Return unmatched detections
        return [
            MatchedFeature(
                detection=d,
                cluster=None,
                match_score=0.0,
                class_name=d.class_name,
            )
            for d in detections
        ]

    # Project all cluster centroids to 2D
    centroids_3d = np.array([c.centroid for c in clusters])
    centroids_2d = project_3d_to_equirectangular(
        centroids_3d, camera, image_width, image_height
    )

    # Build synthetic bboxes around projected centroids
    cluster_bboxes = []
    for i, cluster in enumerate(clusters):
        u, v = centroids_2d[i]
        if np.isnan(u) or np.isnan(v):
            cluster_bboxes.append(None)
            continue

        # Estimate bbox size based on cluster dimensions and distance
        # (simplified: fixed fraction of image size based on point count)
        half_w = max(20, int(image_width * 0.02 * np.log(cluster.point_count + 1)))
        half_h = max(20, int(image_height * 0.02 * np.log(cluster.point_count + 1)))

        cluster_bboxes.append(np.array([
            u - half_w, v - half_h, u + half_w, v + half_h
        ]))

    # Greedy matching: for each detection, find best matching cluster
    matched_features: List[MatchedFeature] = []
    used_clusters = set()

    for det in detections:
        best_iou = 0.0
        best_cluster_idx = -1

        for j, cbbox in enumerate(cluster_bboxes):
            if cbbox is None or j in used_clusters:
                continue

            iou = _compute_iou(det.bbox, cbbox)
            if iou > best_iou:
                best_iou = iou
                best_cluster_idx = j

        # Extract attributes with YOLO context
        attrs = {}
        if clusters and best_cluster_idx >= 0:
            cluster = clusters[best_cluster_idx]
            # Monkey-patch label for attribute logic if needed
            if "pole" in det.class_name or "light" in det.class_name:
                cluster.label = "pole_like"
            elif "tree" in det.class_name:
                cluster.label = "vegetation_like"
            
            # Pass raw class name (e.g. "stop sign") to get sign_type
            raw_name = getattr(det, "raw_class_name", None)
            from geopro.pointcloud.attributes import extract_all_attributes
            attrs = extract_all_attributes(cluster, raw_class_name=raw_name)

        if best_iou >= iou_threshold and best_cluster_idx >= 0:
            used_clusters.add(best_cluster_idx)
            matched_features.append(MatchedFeature(
                detection=det,
                cluster=clusters[best_cluster_idx],
                match_score=best_iou,
                class_name=det.class_name,
                attributes=attrs,
            ))
        else:
            matched_features.append(MatchedFeature(
                detection=det,
                cluster=None,
                match_score=0.0,
                class_name=det.class_name,
            ))

    n_matched = sum(1 for f in matched_features if f.cluster is not None)
    logger.info(
        "Matched %d/%d detections to clusters (IoU ≥ %.2f)",
        n_matched, len(detections), iou_threshold,
    )
    return matched_features


def _compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Compute Intersection over Union of two [x1, y1, x2, y2] bboxes."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)
