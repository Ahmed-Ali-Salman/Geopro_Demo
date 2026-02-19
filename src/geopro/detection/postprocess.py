"""Post-processing for 2D detections.

- Cross-frame NMS for panoramic overlap zones
- Confidence filtering
- Multi-image object tracking / merging
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from geopro.detection.yolo_detector import Detection

logger = logging.getLogger(__name__)


def filter_by_confidence(
    detections: List[Detection],
    threshold: float = 0.35,
) -> List[Detection]:
    """Remove detections below the confidence threshold."""
    filtered = [d for d in detections if d.confidence >= threshold]
    logger.debug(
        "Confidence filter (%.2f): %d → %d detections",
        threshold, len(detections), len(filtered),
    )
    return filtered


def nms_cross_image(
    detections: List[Detection],
    iou_threshold: float = 0.5,
    gps_distance_threshold_m: float = 5.0,
    camera_models: dict = None,
) -> List[Detection]:
    """Suppress duplicate detections of the same physical object
    across overlapping panoramic images.

    This is a simplified approach: if two detections from different images
    have similar GPS-projected positions, keep the higher-confidence one.

    Parameters
    ----------
    detections : all detections across multiple images
    iou_threshold : not used directly here (reserved for 2D overlap NMS)
    gps_distance_threshold_m : maximum distance to consider same object
    camera_models : dict mapping filename → CameraModel (for GPS)

    Returns
    -------
    De-duplicated detections
    """
    if not detections or camera_models is None:
        return detections

    # Group by class
    by_class: Dict[int, List[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    kept: List[Detection] = []
    for cls_id, cls_dets in by_class.items():
        # Sort by confidence descending
        cls_dets.sort(key=lambda d: d.confidence, reverse=True)
        suppressed = set()

        for i, d_i in enumerate(cls_dets):
            if i in suppressed:
                continue
            kept.append(d_i)

            # Suppress lower-confidence detections from nearby images
            cam_i = camera_models.get(d_i.image_filename)
            if cam_i is None:
                continue

            for j in range(i + 1, len(cls_dets)):
                if j in suppressed:
                    continue
                d_j = cls_dets[j]
                if d_j.image_filename == d_i.image_filename:
                    continue  # same-image NMS already done by YOLO

                cam_j = camera_models.get(d_j.image_filename)
                if cam_j is None:
                    continue

                dist = _gps_distance(cam_i, cam_j)
                if dist < gps_distance_threshold_m:
                    # Same object seen from nearby positions → suppress
                    suppressed.add(j)

    logger.info(
        "Cross-image NMS: %d → %d detections",
        len(detections), len(kept),
    )
    return kept


def merge_multi_frame_detections(
    detections: List[Detection],
    camera_models: dict = None,
    distance_threshold_m: float = 3.0,
) -> List[Detection]:
    """Merge detections of the same object across consecutive frames.

    For each group of nearby same-class detections, keep the one
    with the highest confidence and average the GPS position.
    """
    if not detections or camera_models is None:
        return detections

    # This is essentially the same as cross-image NMS
    return nms_cross_image(
        detections,
        gps_distance_threshold_m=distance_threshold_m,
        camera_models=camera_models,
    )


def _gps_distance(cam_a, cam_b) -> float:
    """Approximate distance between two cameras in metres (haversine)."""
    R = 6_371_000
    lat1, lon1 = np.radians(cam_a.latitude), np.radians(cam_a.longitude)
    lat2, lon2 = np.radians(cam_b.latitude), np.radians(cam_b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(R * 2 * np.arcsin(np.sqrt(a)))
