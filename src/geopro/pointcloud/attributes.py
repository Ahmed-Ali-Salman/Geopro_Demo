"""Attribute extraction from point cloud clusters.

Per-class geometric measurements: height, trunk diameter,
shape classification, size categories.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from geopro.pointcloud.segmentation import PointCluster

logger = logging.getLogger(__name__)


def extract_height(cluster: PointCluster, ground_z: float = 0.0) -> float:
    """Estimate object height above ground level."""
    return float(cluster.bbox_max[2] - ground_z)


def extract_trunk_diameter(cluster: PointCluster, trunk_height_range: tuple = (0.5, 2.0)) -> Optional[float]:
    """Estimate trunk diameter for tree-like clusters.

    Fits a circle to points within the trunk height range (relative to base).

    Parameters
    ----------
    cluster : point cluster
    trunk_height_range : (min_z_offset, max_z_offset) metres above cluster base

    Returns
    -------
    Estimated trunk diameter in metres, or None if insufficient points.
    """
    base_z = cluster.bbox_min[2]
    min_z = base_z + trunk_height_range[0]
    max_z = base_z + trunk_height_range[1]

    # Select trunk-level points
    mask = (cluster.points[:, 2] >= min_z) & (cluster.points[:, 2] <= max_z)
    trunk_pts = cluster.points[mask]

    if len(trunk_pts) < 5:
        return None

    # Fit a circle in the XY plane
    xy = trunk_pts[:, :2]
    cx, cy = xy.mean(axis=0)
    radii = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    diameter = 2.0 * float(np.median(radii))

    return diameter


def classify_shape(cluster: PointCluster) -> str:
    """Classify the 2D footprint shape of a cluster.

    Used primarily for manholes / utility covers.

    Returns
    -------
    "round", "square", "rectangular", or "irregular"
    """
    xy = cluster.points[:, :2]
    if len(xy) < 4:
        return "irregular"

    # Compute the convex hull and check aspect ratio
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(xy)
    except Exception:
        return "irregular"

    hull_area = hull.volume  # In 2D, ConvexHull.volume gives area
    bbox_area = cluster.footprint_area

    # Circularity: ratio of hull area to the area of the bounding circle
    hull_pts = xy[hull.vertices]
    cx, cy = hull_pts.mean(axis=0)
    max_radius = np.max(np.sqrt((hull_pts[:, 0] - cx) ** 2 + (hull_pts[:, 1] - cy) ** 2))
    circle_area = np.pi * max_radius ** 2

    circularity = hull_area / max(circle_area, 1e-6)
    compactness = hull_area / max(bbox_area, 1e-6)

    # Bounding box aspect ratio
    dims = cluster.bbox_max[:2] - cluster.bbox_min[:2]
    short, long = sorted(dims)
    bb_aspect = short / max(long, 1e-6)

    if circularity > 0.75:
        return "round"
    elif bb_aspect > 0.8 and compactness > 0.85:
        return "square"
    elif bb_aspect > 0.4:
        return "rectangular"
    else:
        return "irregular"


def size_category(cluster: PointCluster) -> str:
    """Classify cluster into a size category based on footprint area."""
    area = cluster.footprint_area
    if area < 0.5:
        return "small"
    elif area < 2.0:
        return "medium"
    else:
        return "large"



def classify_material(cluster: PointCluster) -> str:
    """Classify material based on intensity (if available) or class heuristic."""
    # Heuristics for demo (since intensity varies by sensor)
    if cluster.label == "pole_like":
        return "Metal"
    elif cluster.label == "vegetation_like":
        return "Wood"
    elif cluster.label == "ground_object":
        return "Concrete"  # Manhole default
    
    # Check raw class if available (passed via extraction)
    # Note: This function doesn't receive raw_class_name directly, 
    # but we can infer or it's just a heuristic fallback.
    return "Unknown"


def classify_pole_type(height: float) -> str:
    """Classify pole type based on height."""
    if height > 8.0:
        return "High Mast / Highway"
    elif height > 4.5:
        return "Street Light"
    else:
        return "Utility / Pedestrian"


def extract_all_attributes(
    cluster: PointCluster,
    ground_z: float = 0.0,
    raw_class_name: Optional[str] = None,
) -> Dict[str, object]:
    """Extract all applicable attributes for a cluster.
    
    Parameters
    ----------
    cluster : PointCluster
    ground_z : ground elevation
    raw_class_name : original YOLO class name (e.g., 'stop sign')
    """
    height = float(cluster.bbox_max[2] - ground_z)
    
    attrs: Dict[str, object] = {
        "height_m": round(height, 2),
        "point_count": cluster.point_count,
        "centroid_x": round(float(cluster.centroid[0]), 6),
        "centroid_y": round(float(cluster.centroid[1]), 6),
        "centroid_z": round(float(cluster.centroid[2]), 2),
        "material": classify_material(cluster),
    }

    # Label-dependent attributes
    if cluster.label in ("vegetation_like", "tree"):
        trunk_d = extract_trunk_diameter(cluster)
        if trunk_d is not None:
            attrs["trunk_diameter_m"] = round(trunk_d, 3)

    if cluster.label in ("ground_object", "manhole"):
        attrs["shape"] = classify_shape(cluster)
        attrs["size_category"] = size_category(cluster)
        
    if cluster.label == "pole_like":
        attrs["pole_type"] = classify_pole_type(height)
        
    # Pass-through YOLO details
    if raw_class_name:
        attrs["yolo_class"] = raw_class_name
        if "sign" in raw_class_name:
             attrs["sign_type"] = raw_class_name  # e.g. stop sign

    return attrs
