"""Feature export — GeoJSON, Shapefile, CSV.

Exports detected features from the database to standard geospatial formats.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Literal, Optional

from geopro.storage.models import FeatureRead

logger = logging.getLogger(__name__)


def export_features(
    features: List[FeatureRead],
    output_path: str | Path,
    fmt: Literal["geojson", "shapefile", "csv"] = "geojson",
) -> Path:
    """Export features to the specified format.

    Parameters
    ----------
    features : list of FeatureRead objects
    output_path : output file/directory path
    fmt : export format

    Returns
    -------
    Path to the created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "geojson":
        return _export_geojson(features, output_path)
    elif fmt == "shapefile":
        return _export_shapefile(features, output_path)
    elif fmt == "csv":
        return _export_csv(features, output_path)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")


def _export_geojson(features: List[FeatureRead], output_path: Path) -> Path:
    """Export as GeoJSON FeatureCollection."""
    if not output_path.suffix:
        output_path = output_path.with_suffix(".geojson")

    geojson = {
        "type": "FeatureCollection",
        "features": [_feature_to_geojson(f) for f in features],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(geojson, fh, indent=2, ensure_ascii=False)

    logger.info("Exported %d features to %s", len(features), output_path)
    return output_path


def _feature_to_geojson(feature: FeatureRead) -> dict:
    """Convert a single feature to a GeoJSON Feature dict."""
    properties = {
        "id": feature.id,
        "feature_class": feature.feature_class,
        "label": feature.label,
        "confidence": feature.confidence,
        "source_image": feature.source_image,
        "source_lidar": feature.source_lidar,
        "ocr_text": feature.ocr_text,
        "ocr_text_ar": feature.ocr_text_ar,
        "ocr_text_en": feature.ocr_text_en,
        "review_status": feature.review_status,
        "notes": feature.notes,
        "created_at": feature.created_at,
    }
    # Flatten attributes into properties
    properties.update(feature.attributes)

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [feature.longitude, feature.latitude, feature.altitude],
        },
        "properties": properties,
    }


def _export_shapefile(features: List[FeatureRead], output_path: Path) -> Path:
    """Export as ESRI Shapefile via geopandas."""
    import geopandas as gpd
    from shapely.geometry import Point

    if not output_path.suffix:
        output_path = output_path.with_suffix(".shp")

    records = []
    for f in features:
        rec = {
            "id": f.id,
            "class": f.feature_class,
            "label": f.label,
            "confidence": f.confidence,
            "ocr_text": f.ocr_text[:254] if f.ocr_text else "",  # Shapefile field limit
            "status": f.review_status,
            "geometry": Point(f.longitude, f.latitude, f.altitude),
        }
        # Add a subset of attributes (Shapefile has field-name length limits)
        for k, v in list(f.attributes.items())[:10]:
            safe_key = k[:10]  # Shapefile 10-char limit
            rec[safe_key] = str(v)[:254] if isinstance(v, str) else v
        records.append(rec)

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf.to_file(str(output_path), driver="ESRI Shapefile", encoding="utf-8")

    logger.info("Exported %d features to %s", len(features), output_path)
    return output_path


def _export_csv(features: List[FeatureRead], output_path: Path) -> Path:
    """Export as CSV with WKT geometry column."""
    if not output_path.suffix:
        output_path = output_path.with_suffix(".csv")

    fieldnames = [
        "id", "feature_class", "label", "latitude", "longitude", "altitude",
        "confidence", "source_image", "source_lidar",
        "ocr_text", "ocr_text_ar", "ocr_text_en",
        "review_status", "notes", "attributes", "wkt_geometry",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for f in features:
            writer.writerow({
                "id": f.id,
                "feature_class": f.feature_class,
                "label": f.label,
                "latitude": f.latitude,
                "longitude": f.longitude,
                "altitude": f.altitude,
                "confidence": f.confidence,
                "source_image": f.source_image,
                "source_lidar": f.source_lidar,
                "ocr_text": f.ocr_text,
                "ocr_text_ar": f.ocr_text_ar,
                "ocr_text_en": f.ocr_text_en,
                "review_status": f.review_status,
                "notes": f.notes,
                "attributes": json.dumps(f.attributes, ensure_ascii=False),
                "wkt_geometry": f"POINT Z ({f.longitude} {f.latitude} {f.altitude})",
            })

    logger.info("Exported %d features to %s", len(features), output_path)
    return output_path
