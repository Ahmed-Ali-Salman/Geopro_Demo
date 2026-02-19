"""GeoPro CLI — Command-line interface for the pipeline.

Commands:
  geopro ingest   — Validate & load input data
  geopro detect   — Run detection + OCR pipeline
  geopro export   — Export results to GeoJSON/Shapefile/CSV
  geopro serve    — Launch the review UI
  geopro run      — Execute full pipeline end-to-end
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="geopro",
    help="GeoPro Demo — Roadside Asset Detection Pipeline",
    add_completion=False,
)
console = Console()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def ingest(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Validate and load input data (LiDAR, images, metadata CSV)."""
    _setup_logging(verbose)
    from geopro.config import load_config

    cfg = load_config(config)
    lidar_dir = _PROJECT_ROOT / cfg.paths.lidar_dir
    images_dir = _PROJECT_ROOT / cfg.paths.images_dir
    csv_path = _PROJECT_ROOT / cfg.paths.metadata_csv

    console.print("\n[bold cyan]📂 Data Ingestion[/bold cyan]\n")

    # Validate paths
    errors = []
    if not lidar_dir.exists():
        errors.append(f"LiDAR directory not found: {lidar_dir}")
    if not images_dir.exists():
        errors.append(f"Images directory not found: {images_dir}")
    if not csv_path.exists():
        errors.append(f"Metadata CSV not found: {csv_path}")

    if errors:
        for e in errors:
            console.print(f"  [red]✗[/red] {e}")
        raise typer.Exit(1)

    # Count files
    from geopro.ingest.lidar_reader import COLUMNS
    lidar_files = list(lidar_dir.glob("*.la[sz]")) + list(lidar_dir.glob("*.ply")) + list(lidar_dir.glob("*.pcd"))
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

    console.print(f"  [green]✓[/green] LiDAR files: {len(lidar_files)}")
    console.print(f"  [green]✓[/green] Image files: {len(image_files)}")

    # Parse CSV
    from geopro.ingest.csv_parser import parse_metadata_csv
    cameras = parse_metadata_csv(csv_path)
    console.print(f"  [green]✓[/green] Camera models: {len(cameras)}")
    console.print("\n[green]Ingestion validation complete![/green]")


@app.command()
def detect(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the full detection + OCR pipeline."""
    _setup_logging(verbose)
    from geopro.config import load_config

    cfg = load_config(config)
    console.print("\n[bold cyan]🔍 Detection Pipeline[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # 1. Load data
        task = progress.add_task("Loading input data...", total=None)
        from geopro.ingest.csv_parser import parse_metadata_csv
        from geopro.ingest.image_loader import load_images_from_dir
        import numpy as np

        csv_path = _PROJECT_ROOT / cfg.paths.metadata_csv
        images_dir = _PROJECT_ROOT / cfg.paths.images_dir
        lidar_dir = _PROJECT_ROOT / cfg.paths.lidar_dir

        cameras = parse_metadata_csv(csv_path)
        images = load_images_from_dir(images_dir)
        lidar_files = (
            list(lidar_dir.glob("*.la[sz]"))
            + list(lidar_dir.glob("*.ply"))
            + list(lidar_dir.glob("*.pcd"))
        )
        progress.update(task, description=f"Loaded {len(images)} images, {len(lidar_files)} LiDAR files")

        # 2. 2D Detection
        progress.update(task, description="Running YOLO detection...")
        from geopro.detection.yolo_detector import YOLODetector

        model_path = _PROJECT_ROOT / cfg.detection.model
        if not model_path.exists():
            console.print(f"  [yellow]⚠[/yellow] Model not found at {model_path}")
            console.print("  [yellow]  Using default YOLO26n model[/yellow]")
            model_path = "yolo26n.pt"

        detector = YOLODetector(
            model_path=str(model_path),
            confidence_threshold=cfg.detection.confidence_threshold,
            iou_threshold=cfg.detection.iou_threshold,
            image_size=cfg.detection.image_size,
            device=cfg.detection.device,
        )

        all_detections = []
        for img in images:
            dets = detector.detect(img.image, img.filepath.name)
            all_detections.extend(dets)
        progress.update(task, description=f"Detected {len(all_detections)} objects")

        # 3. OCR on sign detections
        progress.update(task, description="Running OCR on sign regions...")
        from geopro.ocr.engine import OCREngine
        from geopro.ocr.text_cleanup import crop_detection_region, preprocess_for_ocr
        from geopro.ocr.arabic_postproc import postprocess_arabic

        ocr_engine = OCREngine(
            engine=cfg.ocr.engine,
            languages=cfg.ocr.languages,
            use_gpu=cfg.ocr.use_gpu,
            confidence_threshold=cfg.ocr.confidence_threshold,
        )

        sign_classes = {"traffic_sign", "road_sign", "traffic_light", "vehicle"}
        
        # Prioritize larger detections for OCR (likely closer/more legible)
        # Sort all detections once by area descending
        def get_area(d):
            if hasattr(d, 'bbox'):
                return (d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1])
            return 0
        
        sorted_detections = sorted(all_detections, key=get_area, reverse=True)
        
        # Track counts per (image, class)
        ocr_counts = {}

        for det in sorted_detections:
            if det.class_name in sign_classes:
                # Find the source image
                src_img = next((img for img in images if img.filepath.name == det.image_filename), None)
                if src_img is not None:
                    # Limit OCR on vehicles/lights to avoid processing thousands in demo
                    if det.class_name in ("vehicle", "traffic_light"):
                        key = (det.image_filename, det.class_name)
                        count = ocr_counts.get(key, 0)
                        if count >= 10: # Increased limit to 10
                            continue
                        ocr_counts[key] = count + 1
                            
                    crop = crop_detection_region(src_img.image, det.bbox)
                    crop = preprocess_for_ocr(crop, deskew=cfg.ocr.deskew, enhance_contrast=cfg.ocr.enhance_contrast)
                    ocr_results = ocr_engine.recognize(crop)
                    ocr_results = postprocess_arabic(ocr_results)
                    det._ocr_results = ocr_results

        # 4. LiDAR processing
        progress.update(task, description="Processing point clouds...")
        from geopro.ingest.lidar_reader import read_point_cloud
        from geopro.pointcloud.preprocess import preprocess_pipeline
        from geopro.pointcloud.segmentation import segment_clusters, classify_clusters_heuristic

        all_clusters = []
        for lf in lidar_files:
            pts = read_point_cloud(lf)
            if hasattr(pts, '__next__'):
                import numpy as np
                pts = np.vstack(list(pts))
            non_ground, ground = preprocess_pipeline(
                pts,
                voxel_size=cfg.lidar.voxel_size,
                outlier_nb_neighbors=cfg.lidar.outlier_nb_neighbors,
                outlier_std_ratio=cfg.lidar.outlier_std_ratio,
            )
            clusters = segment_clusters(
                non_ground,
                algorithm=cfg.segmentation.algorithm,
                eps=cfg.segmentation.eps,
                min_samples=cfg.segmentation.min_samples,
                min_cluster_points=cfg.segmentation.min_cluster_points,
            )
            clusters = classify_clusters_heuristic(clusters)
            all_clusters.extend(clusters)
        progress.update(task, description=f"Segmented {len(all_clusters)} clusters")

        # 5. Fusion & georeferencing
        progress.update(task, description="Fusing 2D and 3D detections...")
        from geopro.fusion.matcher import match_detections_to_clusters
        from geopro.fusion.geolocation import geolocate_feature
        from geopro.pointcloud.attributes import extract_all_attributes
        from geopro.storage.models import FeatureCreate
        from geopro.storage.database import FeatureDatabase
        
        # Ensure raw_class_name is available on Detection object
        # It is monkey-patched in yolo_detector.py

        db = FeatureDatabase(_PROJECT_ROOT / cfg.paths.output_db)
        features_to_insert = []

        # Track used clusters to avoid duplication and find unmatched ones
        used_cluster_ids = set()

        for img in images:
            img_dets = [d for d in all_detections if d.image_filename == img.filepath.name]
            cam = cameras.get(img.filepath.name)
            if not cam or not img_dets:
                continue

            matched = match_detections_to_clusters(
                img_dets, all_clusters, cam,
                img.width, img.height,
                iou_threshold=cfg.fusion.iou_match_threshold,
            )

            for mf in matched:
                if mf.cluster:
                    used_cluster_ids.add(id(mf.cluster))

                # Handle 3D-only features (no detection) - though here mf always has detection if from matcher
                # ... (rest of the loop logic I modified earlier) ...
                
                # ... (I need to keep the content I wrote in step 1914 but wrap it) ...
                # Actually, I should just modify the loop start and end.
                
                # RE-INSERTING THE LOOP BODY FROM PREVIOUS STEP TO KEEP CONTEXT
                # Handle 3D-only features from matcher? 
                # Wait, matcher ONLY returns fused items now (reverted). 
                # So "mf.detection" is always present in this loop 
                # BUT I updated MatchedFeature to be Optional. 
                # And I updated cli.py to handle None.
                # So the previous code is fine, but effectively mf.detection will rarely be None here 
                # unless matcher returns it (which it doesn't anymore).
                
                bbox_center = getattr(mf.detection, "bbox_center", None)
                conf = getattr(mf.detection, "confidence", 0.9)
                src_img_name = getattr(mf.detection, "image_filename", img.filepath.name)

                lat, lon, alt = geolocate_feature(
                    cluster=mf.cluster,
                    camera=cam,
                    bbox_center_px=bbox_center,
                    image_width=img.width,
                    source_crs=cfg.lidar.target_crs,
                )

                attrs = mf.attributes or {}
                if not attrs and mf.cluster:
                    raw_cls = getattr(mf.detection, "raw_class_name", None)
                    attrs = extract_all_attributes(mf.cluster, raw_class_name=raw_cls)
                
                # Check for support pole if it's a sign
                if mf.detection and "sign" in mf.class_name:
                    attrs["has_support_pole"] = (mf.cluster and mf.cluster.label == "pole_like")
                
                # Add 2D bbox for UI cropping (convert numpy to list)
                if mf.detection and hasattr(mf.detection, 'bbox'):
                    attrs['bbox'] = str(list(mf.detection.bbox))

                ocr_ar = ""
                ocr_en = ""
                ocr_text = ""
                if mf.detection and hasattr(mf.detection, '_ocr_results') and mf.detection._ocr_results:
                    for ocr_r in mf.detection._ocr_results:
                        if ocr_r.language == "ar":
                            ocr_ar += ocr_r.text + " "
                        else:
                            ocr_en += ocr_r.text + " "
                        ocr_text += ocr_r.text + " "

                features_to_insert.append(FeatureCreate(
                    feature_class=mf.class_name,
                    label=mf.class_name,
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    confidence=conf,
                    source_image=src_img_name,
                    attributes=attrs,
                    ocr_text=ocr_text.strip(),
                    ocr_text_ar=ocr_ar.strip(),
                    ocr_text_en=ocr_en.strip(),
                ))
        
        # ---------------------------------------------------------
        # Promote unmatched 3D clusters (Global Pass)
        # ---------------------------------------------------------
        progress.update(task, description="Promoting unmatched 3D features...")
        for cluster in all_clusters:
            if id(cluster) in used_cluster_ids:
                continue
                
            # Heuristic classification for 3D-only
            feature_class = None
            if cluster.label == "vegetation_like":
                feature_class = "tree"
            elif cluster.label == "pole_like":
                height = cluster.bbox_max[2] - cluster.bbox_min[2]
                feature_class = "street_light" if height > 4.5 else "pole"
            elif cluster.label == "ground_object":
                feature_class = "manhole"
                
            if feature_class:
                # Geolocate using cluster centroid
                lat, lon, alt = geolocate_feature(
                    cluster=cluster,
                    camera=None,
                    source_crs=cfg.lidar.target_crs
                )
                
                attrs = extract_all_attributes(cluster, raw_class_name=feature_class)

                # ---------------------------------------------------------
                # NEW: Project 3D cluster to available images to find a crop
                # ---------------------------------------------------------
                best_image = ""
                best_bbox = None
                min_dist_sq = float('inf')
                
                # Check a subset of images or all? checking all 39 is fast enough
                # But we need cluster center.
                center_3d = (cluster.bbox_min + cluster.bbox_max) / 2.0
                
                from geopro.fusion.projector import project_3d_to_equirectangular
                
                for img in images:
                    cam = cameras.get(img.filepath.name)
                    if not cam:
                        continue
                        
                    # Quick check: is cluster roughly in front? 
                    # Equirectangular covers 360, so mostly yes, but we want 'best'
                    # dist to camera
                    dist_sq = np.sum((center_3d - cam.translation)**2)
                    
                    if dist_sq < min_dist_sq:
                        # Try projecting the 8 corners of the 3D bbox
                        corners_3d = np.array([
                            [cluster.bbox_min[0], cluster.bbox_min[1], cluster.bbox_min[2]],
                            [cluster.bbox_max[0], cluster.bbox_min[1], cluster.bbox_min[2]],
                            [cluster.bbox_max[0], cluster.bbox_max[1], cluster.bbox_min[2]],
                            [cluster.bbox_min[0], cluster.bbox_max[1], cluster.bbox_min[2]],
                            [cluster.bbox_min[0], cluster.bbox_min[1], cluster.bbox_max[2]],
                            [cluster.bbox_max[0], cluster.bbox_min[1], cluster.bbox_max[2]],
                            [cluster.bbox_max[0], cluster.bbox_max[1], cluster.bbox_max[2]],
                            [cluster.bbox_min[0], cluster.bbox_max[1], cluster.bbox_max[2]],
                        ])
                        
                        uv = project_3d_to_equirectangular(corners_3d, cam, img.width, img.height)
                        
                        # Check if any valid
                        if not np.isnan(uv).all():
                            # Valid
                            valid_uv = uv[~np.isnan(uv).any(axis=1)]
                            if len(valid_uv) > 0:
                                u_min, v_min = valid_uv.min(axis=0)
                                u_max, v_max = valid_uv.max(axis=0)
                                
                                # Check if bbox is reasonably sized (not wrapping around edge too wildly)
                                # For simplicity, if it's huge, ignore or clamp?
                                # Wrapping handling in equirectangular is tricky. 
                                # For now, take simple bounding box.
                                
                                best_image = img.filepath.name
                                best_bbox = [float(u_min), float(v_min), float(u_max), float(v_max)]
                                min_dist_sq = dist_sq

                if best_image:
                    features_to_insert.append(FeatureCreate(
                        feature_class=feature_class,
                        label=feature_class,
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                        confidence=0.85, 
                        source_image=best_image,
                        source_lidar="3D_LIDAR",
                        attributes={**attrs, "bbox": str(best_bbox)},
                        ocr_text="",
                        ocr_text_ar="",
                        ocr_text_en="",
                    ))
                else:
                     # Fallback if no projection found (rare)
                     features_to_insert.append(FeatureCreate(
                        feature_class=feature_class,
                        label=feature_class,
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                        confidence=0.85,
                        source_image="",
                        source_lidar="3D_LIDAR",
                        attributes=attrs,
                        ocr_text="",
                        ocr_text_ar="",
                        ocr_text_en="",
                    ))

        if features_to_insert:
            db.insert_features_bulk(features_to_insert)

        progress.update(task, description=f"[green]Done! {len(features_to_insert)} features stored.[/green]")

    console.print(f"\n[green]✓ Pipeline complete. {len(features_to_insert)} features in database.[/green]")


@app.command()
def export(
    format: str = typer.Option("geojson", "--format", "-f", help="Export format: geojson, shapefile, csv"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Export detected features to GeoJSON, Shapefile, or CSV."""
    _setup_logging(verbose)
    from geopro.config import load_config
    from geopro.storage.database import FeatureDatabase
    from geopro.export.exporter import export_features

    cfg = load_config(config)
    db = FeatureDatabase(_PROJECT_ROOT / cfg.paths.output_db)
    features = db.get_all_features()

    if not features:
        console.print("[yellow]No features to export.[/yellow]")
        raise typer.Exit(0)

    if output is None:
        export_dir = _PROJECT_ROOT / cfg.paths.export_dir
        output = str(export_dir / f"features.{format}")

    result_path = export_features(features, output, fmt=format)
    console.print(f"\n[green]✓ Exported {len(features)} features to {result_path}[/green]")


@app.command()
def serve(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Launch the interactive review UI on localhost."""
    _setup_logging(verbose)
    from geopro.config import load_config
    from geopro.ui.app import run_server

    cfg = load_config(config)
    console.print(f"\n[bold cyan]🌐 Starting GeoPro UI at http://{cfg.ui.host}:{cfg.ui.port}[/bold cyan]\n")
    run_server(cfg)


@app.command()
def run(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Execute the full pipeline: ingest → detect → export."""
    ingest(config=config, verbose=verbose)
    detect(config=config, verbose=verbose)
    export(format="geojson", output=None, config=config, verbose=verbose)


if __name__ == "__main__":
    app()
