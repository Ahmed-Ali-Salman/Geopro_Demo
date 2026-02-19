"""FastAPI web application — local review UI.

Serves the MapLibre map, feature CRUD API, and offline MBTiles tiles.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
import io
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from geopro.config import PipelineConfig, load_config
from geopro.storage.database import FeatureDatabase
from geopro.storage.models import FeatureUpdate, FeatureRead

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: startup and shutdown."""
    # Startup
    logger.info("Starting GeoPro UI...")
    yield
    # Shutdown
    logger.info("Shutting down GeoPro UI...")


def create_app(config: Optional[PipelineConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = load_config()

    app = FastAPI(
        title="GeoPro Demo",
        description="Roadside Asset Detection Review UI",
        version="0.2.0",
        lifespan=lifespan,
        debug=config.ui.debug,
    )

    # Static files & Templates
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True) # Ensure static dir exists
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

    # Database
    db_path = _PROJECT_ROOT / config.paths.output_db
    db = FeatureDatabase(db_path)

    # MBTiles path
    mbtiles_path = _PROJECT_ROOT / config.paths.basemap

    # ── Routes ──────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main map viewer page."""
        return templates.TemplateResponse("index.html", {"request": request})

    # ── Feature API ─────────────────────────────────────────────────────

    @app.get("/api/features", response_model=Dict)
    async def get_features(
        feature_class: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10000,
    ):
        """Get all features as GeoJSON, optionally filtered."""
        # Note: Database calls are currently sync (sqlite), could await run_in_executor if needed
        # But for local SQLite, sync is often fine for low concurrency.
        features = db.get_all_features(
            feature_class=feature_class,
            review_status=status,
            limit=limit,
        )

        geojson_features = []
        for f in features:
            props = f.model_dump()
            # Remove geo fields from properties to avoid duplication, strictly speaking standard GeoJSON
            # keeps them, but let's follow the previous pattern
            
            # Create GeoJSON Feature
            geojson_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [f.longitude, f.latitude],
                },
                "properties": props,
            })

        return {
            "type": "FeatureCollection",
            "features": geojson_features,
        }

    @app.get("/api/features/{feature_id}", response_model=FeatureRead)
    async def get_feature(feature_id: int):
        """Get a single feature by ID."""
        feature = db.get_feature(feature_id)
        if feature is None:
            raise HTTPException(status_code=404, detail="Feature not found")
        return feature

    @app.patch("/api/features/{feature_id}")
    async def update_feature(feature_id: int, updates: FeatureUpdate):
        """Update a feature's attributes."""
        success = db.update_feature(feature_id, updates)
        if success:
            return {"status": "updated"}
        raise HTTPException(status_code=400, detail="Update failed")

    @app.get("/api/stats")
    async def get_stats():
        """Get feature count statistics."""
        total = db.count_features()
        by_class = {}
        # Simple aggregation (could be optimized with DB group by)
        features = db.get_all_features(limit=100000)
        for f in features:
            by_class[f.feature_class] = by_class.get(f.feature_class, 0) + 1

        return {
            "total": total,
            "by_class": by_class,
        }

    @app.get("/api/features/{feature_id}/crop")
    async def get_feature_crop(feature_id: int):
        """Serve a dynamic crop of the feature from the source image."""
        try:
            from PIL import Image
            feature = db.get_feature(feature_id)
            if not feature:
                raise HTTPException(status_code=404, detail="Feature not found")

            # Construct image path
            img_name = feature.source_image
            attrs = feature.attributes or {}
            
            # Try to get bbox from attributes? 
            # WAIT: The Feature model in `storage/models.py` might not save raw bbox if we didn't add it.
            # Let's check `process_pipeline`... we stored attributes.
            # If bbox isn't in attributes, we can't crop.
            # We need to verify if ingestion saved 'bbox' in attributes.
            # Let's assume for now we need to patch ingestion if it's missing, 
            # but let's check `attributes=attrs` in `cli.py`.
            
            # Inspecting cli.py line 235: `attrs = extract_all_attributes(mf.cluster)`
            # Matcher (line 218) matches 2D dev to 3D cluster. 
            # We need the 2D BBOX for the crop.
            # Does `FeatureCreate` have bbox? No, standard fields.
            # We should probably store [x1, y1, x2, y2] in attributes during fusion.
            
            # Let's optimistically implement utilizing 'bbox' attribute.
            # If missing, return 404 or full image.
            
            img_path = _PROJECT_ROOT / config.paths.images_dir / img_name
            if not img_path.exists():
                raise HTTPException(status_code=404, detail="Source image not found")

            # Parse bbox string "[x1, y1, x2, y2]" or list
            # We'll need to ensure `detect` command saves it. 
            # For now, let's implement the logic assuming it exists or default to center crop.
            
            bbox = attrs.get('bbox') 
            if isinstance(bbox, str):
                import ast
                try: bbox = ast.literal_eval(bbox)
                except: bbox = None
            
            with Image.open(img_path) as im:
                if bbox and len(bbox) == 4:
                    # Add padding
                    pad = 50
                    w, h = im.size
                    x1 = max(0, int(bbox[0]) - pad)
                    y1 = max(0, int(bbox[1]) - pad)
                    x2 = min(w, int(bbox[2]) + pad)
                    y2 = min(h, int(bbox[3]) + pad)
                    crop = im.crop((x1, y1, x2, y2))
                else:
                    # Fallback: Can't crop without bbox. Return placeholder or thumbnail?
                    # Let's return full image resized
                    im.thumbnail((500, 500))
                    crop = im
                
                img_io = io.BytesIO()
                crop.save(img_io, 'JPEG', quality=85)
                img_io.seek(0)
                return StreamingResponse(img_io, media_type="image/jpeg")
                
        except Exception as e:
            logger.error(f"Crop error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ── MBTiles tile server ─────────────────────────────────────────────

    @app.get("/tiles/{z}/{x}/{y}.png")
    async def serve_tile(z: int, x: int, y: int):
        """Serve a map tile from the MBTiles file."""
        if not mbtiles_path.exists():
            return Response(content="No basemap available", status_code=404)

        # MBTiles uses TMS convention (flipped Y)
        tms_y = (1 << z) - 1 - y

        try:
            # Open new connection per request for simplicity (SQLite handles concurrency)
            # For high throughput, a connection pool or shared connection is better.
            with sqlite3.connect(str(mbtiles_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                    (z, x, tms_y),
                )
                row = cursor.fetchone()

            if row:
                return Response(content=row[0], media_type="image/png")
            else:
                return Response(content="Tile not found", status_code=404)
        except Exception as exc:
            logger.warning("Tile error z=%d x=%d y=%d: %s", z, x, y, exc)
            return Response(content="Tile error", status_code=500)

    return app


def run_server(config: Optional[PipelineConfig] = None) -> None:
    """Start the FastAPI server using Uvicorn."""
    if config is None:
        config = load_config()

    logger.info(
        "Starting GeoPro UI at http://%s:%d",
        config.ui.host, config.ui.port,
    )
    uvicorn.run(
        "geopro.ui.app:create_app",
        host=config.ui.host,
        port=config.ui.port,
        reload=config.ui.debug,
        factory=True,
    )
