# Tech Stack Upgrade

## 1. Dependencies & Config
- [x] Update `requirements.txt`
- [x] Update `pyproject.toml`
- [x] Update `config.py` (models, config sections)
- [x] Update `default.yaml`

## 2. YOLO26 Detection
- [x] Update `yolo_detector.py` docstrings/logs
- [x] Update `detection/__init__.py`
- [x] Update CLI fallback model
- [x] Download YOLO26 model and configure GPU

## 3. Surya OCR
- [x] Replace EasyOCR with Surya in `engine.py`
- [x] Update `ocr/__init__.py`

## 4. FastAPI + MapLibre GL JS
- [x] Rewrite `app.py` from Flask to FastAPI
- [x] Rewrite `index.html` from Leaflet to MapLibre GL JS
- [x] Update `ui/__init__.py`

## 5. DuckDB Analytics
- [x] Create `duckdb_backend.py`
- [x] Add protocol to `database.py` (skipped, added standalone class)
- [x] Update `storage/__init__.py`

## 6. Finalize
- [x] Update `Dockerfile`
- [x] Update CLI serve command
- [x] Verify imports
- [x] Update walkthrough

## 7. Data Acquisition & Modernization
- [x] Find sample MMS dataset (LiDAR + Spherical Images)
- [x] Download and structure data
- [x] "Modernize" data (Update timestamps to 2026, convert to LAS 1.4)

## 8. End-to-End Testing
- [x] Run Ingestion (LiDAR + Images)
- [x] Run Detection (YOLO + Suray)
- [x] Verify results in UI

## 9. QA/QC UI Enhancements
- [ ] Add `GET /api/features/{id}/crop` endpoint to `app.py`
- [ ] Update `index.html` to display feature image in Inspector Panel
- [ ] Verify RTL support for Arabic OCR editing (HTML `dir="rtl"` present)

## 10. NuScenes Integration
- [x] Create `nuscenes_loader.py` (implemented as `convert_nuscenes.py`)
- [x] Map NuScenes sensors (LIDAR_TOP, CAM_FRONT) to GeoPro format
- [x] Ingest sample scene into local database (Scene-0061 processed)

## 11. Documentation & Handoff
- [x] Create Project Presentation (`presentation.md`)
- [x] Create Technical Deep-Dive (`technical_architecture.md`)
- [x] Final Code Cleanup & README Update

## 12. Feature Expansion
- [x] Implement detection support for all 2.1 Mandatory Feature Classes
- [x] Implement attribute extraction (Height, Material, Shape, etc.)
- [x] Verify attribute storage in DuckDB (Schema uses JSON, no migration needed)
- [x] Debug OCR Pipeline (Fixed PaddleOCR output parsing for List[Dict] format)
- [x] Fix Missing Images for 3D-only Features (Project 3D clusters to 2D)

## 13. Docker Deployment
- [x] Push Docker Image to Docker Hub
- [x] Update README with Docker Run instructions
