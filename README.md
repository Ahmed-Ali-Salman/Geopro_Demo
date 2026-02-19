# GeoPro Demo

**Computer Vision & 3D Visual Perception Pipeline** for automated roadside asset detection from Mobile Mapping System (MMS) data.

## Features

- **Multi-format LiDAR & NuScenes ingestion** — Support for `.las`, `.laz`, `.ply`, `.pcd` and NuScenes v1.0 schema
- **YOLO26 object detection** — Light poles, trees, manholes, traffic signs, guardrails, and more
- **Offline OCR** — Arabic + English via PaddleOCR with RTL post-processing
- **3D point cloud processing** — Ground removal, DBSCAN clustering, geometric attribute extraction
- **Sensor fusion** — 2D↔3D matching with equirectangular + pinhole projection and 3D-to-2D projection for unmatched clusters
- **Hybrid Storage** — SQLite-backed feature database + **DuckDB** for high-performance spatial analytics
- **Interactive review UI** — Offline Leaflet map viewer with editable attributes
- **Multi-format export** — GeoJSON, Shapefile, CSV

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Place your data
#    This project uses the NuScenes v1.0-mini dataset.
#    Download it from: https://www.nuscenes.org/download
#    
#    Extract to data/ so it looks like:
#    data/v1.0-mini/
#      ├── maps/
#      ├── samples/
#      ├── sweeps/
#      └── v1.0-mini/

# 3. Run the full pipeline
geopro run --config config/default.yaml

# 4. Launch the review UI
geopro serve
# → Open http://localhost:8000
```

## Running with Docker (Recommended)

GeoPro is fully containerized and available on [Docker Hub](https://hub.docker.com/r/salmaniv/geopro-demo). This is the easiest way to run the pipeline without managing Python environments or system dependencies (GDAL, Open3D, etc.).

### 1. Pull the Official Image
```bash
docker pull salmaniv/geopro-demo:latest
```

### 2. Run the Full Pipeline
To process the NuScenes dataset, mount your local `data` directory to the container. The following command runs the end-to-end detection and fusion pipeline:

**Windows (PowerShell):**
```powershell
docker run --rm -it `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/output:/app/output" `
  salmaniv/geopro-demo:latest geopro run
```

**Linux / macOS:**
```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  salmaniv/geopro-demo:latest geopro run
```

### 3. Review Results (Interactive UI)
Launch the web-based inspector to review the detected assets:

```bash
docker run -p 8000:8000 `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/output:/app/output" `
  salmaniv/geopro-demo:latest geopro serve
```
*Naviagate to: [http://localhost:8000](http://localhost:8000)*

| Command | Description |
|---------|-------------|
| `geopro ingest` | Validate & load input data |
| `geopro detect` | Run detection + OCR pipeline |
| `geopro export` | Export to GeoJSON / Shapefile / CSV |
| `geopro serve` | Launch the interactive review UI |
| `geopro run` | Execute full pipeline end-to-end |

## Project Structure

```
├── config/              # Pipeline configuration (YAML)
├── models/              # Pre-trained weights (gitignored)
├── data/                # Input data (gitignored)
├── src/geopro/
│   ├── cli.py           # CLI entry point
│   ├── config.py        # Pydantic settings loader
│   ├── ingest/          # LiDAR, image, CSV loaders
│   ├── detection/       # YOLOv8 detector + post-processing
│   ├── pointcloud/      # 3D preprocessing + segmentation
│   ├── ocr/             # PaddleOCR engine + Arabic support
│   ├── fusion/          # 2D↔3D projection + georeferencing
│   ├── storage/         # SQLite/GeoPackage CRUD
│   ├── export/          # GeoJSON, Shapefile, CSV export
│   └── ui/              # Flask web UI + Leaflet map
├── pyproject.toml
└── requirements.txt
```

## Configuration

Edit `config/default.yaml` to adjust:
- Input/output paths
- Detection confidence thresholds
- Voxel downsampling resolution
- OCR engine and languages
- UI host and port

## License

Proprietary
