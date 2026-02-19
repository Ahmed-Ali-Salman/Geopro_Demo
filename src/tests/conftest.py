"""Shared test fixtures for the GeoPro test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_points() -> np.ndarray:
    """Generate a synthetic point cloud with a ground plane and two poles."""
    rng = np.random.default_rng(42)

    # Ground plane: Z ≈ 0, spread across XY
    n_ground = 2000
    ground = np.zeros((n_ground, 6))
    ground[:, 0] = rng.uniform(-50, 50, n_ground)    # X
    ground[:, 1] = rng.uniform(-50, 50, n_ground)    # Y
    ground[:, 2] = rng.normal(0, 0.05, n_ground)     # Z ≈ 0

    # Pole 1: centered at (10, 5), height 0–8m, narrow
    n_pole = 300
    pole1 = np.zeros((n_pole, 6))
    pole1[:, 0] = rng.normal(10, 0.1, n_pole)        # X
    pole1[:, 1] = rng.normal(5, 0.1, n_pole)         # Y
    pole1[:, 2] = rng.uniform(0, 8, n_pole)           # Z

    # Pole 2 / tree: centered at (-15, 20), height 0–6m, wider
    n_tree = 500
    tree = np.zeros((n_tree, 6))
    tree[:, 0] = rng.normal(-15, 0.8, n_tree)
    tree[:, 1] = rng.normal(20, 0.8, n_tree)
    tree[:, 2] = rng.uniform(0, 6, n_tree)

    return np.vstack([ground, pole1, tree])


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate a synthetic 2:1 equirectangular image."""
    return np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8)


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_metadata_csv(tmp_dir: Path) -> Path:
    """Create a minimal metadata CSV for testing."""
    csv_path = tmp_dir / "metadata.csv"
    csv_path.write_text(
        "image_filename,latitude,longitude,altitude,fx,fy,cx,cy,timestamp\n"
        "pano_001.jpg,24.7136,46.6753,620.0,500.0,500.0,500.0,250.0,1700000000.0\n"
        "pano_002.jpg,24.7140,46.6760,621.0,500.0,500.0,500.0,250.0,1700000001.0\n"
    )
    return csv_path
