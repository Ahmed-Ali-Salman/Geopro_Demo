"""DuckDB-based analytics backend for high-performance querying.

Provides analytical capabilities over the feature database using DuckDB's
spatial extension and columnar engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBAnalytics:
    """Analytics engine using DuckDB to query the Feature GeoPackage/SQLite."""

    def __init__(self, db_path: str | Path):
        """Initialize the DuckDB connection.

        Parameters
        ----------
        db_path : Path to the SQLite/GeoPackage file.
        """
        self.db_path = str(Path(db_path).resolve())
        self.conn = duckdb.connect(database=":memory:")
        self._init_extensions()
        self._attach_db()

    def _init_extensions(self) -> None:
        """Load required DuckDB extensions."""
        try:
            self.conn.install_extension("spatial")
            self.conn.load_extension("spatial")
            logger.info("DuckDB spatial extension loaded.")
        except Exception as e:
            logger.warning("Could not load DuckDB spatial extension: %s", e)

    def _attach_db(self) -> None:
        """Attach the SQLite database."""
        try:
            self.conn.execute(f"INSTALL sqlite;")
            self.conn.execute(f"LOAD sqlite;")
            self.conn.execute(
                f"ATTACH '{self.db_path}' AS features_db (TYPE SQLITE);"
            )
            logger.info("Attached SQLite DB at %s", self.db_path)
        except Exception as e:
            logger.error("Failed to attach SQLite DB: %s", e)
            raise

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a raw SQL query and return a Pandas DataFrame."""
        return self.conn.sql(sql).df()

    def summary_all_classes(self) -> pd.DataFrame:
        """Get a summary of count and average confidence per class."""
        query = """
            SELECT 
                feature_class,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                AVG(altitude) as avg_altitude
            FROM features_db.features
            GROUP BY feature_class
            ORDER BY count DESC
        """
        return self.query(query)

    def spatial_density(self, grid_size: float = 0.001) -> pd.DataFrame:
        """Compute feature density over a spatial grid."""
        # Simple grid aggregation
        query = f"""
            SELECT
                FLOOR(longitude / {grid_size}) * {grid_size} as grid_lon,
                FLOOR(latitude / {grid_size}) * {grid_size} as grid_lat,
                COUNT(*) as feature_count,
                mode(feature_class) as dominant_class
            FROM features_db.features
            GROUP BY grid_lon, grid_lat
        """
        return self.query(query)

    def text_search(self, keyword: str) -> pd.DataFrame:
        """Search OCR text for keywords."""
        query = f"""
            SELECT *
            FROM features_db.features
            WHERE 
                ocr_text ILIKE '%{keyword}%' OR 
                ocr_text_ar LIKE '%{keyword}%'
        """
        return self.query(query)

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
