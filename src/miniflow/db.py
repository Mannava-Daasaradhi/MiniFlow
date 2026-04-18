import sqlite3
from pathlib import Path
from typing import Any, Dict, List

# Module-level constant for the default path
DEFAULT_DB_PATH = Path.home() / ".miniflow" / "miniflow.db"
class DBUtils:
    @staticmethod
    def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
        """Returns a connection with row_factory = sqlite3.Row and WAL mode enabled."""
        
        # Ensure db_path is a Path object
        path_obj = Path(db_path)
        
        # Create the parent directory (e.g., ~/.miniflow) if it doesn't exist
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Now it is safe to let SQLite create or connect to the file
        conn = sqlite3.connect(str(path_obj))
        
        # Return rows as dictionaries instead of plain tuples
        conn.row_factory = sqlite3.Row 
        
        # Enable WAL mode for better concurrency/performance
        conn.execute("PRAGMA journal_mode=WAL;") 
        conn.execute("PRAGMA foreign_keys = ON;")
        
        return conn
    
    @staticmethod
    def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
        """Initializes the database schema if it doesn't already exist."""
        conn = DBUtils.get_connection(db_path)
        
        # Using executescript is a clean way to run multiple statements at once
        conn.executescript("""
            -- Runs table: one row per experiment run
            CREATE TABLE IF NOT EXISTS runs (
                run_id    TEXT PRIMARY KEY,   -- UUID4
                name      TEXT NOT NULL,      -- experiment name
                git_hash  TEXT,               -- output of git rev-parse HEAD
                timestamp TEXT NOT NULL,      -- ISO 8601 UTC
                status    TEXT DEFAULT 'running'  -- running | finished | failed
            );

            -- Params: hyperparameters logged for a run (string key-value)
            CREATE TABLE IF NOT EXISTS params (
                run_id TEXT NOT NULL,
                key    TEXT NOT NULL,
                value  TEXT NOT NULL,
                PRIMARY KEY (run_id, key),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            -- Metrics: numeric values over steps
            CREATE TABLE IF NOT EXISTS metrics (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id    TEXT NOT NULL,
                key       TEXT NOT NULL,
                value     REAL NOT NULL,
                step      INTEGER NOT NULL DEFAULT 0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            -- Model registry
            CREATE TABLE IF NOT EXISTS models (
                model_id   TEXT PRIMARY KEY,    -- "name_v{version}"
                name       TEXT NOT NULL,
                version    INTEGER NOT NULL,
                path       TEXT NOT NULL,       -- absolute path to serialized artifact
                metadata   TEXT NOT NULL,       -- JSON blob
                created_at TEXT NOT NULL,
                UNIQUE(name, version)
            );

            -- Feature store
            CREATE TABLE IF NOT EXISTS feature_definitions (
                name        TEXT NOT NULL,
                version     INTEGER NOT NULL,
                dtype       TEXT NOT NULL,
                description TEXT,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (name, version)
            );

            -- Feature values
            CREATE TABLE IF NOT EXISTS feature_values (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                version    INTEGER NOT NULL,
                entity_id  TEXT NOT NULL,
                value      TEXT NOT NULL,  -- JSON-serialized
                updated_at TEXT NOT NULL,
                UNIQUE(name, version, entity_id)
            );
        """)
        
        conn.commit()
        conn.close()