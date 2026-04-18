# src/miniflow/__init__.py

from miniflow.tracker import ExperimentTracker
from miniflow.db import init_db

# Auto-initialize the schema at the default path for standard users
init_db()

# Explicitly define the public API
__all__ = ["ExperimentTracker"]