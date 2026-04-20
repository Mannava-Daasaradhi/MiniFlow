# src/miniflow/__init__.py

from miniflow.tracker import ExperimentTracker
from miniflow.db import DBUtils
from miniflow.feature_store import FeatureStore
from miniflow.registry import ModelRegistry

# Auto-initialize the schema at the default path for standard users
DBUtils.init_db()

# Explicitly define the public API
__all__ = ["ExperimentTracker", "ModelRegistry", "FeatureStore"]