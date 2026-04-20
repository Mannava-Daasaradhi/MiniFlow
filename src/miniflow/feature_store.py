import json
import sqlite3
from miniflow.db import DBUtils, DEFAULT_DB_PATH
from miniflow import utils
from typing import List, Tuple, Dict, Any

# Module-level constant mapping string definitions to actual Python types
DTYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict
}

class FeatureStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Standard init, same db_path pattern as above."""
        self.db_path = db_path

    def define(self, name: str, dtype: str, version: int = 1, description: str = "") -> None:
        """
        Registers a feature schema.
        dtype must be one of: "int", "float", "str", "list", "dict"
        Raises ValueError on duplicate (name, version).
        """
        if dtype not in DTYPE_MAP:
            raise ValueError(f"dtype must be one of {list(DTYPE_MAP.keys())}")
            
        timestamp = utils.now_iso()
        conn = DBUtils.get_connection(self.db_path)
        
        try:
            conn.execute(
                """
                INSERT INTO feature_definitions (name, version, dtype, description, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, version, dtype, description, timestamp)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # Catching the DB-level constraint violation (Option B)
            conn.rollback()
            raise ValueError(f"Feature '{name}' (version {version}) is already defined.")
        finally:
            conn.close()

    def set(self, name: str, entity_id: str, value: object, version: int = 1) -> None:
        """
        Stores feature value for an entity.
        Validates that (name, version) is defined. Raises KeyError if not.
        Validates that type(value) matches dtype. Raises TypeError if not.
        Value is stored as JSON string (json.dumps).
        If (name, version, entity_id) already exists: UPDATE. Else: INSERT.
        """
        timestamp = utils.now_iso()
        conn = DBUtils.get_connection(self.db_path)
        
        try:
            # 1. Validate the feature exists and fetch its expected dtype
            row = conn.execute(
                "SELECT dtype FROM feature_definitions WHERE name = ? AND version = ?",
                (name, version)
            ).fetchone()
            
            if not row:
                raise KeyError(f"Feature '{name}' (version {version}) is not defined.")
                
            defined_dtype = row["dtype"]
            if type(value) is bool and defined_dtype != "bool": # Assuming we didn't add "bool" to DTYPE_MAP
                raise TypeError(f"Value for '{name}' must be {defined_dtype}, got bool.")
            
            # 2. Validate the value matches the defined dtype
            if not isinstance(value, DTYPE_MAP[defined_dtype]):
                raise TypeError(
                    f"Value for '{name}' must be of type {defined_dtype}, "
                    f"got {type(value).__name__} instead."
                )
                
            # 3. Serialize to JSON
            json_val = json.dumps(value)
            
            # 4. Upsert the value 
            # Note: This uses standard SQLite 3.24.0+ ON CONFLICT syntax
            conn.execute(
                """
                INSERT INTO feature_values (name, version, entity_id, value, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name, version, entity_id) 
                DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (name, version, entity_id, json_val, timestamp)
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, name: str, entity_id: str, version: int = 1) -> object:
        """
        Returns feature value, deserialized from JSON.
        Raises KeyError if entity_id not found for this feature.
        """
        conn = DBUtils.get_connection(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT value FROM feature_values 
                WHERE name = ? AND version = ? AND entity_id = ?
                """,
                (name, version, entity_id)
            ).fetchone()
            
            if not row:
                raise KeyError(
                    f"Feature '{name}' (version {version}) not found for entity '{entity_id}'."
                )
                
            return json.loads(row["value"])
        finally:
            conn.close()

    def get_many(self, feature_specs: list[tuple], entity_id: str) -> dict:
        """
        feature_specs: list of (name, version) tuples
        Returns {name: value, ...} dict for the entity.
        Missing features returned as None (not raised).
        Example:
            fs.get_many([("age", 1), ("income", 1)], "user_123")
            -> {"age": 27, "income": 95000.0}
        """
        if not feature_specs:
            return {}

        # Pre-fill results with None so missing features are handled automatically
        results = {name: None for name, _ in feature_specs}
        
        # Build the dynamic query for (name, version) pairs
        conditions = " OR ".join(["(name = ? AND version = ?)"] * len(feature_specs))
        
        # Flatten the feature_specs tuples for the parameter list
        params = [entity_id]
        for name, version in feature_specs:
            params.extend([name, version])
            
        conn = DBUtils.get_connection(self.db_path)
        try:
            # Single query fetches all requested features for the entity
            rows = conn.execute(
                f"""
                SELECT name, value 
                FROM feature_values 
                WHERE entity_id = ? AND ({conditions})
                """,
                params
            ).fetchall()
            
            for row in rows:
                results[row["name"]] = json.loads(row["value"])
                
            return results
        finally:
            conn.close()

    def list_features(self) -> list[dict]:
        """Returns all defined feature schemas."""
        conn = DBUtils.get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT name, version, dtype, description, created_at FROM feature_definitions"
            ).fetchall()
            
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def history(self, name: str, entity_id: str) -> list[dict]:
        """
        Returns all historical values for (name, entity_id) across all versions.
        Useful for debugging feature drift.
        """
        conn = DBUtils.get_connection(self.db_path)
        try:
            # Query the feature_values table across all versions for this entity
            rows = conn.execute(
                """
                SELECT version, value, updated_at 
                FROM feature_values 
                WHERE name = ? AND entity_id = ?
                ORDER BY version ASC
                """,
                (name, entity_id)
            ).fetchall()
            
            history_list = []
            for row in rows:
                history_list.append({
                    "version": row["version"], 
                    "updated_at": row["updated_at"],
                    "value": json.loads(row["value"])
                })
                
            return history_list
        finally:
            conn.close()