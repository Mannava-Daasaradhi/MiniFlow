import pickle, json
from pathlib import Path
from typing import Any, Dict, List
from miniflow.db import DBUtils
from miniflow import utils
from miniflow.db import DEFAULT_DB_PATH

class ModelRegistry:
    def __init__(self, storage_dir: str = None, db_path: str = DEFAULT_DB_PATH):
        """
        storage_dir: where model artifacts (pickle files) are saved
        Defaults to ~/.miniflow/models/
        """
        self.storage_dir = Path(storage_dir) if storage_dir else Path.home() / ".miniflow" / "models"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # db_path is passed to db functions for registry operations
        self.db_path = db_path



    def save(self, name: str, model_obj: Any, metadata: Dict[str, Any] = None) -> str:
        """
        Saves model_obj with pickle to storage_dir/{name}_v{version}.pkl
        Auto-increments version and stores metadata as JSON in models table.
        """
        conn = DBUtils.get_connection(self.db_path)
        try:
            # 1. Auto-increment the version
            # If the model doesn't exist yet, max_v will be None
            row = conn.execute("SELECT MAX(version) as max_v FROM models WHERE name = ?", (name,)).fetchone()
            version = (row["max_v"] or 0) + 1
            
            # 2. Construct the exact IDs and filenames
            model_id = f"{name}_v{version}"
            file_path = self.storage_dir / f"{model_id}.pkl"
            
            # 3. Save the physical file using pickle
            with open(file_path, "wb") as f:
                pickle.dump(model_obj, f)
                
            # 4. Serialize the dictionary to a JSON string
            meta_json = json.dumps(metadata or {})
            created_at = utils.now_iso()
            
            # 5. Insert the database record
            conn.execute(
                """
                INSERT INTO models (model_id, name, version, path, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (model_id, name, version, str(file_path), meta_json, created_at)
            )
            conn.commit()
            
            return model_id
        finally:
            conn.close()

    def load(self, name: str, version: int | None = None) -> object:
        """
        Loads model from pickle file.
        If version=None, loads latest version.
        Raises FileNotFoundError if name+version not found.
        """
        conn = DBUtils.get_connection(self.db_path)
        try:
            # Route the query based on whether a specific version was requested
            if version is None:
                query = "SELECT path FROM models WHERE name = ? ORDER BY version DESC LIMIT 1"
                params = (name,)
            else:
                query = "SELECT path FROM models WHERE name = ? AND version = ?"
                params = (name, version)
                
            row = conn.execute(query, params).fetchone()
            
            if not row:
                raise FileNotFoundError(f"Model '{name}' (version {version or 'latest'}) not found in registry.")
                
            file_path = Path(row["path"])
            
            # Future-proofing: DB and file system could get out of sync
            if not file_path.exists():
                raise FileNotFoundError(f"Database record exists, but physical file is missing at {file_path}")
                
            with open(file_path, "rb") as f:
                return pickle.load(f)
        finally:
            conn.close()

    def list_models(self, name: str = None) -> list[dict]:
        """
        Returns list of model dicts: {model_id, name, version, path, metadata, created_at}
        If name given, filter to that model family.
        """
        conn = DBUtils.get_connection(self.db_path)
        try:
            if name:
                query = "SELECT * FROM models WHERE name = ? ORDER BY version DESC"
                params = (name,)
            else:
                query = "SELECT * FROM models ORDER BY name ASC, version DESC"
                params = ()
                
            rows = conn.execute(query, params).fetchall()
            
            results = []
            for r in rows:
                model_dict = dict(r)
                # Parse the JSON string back into a Python dictionary
                model_dict["metadata"] = json.loads(model_dict["metadata"])
                results.append(model_dict)
                
            return results
        finally:
            conn.close()

    def compare(self, model_ids: list[str]) -> None:
        """
        Prints a human-readable side-by-side table of metadata for given model_ids.
        Use only print() — no external dependencies.
        """
        if not model_ids:
            print("No model IDs provided.")
            return

        conn = DBUtils.get_connection(self.db_path)
        try:
            # Query metadata for all requested models
            placeholders = ",".join(["?"] * len(model_ids))
            query = f"SELECT model_id, metadata FROM models WHERE model_id IN ({placeholders})"
            rows = conn.execute(query, model_ids).fetchall()

            if not rows:
                print("No matching models found in the registry.")
                return

            # Map model_id -> parsed metadata dict
            models_meta = {row["model_id"]: json.loads(row["metadata"]) for row in rows}

            # Find all unique metadata keys across all models (sorted alphabetically)
            all_keys = set()
            for meta in models_meta.values():
                all_keys.update(meta.keys())
            all_keys = sorted(list(all_keys))

            if not all_keys:
                print("No metadata found for the specified models.")
                return

            # --- Dynamic Width Calculations ---
            # Key column width: max of "key", "-----------", or the longest metadata key
            key_width = max([len(k) for k in all_keys] + [11, len("key")])

            # Model column widths: max of model_id, "-------------", or its longest metadata value
            col_widths = {}
            for m_id in model_ids:
                meta = models_meta.get(m_id, {})
                val_lengths = [len(str(meta.get(k, "N/A"))) for k in all_keys]
                col_widths[m_id] = max(val_lengths + [13, len(m_id)])

            # --- Printing the Table ---
            # 1. Header
            header = "key".ljust(key_width) + "  "
            for m_id in model_ids:
                header += m_id.ljust(col_widths[m_id]) + "  "
            print(header)

            # 2. Separator
            separator = ("-" * key_width) + "  "
            for m_id in model_ids:
                separator += ("-" * col_widths[m_id]) + "  "
            print(separator)

            # 3. Data Rows
            for key in all_keys:
                row_str = key.ljust(key_width) + "  "
                for m_id in model_ids:
                    # Default to "N/A" if a model is missing a specific metadata key
                    val = str(models_meta.get(m_id, {}).get(key, "N/A"))
                    row_str += val.ljust(col_widths[m_id]) + "  "
                print(row_str)

        finally:
            conn.close()

    def delete(self, name: str, version: int) -> None:
        """
        Removes pkl file from disk AND deletes row from models table.
        Raises ValueError if model is referenced by any run.
        """
        model_id = f"{name}_v{version}"
        conn = DBUtils.get_connection(self.db_path)
        try:
            # 1. Retrieve the file path before we delete the record
            row = conn.execute("SELECT path FROM models WHERE name = ? AND version = ?", (name, version)).fetchone()
            if not row:
                return  # Idempotent: if it doesn't exist, do nothing and return safely
                
            file_path = Path(row["path"])
            
            # 2. Future-proofing: Check if this model_id is referenced in any run's params
            # We assume users log models like: tracker.log_params({"model": "resnet50_v1"})
            ref = conn.execute("SELECT run_id FROM params WHERE value = ? LIMIT 1", (model_id,)).fetchone()
            if ref:
                raise ValueError(
                    f"Cannot delete model '{model_id}' because it is referenced "
                    f"by run_id '{ref['run_id']}'. Delete the run first."
                )
                
            # 3. Delete from DB first to maintain state integrity
            conn.execute("DELETE FROM models WHERE name = ? AND version = ?", (name, version))
            conn.commit()
            
            # 4. Delete the physical file from disk
            if file_path.exists():
                file_path.unlink()
                
        finally:
            conn.close()