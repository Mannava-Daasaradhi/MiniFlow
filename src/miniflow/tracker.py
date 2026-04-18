import sqlite3
from miniflow.db import DBUtils
from miniflow import utils
from miniflow.db import DEFAULT_DB_PATH
from typing import Any, Dict, List, Union
class ExperimentTracker:
    def __init__(self, name: str, db_path: str = DEFAULT_DB_PATH):
        """
        Creates a new run immediately on instantiation.
        """
        self.name = name
        self.db_path = db_path
        self.run_id = utils.generate_run_id()
        
        git_hash = utils.get_git_hash()
        timestamp = utils.now_iso()
        
        conn = DBUtils.get_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO runs (run_id, name, git_hash, timestamp, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (self.run_id, self.name, git_hash, timestamp)
            )
            conn.commit()
        finally:
            conn.close()

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Inserts key-value pairs into params table for current run.
        All values are cast to strings.
        Raises ValueError if a key is logged twice for the same run.
        """
        conn = DBUtils.get_connection(self.db_path)
        try:
            for key, value in params.items():
                try:
                    conn.execute(
                        """
                        INSERT INTO params (run_id, key, value)
                        VALUES (?, ?, ?)
                        """,
                        (self.run_id, key, str(value))
                    )
                except sqlite3.IntegrityError:
                    # Rolling back ensures partial param dictionaries aren't saved if one fails
                    conn.rollback()
                    raise ValueError(f"Param '{key}' has already been logged for this run.")
            conn.commit()
        finally:
            conn.close()

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """
        Inserts one metric row. Multiple calls with same key = time series.
        step must be >= 0. value must be finite float (reject NaN, inf).
        """
        if step < 0:
            raise ValueError("Step must be >= 0")
        
        utils.validate_finite(value, key)
        timestamp = utils.now_iso()
        
        conn = DBUtils.get_connection(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO metrics (run_id, key, value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.run_id, key, value, step, timestamp)
            )
            conn.commit()
        finally:
            conn.close()

    def finish(self, status: str = "finished") -> None:
        """Updates runs.status to 'finished' or 'failed'."""
        if status not in ("finished", "failed"):
            raise ValueError("Status must be 'finished' or 'failed'")
            
        conn = DBUtils.get_connection(self.db_path)
        try:
            conn.execute(
                "UPDATE runs SET status = ? WHERE run_id = ?",
                (status, self.run_id)
            )
            conn.commit()
        finally:
            conn.close()

    def get_runs(self, name: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Returns list of run dicts stitched together from 3 tables.
        """
        conn = DBUtils.get_connection(self.db_path)
        try:
            # 1. Fetch the core run rows
            query = "SELECT * FROM runs"
            params = []
            if name:
                query += " WHERE name = ?"
                params.append(name)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            runs_rows = conn.execute(query, params).fetchall()
            if not runs_rows:
                return []
                
            runs_list = []
            runs_map = {}
            run_ids = []
            
            # Initialize dictionaries
            for r in runs_rows:
                run_dict = dict(r)
                run_dict["params"] = {}
                run_dict["metrics"] = {}
                runs_list.append(run_dict)
                runs_map[r["run_id"]] = run_dict
                run_ids.append(r["run_id"])
                
            placeholders = ",".join(["?"] * len(run_ids))
            
            # 2. Fetch Params in one query
            p_rows = conn.execute(
                f"SELECT run_id, key, value FROM params WHERE run_id IN ({placeholders})", 
                run_ids
            ).fetchall()
            for p in p_rows:
                runs_map[p["run_id"]]["params"][p["key"]] = p["value"]
                
            # 3. Fetch Metrics in one query (ordered by step)
            m_rows = conn.execute(
                f"SELECT run_id, key, value FROM metrics WHERE run_id IN ({placeholders}) ORDER BY step ASC", 
                run_ids
            ).fetchall()
            for m in m_rows:
                m_key = m["key"]
                r_id = m["run_id"]
                if m_key not in runs_map[r_id]["metrics"]:
                    runs_map[r_id]["metrics"][m_key] = []
                runs_map[r_id]["metrics"][m_key].append(m["value"])
                
            return runs_list
        finally:
            conn.close()

    def compare(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Returns side-by-side comparison dict by pivoting the DB rows.
        """
        if not run_ids:
            return {"run_ids": [], "params": {}, "metrics": {}}
            
        result = {
            "run_ids": run_ids,
            "params": {},
            "metrics": {}
        }
        
        conn = DBUtils.get_connection(self.db_path)
        try:
            placeholders = ",".join(["?"] * len(run_ids))
            
            # Pivot params
            p_rows = conn.execute(
                f"SELECT run_id, key, value FROM params WHERE run_id IN ({placeholders})", 
                run_ids
            ).fetchall()
            for p in p_rows:
                key, r_id, val = p["key"], p["run_id"], p["value"]
                if key not in result["params"]:
                    result["params"][key] = {}
                result["params"][key][r_id] = val
                
            # Pivot metrics
            m_rows = conn.execute(
                f"SELECT run_id, key, value FROM metrics WHERE run_id IN ({placeholders}) ORDER BY step ASC", 
                run_ids
            ).fetchall()
            for m in m_rows:
                key, r_id, val = m["key"], m["run_id"], m["value"]
                if key not in result["metrics"]:
                    result["metrics"][key] = {}
                if r_id not in result["metrics"][key]:
                    result["metrics"][key][r_id] = []
                result["metrics"][key][r_id].append(val)
                
            return result
        finally:
            conn.close()

    def get_best_run(self, metric: str, mode: str = "min") -> Dict[str, Any]:
        """
        Returns run dict for best run by final value of `metric`.
        """
        if mode not in ("min", "max"):
            raise ValueError("Mode must be 'min' or 'max'")
            
        conn = DBUtils.get_connection(self.db_path)
        try:
            # Subquery finds the max step for each run's metric, 
            # outer query gets the actual value at that step.
            query = """
                SELECT run_id, value FROM metrics m1
                WHERE key = ?
                AND step = (
                    SELECT MAX(step) FROM metrics m2
                    WHERE m2.run_id = m1.run_id AND m2.key = m1.key
                )
            """
            rows = conn.execute(query, (metric,)).fetchall()
            if not rows:
                raise ValueError(f"Metric '{metric}' not found in any run.")
                
            # Sort in Python to find the winner
            best_row = min(rows, key=lambda x: x["value"]) if mode == "min" else max(rows, key=lambda x: x["value"])
            best_run_id = best_row["run_id"]
            
            # Reconstruct the single best run dictionary
            run_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (best_run_id,)).fetchone()
            run_dict = dict(run_row)
            run_dict["params"] = {}
            run_dict["metrics"] = {}
            
            p_rows = conn.execute("SELECT key, value FROM params WHERE run_id = ?", (best_run_id,)).fetchall()
            for p in p_rows: 
                run_dict["params"][p["key"]] = p["value"]
                
            m_rows = conn.execute("SELECT key, value FROM metrics WHERE run_id = ? ORDER BY step ASC", (best_run_id,)).fetchall()
            for m in m_rows:
                if m["key"] not in run_dict["metrics"]: 
                    run_dict["metrics"][m["key"]] = []
                run_dict["metrics"][m["key"]].append(m["value"])
                
            return run_dict
        finally:
            conn.close()