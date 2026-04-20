# tests/test_tracker.py

import pytest
import math
from miniflow.tracker import ExperimentTracker
from miniflow.db import DBUtils

def test_new_run_creates_db_entry(tmp_db):
    t = ExperimentTracker(name="test_exp", db_path=tmp_db)
    
    # Query DB directly to verify the raw insert worked
    conn = DBUtils.get_connection(tmp_db)
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (t.run_id,)).fetchone()
    conn.close()
    
    assert row is not None
    assert row["name"] == "test_exp"
    assert row["status"] == "running"
    assert "git_hash" in row.keys()  # Verifies column exists, allowing None
    assert row["timestamp"] is not None

def test_log_params_stores_correctly(tmp_db):
    t = ExperimentTracker("test_exp", db_path=tmp_db)
    t.log_params({"lr": 0.01, "epochs": 10, "model": "resnet"})
    
    runs = t.get_runs()
    assert len(runs) == 1
    
    params = runs[0]["params"]
    # Verify everything was cast to a string as intended
    assert params["lr"] == "0.01"
    assert params["epochs"] == "10"
    assert params["model"] == "resnet"

def test_log_metric_time_series(tmp_db):
    t = ExperimentTracker("test_exp", db_path=tmp_db)
    
    # Log over multiple steps
    t.log_metric("loss", 0.50, step=0)
    t.log_metric("loss", 0.35, step=1)
    t.log_metric("loss", 0.20, step=2)
    
    runs = t.get_runs()
    assert len(runs) == 1
    
    # Verify it retrieves as a time-series list ordered by step
    assert runs[0]["metrics"]["loss"] == [0.50, 0.35, 0.20]

def test_duplicate_param_raises(tmp_db):
    t = ExperimentTracker("test_exp", db_path=tmp_db)
    t.log_params({"lr": 0.01})
    
    with pytest.raises(ValueError, match="already been logged"):
        t.log_params({"lr": 0.02})

def test_nan_metric_raises(tmp_db):
    t = ExperimentTracker("test_exp", db_path=tmp_db)
    
    with pytest.raises(ValueError, match="must be a finite number"):
        t.log_metric("loss", float("nan"))
        
    with pytest.raises(ValueError, match="must be a finite number"):
        t.log_metric("loss", float("inf"))

def test_compare_runs(tmp_db):
    # Setup Run 1
    t1 = ExperimentTracker("run1", db_path=tmp_db)
    t1.log_params({"batch_size": 32})
    t1.log_metric("accuracy", 0.85, step=1)
    
    # Setup Run 2
    t2 = ExperimentTracker("run2", db_path=tmp_db)
    t2.log_params({"batch_size": 64})
    t2.log_metric("accuracy", 0.92, step=1)
    
    # Compare
    comp = t1.compare([t1.run_id, t2.run_id])
    
    # Verify structure and data
    assert t1.run_id in comp["run_ids"]
    assert t2.run_id in comp["run_ids"]
    assert comp["params"]["batch_size"][t1.run_id] == "32"
    assert comp["params"]["batch_size"][t2.run_id] == "64"
    assert comp["metrics"]["accuracy"][t1.run_id] == [0.85]
    assert comp["metrics"]["accuracy"][t2.run_id] == [0.92]

def test_get_best_run_min(tmp_db):
    # Setup 3 runs with different final losses
    t1 = ExperimentTracker("exp", db_path=tmp_db)
    t1.log_metric("loss", 0.8, step=1)
    
    t2 = ExperimentTracker("exp", db_path=tmp_db)
    t2.log_metric("loss", 0.2, step=1)  # Best min loss
    
    t3 = ExperimentTracker("exp", db_path=tmp_db)
    t3.log_metric("loss", 0.5, step=1)
    
    best_run = t1.get_best_run(metric="loss", mode="min")
    
    # t2 should be the winner
    assert best_run["run_id"] == t2.run_id
    assert best_run["metrics"]["loss"][-1] == 0.2