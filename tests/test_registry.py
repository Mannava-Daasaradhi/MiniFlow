# tests/test_registry.py

import pytest
from pathlib import Path
from miniflow.registry import ModelRegistry
from miniflow.db import DBUtils

# A dummy "model" object to serialize. In reality, this would be a scikit-learn or PyTorch object,
# but a dictionary works perfectly to test the pickle/unpickle mechanism.
DUMMY_MODEL = {"layer_1": [0.1, 0.2], "layer_2": [0.5, 0.9]}

def test_save_increments_version(tmp_db, tmp_path):
    storage_dir = str(tmp_path / "models")
    registry = ModelRegistry(storage_dir=storage_dir, db_path=tmp_db)
    
    # Save first version
    id_1 = registry.save("resnet", DUMMY_MODEL)
    assert id_1 == "resnet_v1"
    
    # Save second version of the same model family
    id_2 = registry.save("resnet", DUMMY_MODEL)
    assert id_2 == "resnet_v2"
    
    # Verify files actually exist on disk
    assert Path(storage_dir, "resnet_v1.pkl").exists()
    assert Path(storage_dir, "resnet_v2.pkl").exists()

def test_load_latest(tmp_db, tmp_path):
    storage_dir = str(tmp_path / "models")
    registry = ModelRegistry(storage_dir=storage_dir, db_path=tmp_db)
    
    registry.save("mymodel", "version_1_data")
    registry.save("mymodel", "version_2_data")
    
    # Loading without a version should fetch the highest version number
    loaded = registry.load("mymodel")
    assert loaded == "version_2_data"

def test_load_specific_version(tmp_db, tmp_path):
    storage_dir = str(tmp_path / "models")
    registry = ModelRegistry(storage_dir=storage_dir, db_path=tmp_db)
    
    registry.save("mymodel", "version_1_data")
    registry.save("mymodel", "version_2_data")
    
    # Explicitly load an older version
    loaded = registry.load("mymodel", version=1)
    assert loaded == "version_1_data"

def test_compare_output(tmp_db, tmp_path, capsys):
    storage_dir = str(tmp_path / "models")
    registry = ModelRegistry(storage_dir=storage_dir, db_path=tmp_db)
    
    registry.save("clf", DUMMY_MODEL, metadata={"accuracy": 0.85, "dataset": "A"})
    registry.save("clf", DUMMY_MODEL, metadata={"accuracy": 0.92, "dataset": "B"})
    
    # Call compare, which prints to stdout
    registry.compare(["clf_v1", "clf_v2"])
    
    # capsys captures stdout so we can assert against it
    captured = capsys.readouterr()
    
    # Verify the headers and metadata keys/values were printed
    assert "clf_v1" in captured.out
    assert "clf_v2" in captured.out
    assert "accuracy" in captured.out
    assert "dataset" in captured.out
    assert "0.85" in captured.out
    assert "0.92" in captured.out

def test_delete_removes_file(tmp_db, tmp_path):
    storage_dir = str(tmp_path / "models")
    registry = ModelRegistry(storage_dir=storage_dir, db_path=tmp_db)
    
    model_id = registry.save("target_model", DUMMY_MODEL)
    file_path = Path(storage_dir) / f"{model_id}.pkl"
    
    # Confirm it exists before deletion
    assert file_path.exists()
    
    registry.delete("target_model", version=1)
    
    # 1. Verify physical file is gone
    assert not file_path.exists()
    
    # 2. Verify database record is gone
    conn = DBUtils.get_connection(tmp_db)
    row = conn.execute("SELECT * FROM models WHERE name = 'target_model'").fetchone()
    conn.close()
    assert row is None

def test_load_missing_raises(tmp_db, tmp_path):
    storage_dir = str(tmp_path / "models")
    registry = ModelRegistry(storage_dir=storage_dir, db_path=tmp_db)
    
    with pytest.raises(FileNotFoundError, match="not found in registry"):
        registry.load("ghost_model")
        
    registry.save("real_model", DUMMY_MODEL)
    
    with pytest.raises(FileNotFoundError, match="not found in registry"):
        registry.load("real_model", version=99)