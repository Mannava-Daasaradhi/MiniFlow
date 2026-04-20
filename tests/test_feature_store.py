# tests/test_feature_store.py

import pytest
from miniflow.feature_store import FeatureStore

def test_define_and_get(tmp_db):
    fs = FeatureStore(db_path=tmp_db)
    
    # Define the schema
    fs.define(name="user_age", dtype="int", version=1, description="Age in years")
    
    # Set the value
    fs.set("user_age", entity_id="user_123", value=27, version=1)
    
    # Retrieve and assert
    val = fs.get("user_age", entity_id="user_123", version=1)
    assert val == 27
    assert type(val) is int

def test_dtype_validation(tmp_db):
    fs = FeatureStore(db_path=tmp_db)
    fs.define("user_age", dtype="int", version=1)
    
    # Attempting to insert a string into an "int" feature should raise a TypeError
    with pytest.raises(TypeError, match="must be of type int"):
        fs.set("user_age", entity_id="user_123", value="twenty-seven")
        
    # Attempting to insert a float should also fail
    with pytest.raises(TypeError, match="must be of type int"):
        fs.set("user_age", entity_id="user_123", value=27.5)

def test_upsert_behavior(tmp_db):
    fs = FeatureStore(db_path=tmp_db)
    fs.define("lifetime_value", dtype="float")
    
    # First insert
    fs.set("lifetime_value", "user_1", 100.50)
    assert fs.get("lifetime_value", "user_1") == 100.50
    
    # Upsert (update existing entity)
    fs.set("lifetime_value", "user_1", 250.75)
    
    # Retrieve and assert it was overwritten, not duplicated
    assert fs.get("lifetime_value", "user_1") == 250.75

def test_undefined_feature_raises(tmp_db):
    fs = FeatureStore(db_path=tmp_db)
    
    # Trying to set a value for a feature that hasn't been defined with fs.define()
    with pytest.raises(KeyError, match="is not defined"):
        fs.set("ghost_feature", entity_id="user_1", value=100)

def test_get_many(tmp_db):
    fs = FeatureStore(db_path=tmp_db)
    
    # Define multiple features
    fs.define("age", dtype="int")
    fs.define("income", dtype="float")
    
    # Set values for a single entity
    fs.set("age", "user_456", 34)
    fs.set("income", "user_456", 120000.0)
    
    # Request both existing features plus one that hasn't been set for this user
    feature_specs = [("age", 1), ("income", 1), ("missing_feature", 1)]
    results = fs.get_many(feature_specs, entity_id="user_456")
    
    # Assert dictionary structure and values
    assert results["age"] == 34
    assert results["income"] == 120000.0
    assert results["missing_feature"] is None

def test_get_missing_entity_raises(tmp_db):
    fs = FeatureStore(db_path=tmp_db)
    fs.define("user_age", dtype="int")
    
    # The feature is defined, but this specific user has no recorded value
    with pytest.raises(KeyError, match="not found for entity"):
        fs.get("user_age", entity_id="ghost_user")