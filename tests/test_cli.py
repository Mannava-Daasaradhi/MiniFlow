# tests/test_cli.py

import pytest
from click.testing import CliRunner
from miniflow.cli import main
from miniflow.tracker import ExperimentTracker

@pytest.fixture
def runner(tmp_db, monkeypatch):
    """
    Creates a CliRunner and monkeypatches the default database path
    so the CLI natively reads from our isolated test database.
    """
    # Patch the constant in every namespace where it is used as a default
    monkeypatch.setattr("miniflow.db.DEFAULT_DB_PATH", tmp_db)
    monkeypatch.setattr("miniflow.cli.DEFAULT_DB_PATH", tmp_db)
    monkeypatch.setattr("miniflow.registry.DEFAULT_DB_PATH", tmp_db)
    monkeypatch.setattr("miniflow.feature_store.DEFAULT_DB_PATH", tmp_db)
    
    return CliRunner()

def test_runs_list_empty(runner):
    result = runner.invoke(main, ["runs", "list"])
    assert result.exit_code == 0
    assert "No runs found." in result.output

def test_runs_list_with_data(runner, tmp_db):
    t = ExperimentTracker("cli_test_exp", db_path=tmp_db)
    t.log_params({"model": "bert", "lr": 0.001})
    t.log_metric("loss", 0.15)
    
    result = runner.invoke(main, ["runs", "list"])
    
    assert result.exit_code == 0
    assert "cli_test_exp" in result.output
    assert "model=bert" in result.output
    assert "loss=0.1500" in result.output

def test_models_list_empty(runner):
    result = runner.invoke(main, ["models", "list"])
    assert result.exit_code == 0
    assert "No models found." in result.output

def test_features_list_empty(runner):
    result = runner.invoke(main, ["features", "list"])
    assert result.exit_code == 0
    assert "No features defined." in result.output

def test_runs_best_missing_metric(runner, tmp_db):
    t = ExperimentTracker("dummy_exp", db_path=tmp_db)
    t.log_metric("accuracy", 0.99)
    
    result = runner.invoke(main, ["runs", "best", "--metric", "ghost_metric"])
    
    assert result.exit_code == 0 
    assert "Metric 'ghost_metric' not found" in result.output