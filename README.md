# MiniFlow

![Python](https://img.shields.io/badge/python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-82%25-yellow)

Minimal MLOps primitives built from scratch in pure Python — no MLflow, no Weights & Biases, no Neptune.

---

## What is MiniFlow?

MiniFlow is a zero-dependency MLOps library that implements the three core primitives every real ML platform is built on:

- **ExperimentTracker** — log, query, and compare ML runs backed by SQLite
- **ModelRegistry** — version, save, load, and compare trained models with metadata
- **FeatureStore** — define features with schemas, version them, and retrieve by entity

---

## Install

```bash
git clone https://github.com/Mannava-Daasaradhi/MiniFlow.git
cd MiniFlow
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -e .
```

---

## Quick Start

```python
from miniflow import ExperimentTracker, ModelRegistry, FeatureStore

# --- ExperimentTracker ---
t = ExperimentTracker("my_experiment")
t.log_params({"lr": 0.001, "epochs": 10, "model": "resnet50"})
t.log_metric("loss", 0.523, step=1)
t.log_metric("loss", 0.201, step=10)
t.log_metric("accuracy", 0.94, step=10)
runs = t.get_runs()
print(runs)  # all runs with params + metrics visible

# --- ModelRegistry ---
registry = ModelRegistry()
registry.save("resnet50", model_obj, metadata={"val_acc": 0.94, "trained_on": "ImageNet"})
loaded = registry.load("resnet50")
registry.compare(["resnet50_v1", "resnet50_v2"])  # side-by-side metadata diff

# --- FeatureStore ---
fs = FeatureStore()
fs.define("user_age", dtype="int", version=1, description="User age in years")
fs.set("user_age", entity_id="user_123", value=27, version=1)
val = fs.get("user_age", entity_id="user_123")
print(val)  # 27
```

---

## CLI

```bash
miniflow runs list [--name EXPERIMENT_NAME] [--limit 20]
miniflow runs compare <run_id_1> <run_id_2>
miniflow runs best --metric loss --mode min
miniflow models list [--name MODEL_NAME]
miniflow models compare <model_id_1> <model_id_2>
miniflow features list
```

---

## API Reference

### ExperimentTracker

| Method | Description |
|--------|-------------|
| `ExperimentTracker(name, db_path)` | Creates a new run immediately |
| `log_params(params: dict)` | Log hyperparameters as key-value pairs |
| `log_metric(key, value, step)` | Log a metric value at a given step |
| `finish(status)` | Mark run as `finished` or `failed` |
| `get_runs(name, limit)` | Return list of run dicts with params and metrics |
| `compare(run_ids)` | Side-by-side param and metric comparison |
| `get_best_run(metric, mode)` | Find best run by min or max of a metric |

### ModelRegistry

| Method | Description |
|--------|-------------|
| `ModelRegistry(storage_dir, db_path)` | Init with optional custom paths |
| `save(name, model_obj, metadata)` | Pickle model, auto-increment version |
| `load(name, version)` | Load model by name, latest version if unspecified |
| `list_models(name)` | List all saved models |
| `compare(model_ids)` | Print side-by-side metadata table |
| `delete(name, version)` | Remove model from disk and database |

### FeatureStore

| Method | Description |
|--------|-------------|
| `FeatureStore(db_path)` | Standard init |
| `define(name, dtype, version, description)` | Register a feature schema |
| `set(name, entity_id, value, version)` | Store a feature value with type validation |
| `get(name, entity_id, version)` | Retrieve a feature value |
| `get_many(feature_specs, entity_id)` | Retrieve multiple features at once |
| `list_features()` | List all defined feature schemas |
| `history(name, entity_id)` | View values across all versions |

---

## Database

MiniFlow stores everything in a SQLite database at `~/.miniflow/miniflow.db` by default.

Override via constructor argument or environment variable:
```bash
MINIFLOW_DB_PATH=/custom/path/miniflow.db python your_script.py
```

---

## Why I Built This

I wanted to understand how MLflow works internally — not just use it as a black box. Every time I called `mlflow.log_metric()` I didn't really know what was happening underneath. Building MiniFlow from scratch forced me to understand that the abstraction is just a database write, a file save, and a metadata dict. This is also the first in a series of MLOps libraries I plan to build, working my way up from this minimal version to a production-grade system with a REST API, drift detection, and a web UI.

---

## Running Tests

```bash
pytest tests/ -v --cov=miniflow --cov-report=term-missing
# 24 passed, 82% coverage
```

---

## Quick Demo
[Run the full demo script](https://gist.github.com/Mannava-Daasaradhi/a2a70bf5a3aa05813b34d01d1718b53b)

*Engineered a foundational MLOps library from the ground up.*
