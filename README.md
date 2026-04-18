# MiniFlow

Minimal MLOps primitives built from scratch. Work in progress.

# P1a — MiniFlow MLOps v0.1

> **Domain:** AI/ML · **Year:** 1 · **Months:** 1–1 · **Daily Time Budget:** 1.5 hrs  
> **Legal Status:** ✅ Safe — no legal restrictions  
> **Stack:** Python · SQLite · YAML · Git  
> **Signal:** GitHub stars + demonstrates production ML thinking to recruiters

---

## 1. What This Project Is

MiniFlow v0.1 is a **minimal, zero-dependency MLOps library** you build from scratch in pure Python. No MLflow. No Weights & Biases. No Neptune. You will implement the three core primitives that every real ML platform is built on:

1. **ExperimentTracker** — log, query, and compare ML runs backed by SQLite
2. **ModelRegistry** — version, save, load, and compare trained models with metadata
3. **FeatureStore** — define features with schemas, version them, and retrieve by key

This is not a toy. By the end, you will have a library that you would genuinely reach for on a solo ML project. It also proves to any ML engineering interviewer that you understand what MLflow is *doing under the hood* — because you built it yourself.

---

## 2. Problem Statement

Every ML engineer uses experiment tracking tools, but almost none can explain what they are actually storing or why. When you run `mlflow.log_metric("loss", 0.5)`, what happens? Where does it go? How does the UI reconstruct a run comparison?

The answer is embarrassingly simple: a database write. A file save. A metadata dict. The abstraction is thin — but only if you've built it yourself.

**Your job:** build that thin abstraction. Three classes, pure Python, SQLite backend. When you're done, the following must work from a cold Python REPL with no imports except your library:

```python
from miniflow import ExperimentTracker, ModelRegistry, FeatureStore

# ExperimentTracker
t = ExperimentTracker("my_experiment")
t.log_params({"lr": 0.001, "epochs": 10, "model": "resnet50"})
t.log_metric("loss", 0.523, step=1)
t.log_metric("loss", 0.201, step=10)
t.log_metric("accuracy", 0.94, step=10)
runs = t.get_runs()
print(runs)  # all runs with params + metrics visible

# ModelRegistry
registry = ModelRegistry()
registry.save("resnet50_v1", model_obj, metadata={"val_acc": 0.94, "trained_on": "ImageNet"})
loaded = registry.load("resnet50_v1")
registry.compare(["resnet50_v1", "resnet50_v2"])  # side-by-side metadata diff

# FeatureStore
fs = FeatureStore()
fs.define("user_age", dtype="int", version=1, description="User age in years")
fs.set("user_age", entity_id="user_123", value=27, version=1)
val = fs.get("user_age", entity_id="user_123")
print(val)  # 27
```

That single REPL session is your definition of done.

---

## 3. Deliverables

| # | Deliverable | Where |
|---|-------------|-------|
| 1 | `miniflow/` Python package (3 modules) | `src/miniflow/` |
| 2 | SQLite schema for runs, metrics, params, models, features | auto-created on import |
| 3 | Full test suite (pytest, 80%+ coverage) | `tests/` |
| 4 | CLI tool: `miniflow runs list`, `miniflow runs compare <id1> <id2>` | `miniflow/cli.py` |
| 5 | `README.md` with install + usage examples | root |
| 6 | GitHub Gist with a self-contained demo script | link in README |
| 7 | PyPI-ready `pyproject.toml` (not published yet, but structured correctly) | root |

---

## 4. Architecture

```
miniflow/
├── __init__.py              # exports ExperimentTracker, ModelRegistry, FeatureStore
├── tracker.py               # ExperimentTracker class
├── registry.py              # ModelRegistry class
├── feature_store.py         # FeatureStore class
├── db.py                    # SQLite connection pool + schema creation
├── cli.py                   # Click-based CLI
└── utils.py                 # git_hash(), timestamp(), generate_run_id()

tests/
├── test_tracker.py
├── test_registry.py
├── test_feature_store.py
└── conftest.py              # tmp_path fixture, in-memory SQLite

pyproject.toml
README.md
```

The SQLite database file lives at `~/.miniflow/miniflow.db` by default. Users can override via `MINIFLOW_DB_PATH` environment variable or constructor argument.

---

## 5. Step-by-Step Build Instructions

Work through these in order. Do not skip ahead. Each step is one focused session.

---

### Step 1 — Project Skeleton (Session 1, ~45 min)

**Goal:** importable package with no logic yet.

```bash
mkdir miniflow-project && cd miniflow-project
git init
python -m venv .venv && source .venv/bin/activate
pip install pytest pytest-cov click pyyaml
mkdir -p src/miniflow tests
touch src/miniflow/__init__.py
touch src/miniflow/{tracker,registry,feature_store,db,utils,cli}.py
touch tests/{test_tracker,test_registry,test_feature_store,conftest}.py
touch pyproject.toml README.md
```

**`pyproject.toml`** (fill this in now — be disciplined about metadata from day one):

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "miniflow"
version = "0.1.0"
description = "Minimal MLOps primitives: ExperimentTracker, ModelRegistry, FeatureStore"
authors = [{name = "Your Name", email = "you@example.com"}]
requires-python = ">=3.10"
dependencies = ["click", "pyyaml"]

[project.scripts]
miniflow = "miniflow.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

Install in editable mode: `pip install -e .`

Verify: `python -c "import miniflow; print('ok')"` — must not error.

---

### Step 2 — Database Layer (Session 1 continued, ~30 min)

**File:** `src/miniflow/db.py`

This is the foundation. Everything else reads/writes through here.

**Schema to implement:**

```sql
-- Runs table: one row per experiment run
CREATE TABLE IF NOT EXISTS runs (
    run_id    TEXT PRIMARY KEY,   -- UUID4
    name      TEXT NOT NULL,      -- experiment name
    git_hash  TEXT,               -- output of git rev-parse HEAD
    timestamp TEXT NOT NULL,      -- ISO 8601 UTC
    status    TEXT DEFAULT 'running'  -- running | finished | failed
);

-- Params: hyperparameters logged for a run (string key-value)
CREATE TABLE IF NOT EXISTS params (
    run_id TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Metrics: numeric values over steps
CREATE TABLE IF NOT EXISTS metrics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     REAL NOT NULL,
    step      INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Model registry
CREATE TABLE IF NOT EXISTS models (
    model_id   TEXT PRIMARY KEY,    -- "name_v{version}"
    name       TEXT NOT NULL,
    version    INTEGER NOT NULL,
    path       TEXT NOT NULL,       -- absolute path to serialized artifact
    metadata   TEXT NOT NULL,       -- JSON blob
    created_at TEXT NOT NULL,
    UNIQUE(name, version)
);

-- Feature store
CREATE TABLE IF NOT EXISTS feature_definitions (
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    dtype       TEXT NOT NULL,
    description TEXT,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (name, version)
);

CREATE TABLE IF NOT EXISTS feature_values (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    version    INTEGER NOT NULL,
    entity_id  TEXT NOT NULL,
    value      TEXT NOT NULL,  -- JSON-serialized
    updated_at TEXT NOT NULL,
    UNIQUE(name, version, entity_id)
);
```

**What to implement in `db.py`:**
- `get_connection(db_path: str) -> sqlite3.Connection` — returns a connection with `row_factory = sqlite3.Row` and WAL mode enabled
- `init_db(db_path: str)` — runs all CREATE TABLE IF NOT EXISTS statements above
- A module-level constant `DEFAULT_DB_PATH = Path.home() / ".miniflow" / "miniflow.db"`

**Test it:** write `tests/conftest.py` with a `tmp_db` fixture that creates an in-memory database:
```python
import pytest
from miniflow.db import get_connection, init_db

@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    return db_path
```

---

### Step 3 — ExperimentTracker (Session 2, ~60 min)

**File:** `src/miniflow/tracker.py`

**Full interface to implement:**

```python
class ExperimentTracker:
    def __init__(self, name: str, db_path: str = None):
        """
        Creates a new run immediately on instantiation.
        - name: experiment name (e.g. "resnet50_imagenet")
        - db_path: path to SQLite DB; defaults to DEFAULT_DB_PATH
        Sets self.run_id = uuid4(), self.name = name
        Inserts row into runs table with git_hash from utils.get_git_hash()
        """

    def log_params(self, params: dict) -> None:
        """
        Inserts key-value pairs into params table for current run.
        All values are str(value) — params are always strings in storage.
        Raises ValueError if a key is logged twice for the same run.
        """

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """
        Inserts one metric row. Multiple calls with same key = time series.
        step must be >= 0. value must be finite float (reject NaN, inf).
        """

    def finish(self, status: str = "finished") -> None:
        """Updates runs.status to 'finished' or 'failed'."""

    def get_runs(self, name: str = None, limit: int = 20) -> list[dict]:
        """
        Returns list of run dicts. Each dict has:
        {run_id, name, git_hash, timestamp, status, params: {}, metrics: {key: [values]}}
        If name given, filter by experiment name.
        """

    def compare(self, run_ids: list[str]) -> dict:
        """
        Returns side-by-side comparison dict:
        {
          "run_ids": [...],
          "params": {"lr": {"run_abc": 0.001, "run_def": 0.01}, ...},
          "metrics": {"loss": {"run_abc": [0.5, 0.2], "run_def": [0.8, 0.3]}}
        }
        """

    def get_best_run(self, metric: str, mode: str = "min") -> dict:
        """
        Returns run dict for best run by final value of `metric`.
        mode="min" for loss, mode="max" for accuracy.
        """
```

**Tests to write in `tests/test_tracker.py`:**
1. `test_new_run_creates_db_entry` — after `ExperimentTracker("test")`, query DB directly, assert row exists
2. `test_log_params_stores_correctly` — log `{"lr": 0.01}`, call `get_runs()`, assert `runs[0]["params"]["lr"] == "0.01"`
3. `test_log_metric_time_series` — log metric at steps 0, 1, 2; assert `get_runs()[0]["metrics"]["loss"] == [0.5, 0.3, 0.1]`
4. `test_duplicate_param_raises` — log same key twice, assert `ValueError`
5. `test_nan_metric_raises` — `log_metric("loss", float("nan"))` raises `ValueError`
6. `test_compare_runs` — create 2 runs with different params, call `compare()`, verify structure
7. `test_get_best_run_min` — 3 runs with different final losses, assert best run has lowest loss

---

### Step 4 — ModelRegistry (Session 3, ~45 min)

**File:** `src/miniflow/registry.py`

**Full interface:**

```python
import pickle, json
from pathlib import Path

class ModelRegistry:
    def __init__(self, storage_dir: str = None, db_path: str = None):
        """
        storage_dir: where model artifacts (pickle files) are saved
        Defaults to ~/.miniflow/models/
        """

    def save(self, name: str, model_obj: object, metadata: dict = None) -> str:
        """
        Saves model_obj with pickle to storage_dir/{name}_v{version}.pkl
        Auto-increments version (first save = version 1).
        Stores metadata as JSON in models table.
        Returns model_id string: "{name}_v{version}"
        """

    def load(self, name: str, version: int = None) -> object:
        """
        Loads model from pickle file.
        If version=None, loads latest version.
        Raises FileNotFoundError if name+version not found.
        """

    def list_models(self, name: str = None) -> list[dict]:
        """
        Returns list of model dicts: {model_id, name, version, path, metadata, created_at}
        If name given, filter to that model family.
        """

    def compare(self, model_ids: list[str]) -> None:
        """
        Prints a human-readable side-by-side table of metadata for given model_ids.
        Use only print() — no external dependencies.
        Example output:
          key          resnet50_v1    resnet50_v2
          -----------  -------------  -------------
          val_acc      0.94           0.96
          trained_on   ImageNet       ImageNet-21k
        """

    def delete(self, name: str, version: int) -> None:
        """
        Removes pkl file from disk AND deletes row from models table.
        Raises ValueError if model is referenced by any run (future-proofing — check runs metadata).
        """
```

**Tests to write:**
1. `test_save_increments_version` — save same name twice, assert versions are 1 and 2
2. `test_load_latest` — save v1 and v2, `load("mymodel")` returns v2 object
3. `test_load_specific_version` — `load("mymodel", version=1)` returns v1 object
4. `test_compare_output` — save 2 models with different metadata, call `compare()`, assert no exceptions (output is printed, not returned)
5. `test_delete_removes_file` — save, delete, assert pkl file gone and DB row gone
6. `test_load_missing_raises` — `load("nonexistent")` raises `FileNotFoundError`

---

### Step 5 — FeatureStore (Session 4, ~45 min)

**File:** `src/miniflow/feature_store.py`

**Full interface:**

```python
import json

class FeatureStore:
    def __init__(self, db_path: str = None):
        """Standard init, same db_path pattern as above."""

    def define(self, name: str, dtype: str, version: int = 1, description: str = "") -> None:
        """
        Registers a feature schema.
        dtype must be one of: "int", "float", "str", "list", "dict"
        Raises ValueError on duplicate (name, version).
        """

    def set(self, name: str, entity_id: str, value: object, version: int = 1) -> None:
        """
        Stores feature value for an entity.
        Validates that (name, version) is defined. Raises KeyError if not.
        Validates that type(value) matches dtype. Raises TypeError if not.
        Value is stored as JSON string (json.dumps).
        If (name, version, entity_id) already exists: UPDATE. Else: INSERT.
        """

    def get(self, name: str, entity_id: str, version: int = 1) -> object:
        """
        Returns feature value, deserialized from JSON.
        Raises KeyError if entity_id not found for this feature.
        """

    def get_many(self, feature_specs: list[tuple], entity_id: str) -> dict:
        """
        feature_specs: list of (name, version) tuples
        Returns {name: value, ...} dict for the entity.
        Missing features returned as None (not raised).
        Example:
            fs.get_many([("age", 1), ("income", 1)], "user_123")
            -> {"age": 27, "income": 95000.0}
        """

    def list_features(self) -> list[dict]:
        """Returns all defined feature schemas."""

    def history(self, name: str, entity_id: str) -> list[dict]:
        """
        Returns all historical values for (name, entity_id) across all versions.
        Useful for debugging feature drift.
        """
```

**Tests to write:**
1. `test_define_and_get` — define feature, set value, get value, assert equal
2. `test_dtype_validation` — define `dtype="int"`, call `set(value="hello")`, assert `TypeError`
3. `test_upsert_behavior` — set same entity twice, assert only latest value returned
4. `test_undefined_feature_raises` — `set` on undefined feature raises `KeyError`
5. `test_get_many` — define 3 features, set values, `get_many` returns all 3
6. `test_get_missing_entity_raises` — `get` for entity not set raises `KeyError`

---

### Step 6 — Utility Functions (Session 4 continued, ~20 min)

**File:** `src/miniflow/utils.py`

```python
import uuid, subprocess
from datetime import datetime, timezone

def generate_run_id() -> str:
    """Returns str(uuid.uuid4())"""

def get_git_hash() -> str | None:
    """
    Runs: git rev-parse --short HEAD
    Returns 7-char hash string, or None if not in a git repo.
    Must not raise — catch CalledProcessError and return None.
    """

def now_iso() -> str:
    """Returns datetime.now(timezone.utc).isoformat()"""

def validate_finite(value: float, name: str) -> None:
    """Raises ValueError if value is NaN or infinite."""
```

---

### Step 7 — CLI (Session 5, ~45 min)

**File:** `src/miniflow/cli.py`

Use [Click](https://click.palletsprojects.com/). Implement these commands:

```
miniflow runs list [--name EXPERIMENT_NAME] [--limit 20]
miniflow runs compare <run_id_1> <run_id_2> [<run_id_3> ...]
miniflow runs best --metric METRIC_NAME [--mode min|max]
miniflow models list [--name MODEL_NAME]
miniflow models compare <model_id_1> <model_id_2>
miniflow features list
```

Each command calls the relevant class method and pretty-prints the result. For tabular output, implement a simple `print_table(headers, rows)` helper using only Python's built-in `str.ljust()`.

Example output for `miniflow runs list`:
```
RUN_ID    NAME           TIMESTAMP              STATUS     PARAMS           BEST_METRIC
--------  -------------  ---------------------  ---------  ---------------  -----------
a1b2c3    my_exp         2026-04-13T06:00:00Z   finished   lr=0.001 ep=10   loss=0.201
d4e5f6    my_exp         2026-04-13T07:30:00Z   running    lr=0.01  ep=20   loss=0.312
```

---

### Step 8 — Integration Test + Polish (Session 6, ~45 min)

Write `tests/test_integration.py` that runs the exact sequence from Section 2 (the REPL demo) as a pytest test. Every line must pass.

Then run the full test suite:
```bash
pytest tests/ -v --cov=miniflow --cov-report=term-missing
```

Target: **80%+ coverage, 0 failures.**

---

### Step 9 — GitHub + Gist (Session 6 continued, ~20 min)

1. Push to your `miniflow` GitHub repo (created on Day 1)
2. Write a self-contained demo script `demo.py` — it must work on any machine with `pip install miniflow` (once you publish) or `pip install -e .`
3. Post `demo.py` as a public GitHub Gist. Link to it in `README.md` under a "Quick Demo" section
4. Write a proper `README.md` (for the repo, not this document) following the structure: badges → one-line description → install → quick start → full API reference → why I built this

---

## 6. Definition of Done

This project is **complete** when ALL of the following are true — no exceptions:

```bash
# 1. This exact command works from a fresh Python environment:
python -c "
from miniflow import ExperimentTracker
t = ExperimentTracker('test')
t.log_metric('loss', 0.5, step=1)
print(t.get_runs())
"
# Must print a list with one run dict, no errors.

# 2. Full test suite passes with 80%+ coverage:
pytest tests/ --cov=miniflow --cov-report=term-missing
# 0 failed, 0 errors

# 3. CLI works:
miniflow runs list
miniflow models list
miniflow features list
# Each prints a table (may be empty) without errors.

# 4. GitHub repo is public with at minimum:
#    - All source files committed
#    - README.md with install instructions and quick-start
#    - At least one tagged release: git tag v0.1.0 && git push --tags

# 5. Gist exists and is linked in README.md
```

---

## 7. Common Mistakes to Avoid

| Mistake | Why It Kills You | Fix |
|---------|-----------------|-----|
| Using MLflow as a reference for code | You'll copy instead of understand | Look at MLflow UI only, never its source during build |
| Storing metrics as JSON blobs instead of rows | Can't query time series efficiently | One row per metric per step — the schema above is correct |
| Not handling the case where git is not installed | Crashes on CI machines | `get_git_hash()` must return `None` gracefully |
| Making `db_path` a module-level singleton | Breaks parallel tests | Pass `db_path` through constructors, use `tmp_path` in tests |
| Pickling PyTorch models with torch.save instead of pickle | Fine for now, but inconsistent | Use `pickle` for v0.1; note in README that torch.save is better for PyTorch models |
| Writing tests after writing all code | You won't write them | Write test stubs in Step 2, fill them in as you implement |

---

## 8. Extension Ideas (Do These After P1b, Not Now)

- **Artifact logging:** `tracker.log_artifact("confusion_matrix.png", file_path)` — store files alongside runs
- **Tags:** `tracker.set_tag("env", "production")` — free-form labels on runs
- **Run diffing:** given two run IDs, output a colored diff of params + final metrics
- **Export:** `tracker.export_runs(format="csv")` — dump all runs to pandas-compatible CSV
- **Web UI:** 50-line Flask app that renders runs as an HTML table (save this for P1b)

---

## 9. How This Connects to the Rest of the Plan

| Future Project | How P1a Feeds It |
|----------------|-----------------|
| **P1b** (MiniFlow v2.0) | You extend this exact codebase with FastAPI + drift detection |
| **P2** (LLM from scratch) | You use ExperimentTracker to log perplexity curves during training |
| **P5** (ML network IDS) | ExperimentTracker logs anomaly detector training runs |
| **P42** (ML Feature Store Production) | This is the toy version; P42 is the production version with Redis + Kafka |
| **P39** (LLM Inference Server) | ModelRegistry pattern is what production model servers use |
| **All ML projects** | Every training run you do should be tracked here — builds real usage history |

---

## 10. Time Breakdown

| Session | What You Do | Time |
|---------|-------------|------|
| 1 | Skeleton + DB schema + `db.py` | 75 min |
| 2 | `tracker.py` + `test_tracker.py` | 90 min |
| 3 | `registry.py` + `test_registry.py` | 60 min |
| 4 | `feature_store.py` + `utils.py` + their tests | 75 min |
| 5 | CLI + manual smoke test | 45 min |
| 6 | Integration test + polish + GitHub + Gist | 60 min |
| **Total** | | **~7.5 hrs across Month 1** |

At 1.5 hrs/day this is a 5-day project. Finish it in Week 1.

---

*This README is part of the Sovereign Vantablack 365-day plan. Next project: [P1b — MiniFlow v2.0 Production](../P1b_MiniFlow_v2.0_Production/README.md)*
#MINIFLOW
 
 
