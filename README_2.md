# P1b — MiniFlow v2.0 Production

> **Domain:** AI/ML · **Year:** 1 · **Months:** 1–4 · **Daily Time Budget:** 1 hr  
> **Legal Status:** ✅ Safe — no legal restrictions  
> **Stack:** Python · FastAPI · Prometheus · Redis · Docker  
> **Prerequisite:** P1a must be fully complete and passing all tests  
> **Signal:** All ML engineer roles — demonstrates production systems thinking

---

## 1. What This Project Is

MiniFlow v2.0 takes the SQLite-backed library you built in P1a and turns it into a **production ML serving system**. Three new components are added on top of the existing codebase:

1. **FastAPI `/predict` endpoint** — a model serving API with input validation, error handling, and latency tracking
2. **Drift Detector** — monitors incoming prediction requests for distribution shift using Population Stability Index (PSI), alerts when the model's input distribution drifts from the training distribution
3. **A/B Testing Framework** — assigns users to model variants, tracks conversion events, and runs statistical significance tests to determine winners

This is not an extension of P1a in the same file. You create a new `miniflow-serving/` repository that imports `miniflow` as a dependency and adds these three systems on top.

By the end, a Grafana dashboard will show live traffic, drift alerts, and A/B test results — all driven by your code.

---

## 2. Problem Statement

Shipping a model to production is not `model.predict(x)`. The real problems are:

- **Serving:** How do you handle malformed inputs without crashing? How do you track latency per endpoint? How do you hot-reload a model without downtime?
- **Drift:** Your model was trained on March data. It's now September. The input distribution has silently shifted and your model is wrong — but no error is thrown. How do you detect this before your users do?
- **A/B Testing:** You have Model A (in production) and Model B (your new candidate). You can't just swap them — you need statistical evidence that B is better. How do you run this experiment on live traffic without writing a statistics textbook?

These are the problems every ML engineer hits in their first production deployment. You will solve all three.

**Your definition of done in one command:**

```bash
# Start the stack
docker compose up

# Send a prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 0.5, 3.1], "user_id": "user_123"}'

# Expected response:
{
  "prediction": 0.87,
  "model_variant": "B",
  "latency_ms": 4.2,
  "request_id": "a1b2c3d4"
}

# After sending 500 requests with drifted inputs:
# Grafana at http://localhost:3000 shows:
#   - PSI alert: DRIFT DETECTED (PSI=0.28 > threshold 0.2)
#   - A/B test panel: Variant B CTR=0.34 vs Variant A CTR=0.29 (p=0.03, significant)
```

---

## 3. Deliverables

| # | Deliverable | Where |
|---|-------------|-------|
| 1 | FastAPI app with `/predict`, `/health`, `/metrics` endpoints | `app/main.py` |
| 2 | Drift detector class using PSI | `app/drift.py` |
| 3 | A/B testing framework | `app/ab_test.py` |
| 4 | Prometheus metrics instrumentation | `app/metrics.py` |
| 5 | Redis integration for A/B assignment persistence | `app/store.py` |
| 6 | Docker Compose stack (API + Redis + Prometheus + Grafana) | `docker-compose.yml` |
| 7 | Grafana dashboard JSON (importable) | `grafana/dashboard.json` |
| 8 | Full test suite (pytest + httpx for async tests) | `tests/` |
| 9 | Load test script showing latency under 100 req/sec | `scripts/load_test.py` |

---

## 4. Architecture

```
miniflow-serving/
├── app/
│   ├── main.py           # FastAPI app, /predict /health /metrics
│   ├── drift.py          # DriftDetector: PSI computation + alerting
│   ├── ab_test.py        # ABTestFramework: variant assignment + significance
│   ├── metrics.py        # Prometheus counters/histograms
│   ├── store.py          # Redis client wrapper
│   └── model_loader.py   # Hot-reload model from ModelRegistry (P1a)
├── tests/
│   ├── test_predict.py
│   ├── test_drift.py
│   ├── test_ab_test.py
│   └── conftest.py
├── scripts/
│   ├── load_test.py      # httpx async load tester
│   └── simulate_drift.py # sends requests with drifted distribution
├── grafana/
│   └── dashboard.json
├── prometheus/
│   └── prometheus.yml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

**Request flow:**
```
Client
  → FastAPI /predict
    → ABTestFramework.assign_variant(user_id)       # Redis lookup or new assignment
    → model_loader.get_model(variant)               # load from ModelRegistry
    → model.predict(features)                        # inference
    → DriftDetector.record(features)                # async, non-blocking
    → Prometheus histogram.observe(latency)         # instrument
  ← JSON response {prediction, variant, latency_ms, request_id}

Background task (every 60s):
  → DriftDetector.compute_psi()
  → if PSI > 0.2: Prometheus gauge.set(1) → Grafana alert fires
```

---

## 5. Step-by-Step Build Instructions

---

### Step 1 — Project Setup (Session 1, ~30 min)

```bash
mkdir miniflow-serving && cd miniflow-serving
git init
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn[standard] redis prometheus-client httpx pytest pytest-asyncio pyyaml numpy scipy
pip install -e path/to/your/miniflow  # install P1a in editable mode

# Create directory structure
mkdir -p app tests scripts grafana prometheus
touch app/{main,drift,ab_test,metrics,store,model_loader}.py
touch tests/{test_predict,test_drift,test_ab_test,conftest}.py
touch scripts/{load_test,simulate_drift}.py
touch docker-compose.yml Dockerfile requirements.txt
```

Freeze requirements: `pip freeze > requirements.txt`

---

### Step 2 — Prometheus Metrics Layer (Session 1 continued, ~30 min)

**File:** `app/metrics.py`

Define all Prometheus metrics here. Every other module imports from here — never create metrics inline.

```python
from prometheus_client import Counter, Histogram, Gauge

# Total prediction requests, labeled by variant and status
PREDICTION_REQUESTS = Counter(
    "miniflow_prediction_requests_total",
    "Total prediction requests",
    ["variant", "status"]   # status: success | error | validation_error
)

# Prediction latency in seconds
PREDICTION_LATENCY = Histogram(
    "miniflow_prediction_latency_seconds",
    "Prediction latency",
    ["variant"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Current PSI score per feature (set every 60 seconds)
DRIFT_PSI_SCORE = Gauge(
    "miniflow_drift_psi_score",
    "Current PSI score for drift detection",
    ["feature_index"]
)

# 1 if drift detected, 0 if clean
DRIFT_ALERT = Gauge(
    "miniflow_drift_alert",
    "1 if drift detected (PSI > threshold), else 0"
)

# A/B test conversion rates
AB_CONVERSIONS = Counter(
    "miniflow_ab_conversions_total",
    "A/B test conversion events",
    ["variant"]
)

AB_EXPOSURES = Counter(
    "miniflow_ab_exposures_total",
    "A/B test exposure events (user assigned to variant)",
    ["variant"]
)
```

---

### Step 3 — Redis Store (Session 2, ~30 min)

**File:** `app/store.py`

Redis is used for two things: persisting A/B variant assignments (so the same user always gets the same variant) and storing a rolling window of recent feature vectors for PSI computation.

```python
import redis
import json
import os

class RedisStore:
    def __init__(self, host: str = None, port: int = 6379):
        host = host or os.getenv("REDIS_HOST", "localhost")
        self.client = redis.Redis(host=host, port=port, decode_responses=True)

    def get_variant_assignment(self, user_id: str) -> str | None:
        """
        Returns "A" or "B" if user was previously assigned, else None.
        Key: "ab:user:{user_id}"
        """

    def set_variant_assignment(self, user_id: str, variant: str, ttl_days: int = 30) -> None:
        """
        Stores variant assignment with TTL of ttl_days days.
        TTL prevents Redis from growing unbounded.
        """

    def push_feature_vector(self, features: list[float], max_window: int = 10000) -> None:
        """
        Appends features to a Redis list "drift:recent_features".
        Trims list to max_window most recent entries.
        Each entry is json.dumps(features).
        """

    def get_recent_features(self, n: int = 10000) -> list[list[float]]:
        """
        Returns last n feature vectors from drift:recent_features list.
        Each entry is json.loads'd back to list[float].
        """

    def ping(self) -> bool:
        """Returns True if Redis is reachable, False otherwise. Must not raise."""
```

**Test in `tests/conftest.py`:** use `fakeredis` library for unit tests so you don't need a real Redis instance:
```bash
pip install fakeredis
```

```python
import fakeredis
import pytest
from app.store import RedisStore

@pytest.fixture
def fake_store():
    store = RedisStore.__new__(RedisStore)
    store.client = fakeredis.FakeRedis(decode_responses=True)
    return store
```

---

### Step 4 — Drift Detector (Session 2 continued + Session 3, ~90 min)

**File:** `app/drift.py`

**What PSI is and why it works:**

Population Stability Index measures how much a distribution has shifted between a reference period (training data) and a current period (recent predictions). PSI is defined as:

```
PSI = Σ (P_current_i - P_reference_i) × ln(P_current_i / P_reference_i)
```

Where the sum is over bins (typically 10 equal-width or equal-frequency bins). Interpretation:
- PSI < 0.1: no significant shift, model is stable
- 0.1 ≤ PSI < 0.2: slight shift, monitor closely
- PSI ≥ 0.2: significant shift, **drift alert — investigate or retrain**

```python
import numpy as np
from typing import Optional
from app.store import RedisStore
from app.metrics import DRIFT_PSI_SCORE, DRIFT_ALERT

class DriftDetector:
    def __init__(
        self,
        reference_data: np.ndarray,   # shape (N, num_features) from training set
        store: RedisStore,
        n_bins: int = 10,
        psi_threshold: float = 0.2,
        min_samples: int = 100,       # don't compute PSI until we have this many samples
    ):
        """
        reference_data: numpy array of training feature vectors.
        Store the per-feature bin edges computed from reference_data.
        These bin edges are FIXED — they define the buckets for all future PSI computations.
        """
        self.n_features = reference_data.shape[1]
        self.psi_threshold = psi_threshold
        self.min_samples = min_samples
        self.store = store

        # Compute reference distributions (one per feature)
        # self.reference_proportions[i] = array of shape (n_bins,) summing to 1.0
        # self.bin_edges[i] = array of shape (n_bins+1,) defining bucket boundaries
        self.reference_proportions, self.bin_edges = self._compute_reference(reference_data, n_bins)

    def _compute_reference(
        self, data: np.ndarray, n_bins: int
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        For each feature column:
          1. Compute n_bins equal-frequency bin edges using np.percentile
          2. Compute the proportion of data points in each bin
          3. Add epsilon (1e-4) to all proportions to avoid log(0)
          4. Normalize so proportions sum to 1.0
        Returns (proportions_list, bin_edges_list)
        """

    def _compute_psi_for_feature(
        self, feature_idx: int, current_data: np.ndarray
    ) -> float:
        """
        Bins current_data using self.bin_edges[feature_idx].
        Computes current proportions (with epsilon, normalized).
        Returns PSI scalar for this feature.
        Formula: sum((current - reference) * ln(current / reference))
        """

    def record(self, features: list[float]) -> None:
        """
        Push feature vector to Redis store.
        Called on every prediction request. Must be fast — no computation here.
        """
        self.store.push_feature_vector(features)

    def compute_psi(self) -> dict[int, float]:
        """
        Pull recent feature vectors from Redis.
        If fewer than min_samples available: return {} (not enough data).
        Compute PSI for each feature.
        Update Prometheus gauges: DRIFT_PSI_SCORE.labels(feature_index=i).set(psi)
        Set DRIFT_ALERT to 1 if any feature PSI > psi_threshold, else 0.
        Return {feature_idx: psi_score, ...}
        """

    def is_drifted(self) -> bool:
        """Returns True if last compute_psi() found any feature above threshold."""
        return bool(DRIFT_ALERT._value.get())
```

**Tests to write in `tests/test_drift.py`:**

1. `test_no_drift_same_distribution` — reference = normal(0,1), current = normal(0,1), PSI should be < 0.1
2. `test_drift_detected_shifted_distribution` — reference = normal(0,1), current = normal(3,1), PSI should be > 0.2
3. `test_insufficient_samples_returns_empty` — fewer than min_samples pushed, `compute_psi()` returns `{}`
4. `test_psi_formula_manual` — compute PSI by hand for a 2-bin case, assert your implementation matches
5. `test_prometheus_gauge_updated` — after `compute_psi()`, assert `DRIFT_PSI_SCORE` gauge has correct value
6. `test_drift_alert_set` — drifted distribution triggers `DRIFT_ALERT = 1`

**Generating test data:**
```python
import numpy as np

# No drift
reference = np.random.normal(0, 1, size=(1000, 3))
current_clean = np.random.normal(0, 1, size=(500, 3))

# Drift
current_drifted = np.random.normal(3, 1, size=(500, 3))
```

---

### Step 5 — A/B Testing Framework (Session 3 continued + Session 4, ~90 min)

**File:** `app/ab_test.py`

**Statistical background you must understand before coding:**

You're running a two-proportion z-test. You have:
- Variant A: `n_A` exposures, `k_A` conversions → conversion rate `p_A = k_A / n_A`
- Variant B: `n_B` exposures, `k_B` conversions → conversion rate `p_B = k_B / n_B`

Null hypothesis: `p_A == p_B`. The z-statistic is:

```
pooled_p = (k_A + k_B) / (n_A + n_B)
SE = sqrt(pooled_p * (1 - pooled_p) * (1/n_A + 1/n_B))
z = (p_B - p_A) / SE
p_value = 2 * (1 - norm.cdf(abs(z)))   # two-tailed
```

If `p_value < 0.05`: the difference is statistically significant.

```python
import hashlib
import json
from scipy import stats
import numpy as np
from app.store import RedisStore
from app.metrics import AB_CONVERSIONS, AB_EXPOSURES

class ABTestFramework:
    def __init__(
        self,
        store: RedisStore,
        variant_weights: dict[str, float] = None,   # e.g. {"A": 0.5, "B": 0.5}
        significance_level: float = 0.05,
    ):
        """
        variant_weights: traffic split. Must sum to 1.0.
        Default: 50/50 A/B split.
        """
        self.store = store
        self.variant_weights = variant_weights or {"A": 0.5, "B": 0.5}
        self.significance_level = significance_level
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Raises ValueError if weights don't sum to 1.0 (within 1e-6 tolerance)."""

    def assign_variant(self, user_id: str) -> str:
        """
        1. Check Redis for existing assignment → return it if found (sticky assignment)
        2. If new user: deterministically assign using hash bucketing
           - hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 10000
           - bucket = hash_val / 10000  (float in [0, 1))
           - assign "A" if bucket < weight_A, else "B"
           - Store in Redis with TTL
        3. Increment AB_EXPOSURES counter for the assigned variant
        4. Return variant string ("A" or "B")

        IMPORTANT: Hash bucketing ensures the same user_id always maps to the same
        variant even without Redis (Redis is just a cache). This is critical for
        experiment integrity.
        """

    def record_conversion(self, user_id: str) -> None:
        """
        Look up user's variant from Redis.
        Increment AB_CONVERSIONS counter for that variant.
        If user not found in Redis: log warning, do nothing (don't guess variant).
        """

    def get_results(self) -> dict:
        """
        Pull exposure and conversion counts from Prometheus (or maintain in Redis).
        Compute conversion rates, z-statistic, p-value, confidence interval.

        Returns:
        {
          "variants": {
            "A": {"exposures": 523, "conversions": 152, "rate": 0.291},
            "B": {"exposures": 489, "conversions": 166, "rate": 0.340}
          },
          "winner": "B",            # or None if not significant
          "p_value": 0.031,
          "is_significant": True,
          "relative_lift": 0.168,   # (rate_B - rate_A) / rate_A
          "confidence_interval_95": [0.014, 0.083]  # CI on the difference
        }

        If either variant has 0 exposures: return {"error": "insufficient_data"}
        """

    def _two_proportion_z_test(
        self,
        n_a: int, k_a: int,
        n_b: int, k_b: int
    ) -> tuple[float, float]:
        """
        Returns (z_statistic, p_value) using the formula in Section 5 above.
        Handle edge case: if SE == 0 (identical rates), return (0.0, 1.0).
        """
```

**Storing conversion counts:** The cleanest approach is to store exposure/conversion counts in Redis hashes so they survive restarts:

```
Redis key: "ab:counts"
Fields: "A:exposures", "A:conversions", "B:exposures", "B:conversions"
Use HINCRBY for atomic increments.
```

**Tests to write in `tests/test_ab_test.py`:**

1. `test_sticky_assignment` — same user_id always gets same variant across 100 calls
2. `test_weight_split` — assign 10000 unique user_ids, assert ~50% in each variant (within 3%)
3. `test_record_conversion_unknown_user` — calling `record_conversion` for unknown user doesn't raise
4. `test_significant_result` — inject counts where B clearly wins (n=1000, p_A=0.1, p_B=0.2), assert `is_significant=True` and `winner="B"`
5. `test_not_significant_result` — inject counts where rates are nearly equal, assert `is_significant=False` and `winner=None`
6. `test_zero_exposures_returns_error` — no data yet, assert `get_results()` returns `{"error": "insufficient_data"}`
7. `test_z_test_formula` — manually compute z and p for known inputs, assert matches `_two_proportion_z_test()`

---

### Step 6 — Model Loader (Session 4 continued, ~30 min)

**File:** `app/model_loader.py`

This loads models from the P1a ModelRegistry and caches them in memory. It also supports hot-reloading when a new model version is registered.

```python
import threading
import time
from miniflow import ModelRegistry

class ModelLoader:
    def __init__(
        self,
        registry: ModelRegistry,
        variant_model_map: dict[str, str],  # {"A": "model_v1", "B": "model_v2"}
        reload_interval_seconds: int = 60,
    ):
        """
        Loads all models in variant_model_map on instantiation.
        Starts a background thread that reloads models every reload_interval_seconds.
        Uses a threading.RLock to make get_model() thread-safe.
        """
        self._models = {}
        self._lock = threading.RLock()
        self._registry = registry
        self._variant_model_map = variant_model_map
        self._load_all()
        self._start_reload_thread(reload_interval_seconds)

    def get_model(self, variant: str) -> object:
        """
        Thread-safe model retrieval.
        Raises KeyError if variant not in variant_model_map.
        """

    def _load_all(self) -> None:
        """Load all variants. Acquire lock during write."""

    def _start_reload_thread(self, interval: int) -> None:
        """
        Background daemon thread. Every `interval` seconds:
        1. Call _load_all()
        2. Log: "Models reloaded at {timestamp}"
        Daemon thread so it dies when main process exits.
        """
```

---

### Step 7 — FastAPI App (Session 5, ~60 min)

**File:** `app/main.py`

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, field_validator
import uuid, time
from contextlib import asynccontextmanager
from prometheus_client import make_asgi_app

app = FastAPI(title="MiniFlow Serving", version="2.0.0")

# Request/Response models
class PredictRequest(BaseModel):
    features: list[float]
    user_id: str

    @field_validator("features")
    @classmethod
    def features_not_empty(cls, v):
        if not v:
            raise ValueError("features must not be empty")
        return v

    @field_validator("user_id")
    @classmethod
    def user_id_not_empty(cls, v):
        if not v.strip():
            raise ValueError("user_id must not be blank")
        return v

class PredictResponse(BaseModel):
    prediction: float
    model_variant: str
    latency_ms: float
    request_id: str
```

**Endpoints to implement:**

**`POST /predict`**
```
1. Generate request_id = str(uuid.uuid4())[:8]
2. Record start_time = time.perf_counter()
3. ab_framework.assign_variant(user_id) → variant
4. model_loader.get_model(variant) → model
5. prediction = model.predict(features)  # your model interface
6. drift_detector.record(features)  # non-blocking, just push to Redis
7. latency_ms = (time.perf_counter() - start_time) * 1000
8. PREDICTION_LATENCY.labels(variant=variant).observe(latency_ms / 1000)
9. PREDICTION_REQUESTS.labels(variant=variant, status="success").inc()
10. Return PredictResponse
```

Error handling:
- If `model.predict` raises: `PREDICTION_REQUESTS.labels(variant=variant, status="error").inc()`, raise `HTTPException(500)`
- Pydantic validation errors: FastAPI handles these automatically as 422 — also increment `status="validation_error"` counter in a custom exception handler

**`GET /health`**
```json
{
  "status": "ok",
  "redis": "ok",          // or "degraded" if Redis unreachable
  "models_loaded": ["A", "B"],
  "drift_alert": false
}
```
Must return 200 even if Redis is degraded (degraded ≠ down). Return 503 only if models aren't loaded.

**`GET /metrics`**
Mount Prometheus ASGI app at `/metrics`. This is one line with FastAPI:
```python
from prometheus_client import make_asgi_app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

**`POST /convert`**
```python
class ConvertRequest(BaseModel):
    user_id: str

# Calls ab_framework.record_conversion(user_id)
# Returns {"status": "recorded", "user_id": user_id}
```

**`GET /ab/results`**
```python
# Returns ab_framework.get_results() directly
```

**Background drift computation** — add this with FastAPI's `lifespan`:
```python
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: start drift computation loop
    task = asyncio.create_task(drift_loop())
    yield
    # shutdown: cancel task
    task.cancel()

async def drift_loop():
    while True:
        await asyncio.sleep(60)
        drift_detector.compute_psi()

app = FastAPI(lifespan=lifespan, ...)
```

**Dependency injection** — initialize all components at startup using a module-level singleton pattern. In `main.py`:
```python
# These are initialized once at module load time
# In production you'd use FastAPI's dependency injection system
store = RedisStore()
drift_detector = DriftDetector(reference_data=load_reference_data(), store=store)
ab_framework = ABTestFramework(store=store)
model_loader = ModelLoader(registry=ModelRegistry(), variant_model_map={"A": "model_v1", "B": "model_v2"})
```

**Tests in `tests/test_predict.py`** — use `httpx.AsyncClient` with `pytest-asyncio`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_predict_success():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/predict", json={"features": [1.0, 2.0, 3.0], "user_id": "u1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["model_variant"] in ["A", "B"]
    assert data["latency_ms"] > 0

@pytest.mark.asyncio
async def test_predict_empty_features_422():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/predict", json={"features": [], "user_id": "u1"})
    assert resp.status_code == 422

@pytest.mark.asyncio
async def test_health_returns_200():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code in [200, 503]
    assert "status" in resp.json()
```

---

### Step 8 — Docker Compose Stack (Session 6, ~60 min)

**`Dockerfile`:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install miniflow from local (bind mount in dev, copy in prod)
RUN pip install --no-cache-dir -e /miniflow-p1a 2>/dev/null || echo "miniflow not found, install manually"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**`docker-compose.yml`:**
```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ~/.miniflow:/root/.miniflow   # mount ModelRegistry storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:10.4.0
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    volumes:
      - ./grafana:/etc/grafana/provisioning/dashboards
    depends_on:
      - prometheus
```

**`prometheus/prometheus.yml`:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "miniflow-api"
    static_configs:
      - targets: ["api:8000"]
    metrics_path: /metrics
```

**Grafana Dashboard — `grafana/dashboard.json`:**

Create a dashboard with these 5 panels:
1. **Request Rate** — `rate(miniflow_prediction_requests_total[1m])` by variant
2. **Latency p50/p95/p99** — `histogram_quantile(0.95, rate(miniflow_prediction_latency_seconds_bucket[5m]))` by variant
3. **PSI Score per Feature** — `miniflow_drift_psi_score` gauge, with threshold line at 0.2
4. **Drift Alert** — `miniflow_drift_alert` single stat panel, red when = 1
5. **A/B Test Results** — call `/ab/results` via Grafana Infinity plugin or display raw counters

You can create this by running Grafana, building the dashboard in the UI, then exporting as JSON.

---

### Step 9 — Load Test Script (Session 7, ~30 min)

**`scripts/load_test.py`:**

```python
import asyncio
import httpx
import time
import statistics
import random

async def send_request(client: httpx.AsyncClient, user_id: str) -> float:
    start = time.perf_counter()
    resp = await client.post(
        "http://localhost:8000/predict",
        json={"features": [random.gauss(0, 1) for _ in range(3)], "user_id": user_id}
    )
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000  # ms

async def run_load_test(
    n_requests: int = 1000,
    concurrency: int = 50
):
    latencies = []
    errors = 0
    user_ids = [f"user_{i}" for i in range(200)]  # 200 unique users, creates repeat assignments

    async with httpx.AsyncClient(timeout=10.0) as client:
        semaphore = asyncio.Semaphore(concurrency)
        async def bounded_request():
            nonlocal errors
            async with semaphore:
                try:
                    lat = await send_request(client, random.choice(user_ids))
                    latencies.append(lat)
                except Exception as e:
                    errors += 1

        await asyncio.gather(*[bounded_request() for _ in range(n_requests)])

    print(f"\n=== Load Test Results ({n_requests} requests, {concurrency} concurrent) ===")
    print(f"Success: {len(latencies)} | Errors: {errors}")
    print(f"Latency p50:  {statistics.median(latencies):.1f}ms")
    print(f"Latency p95:  {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")
    print(f"Latency p99:  {sorted(latencies)[int(len(latencies)*0.99)]:.1f}ms")
    print(f"Throughput:   {len(latencies) / (sum(latencies)/1000/len(latencies) * len(latencies) / concurrency):.0f} req/s")

if __name__ == "__main__":
    asyncio.run(run_load_test())
```

**`scripts/simulate_drift.py`:**
```python
# Sends 500 requests with normal distribution (no drift)
# Then sends 500 requests with shifted distribution (drift)
# Watch Grafana dashboard during execution
```
Implement this yourself following the same pattern — it's just `run_load_test` with two phases where `features` are drawn from different distributions.

---

### Step 10 — Final Integration (Session 7 continued, ~30 min)

```bash
# Start the full stack
docker compose up --build

# In another terminal, run the full test suite
pytest tests/ -v --cov=app --cov-report=term-missing

# Run the load test
python scripts/load_test.py

# Simulate drift
python scripts/simulate_drift.py

# Check Grafana at http://localhost:3000
# You must see:
#   - Request rate graph showing traffic
#   - PSI scores rising during drift simulation phase
#   - Drift alert turning red when PSI > 0.2

# Check A/B results
curl http://localhost:8000/ab/results | python -m json.tool
```

---

## 6. Definition of Done

Every item must be true. No partial credit.

```bash
# 1. Stack starts cleanly
docker compose up
# No errors in logs after 30 seconds. All 4 containers healthy.

# 2. Predict endpoint works
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 0.5, 3.1], "user_id": "user_123"}'
# Returns 200 JSON with prediction, model_variant, latency_ms, request_id

# 3. Same user always gets same variant
for i in {1..5}; do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [1.0], "user_id": "sticky_test_user"}' | python -m json.tool | grep model_variant
done
# All 5 lines must show the same variant

# 4. Drift detection works
python scripts/simulate_drift.py
# After running: curl http://localhost:8000/health | python -m json.tool
# Shows "drift_alert": true

# 5. A/B results show statistical output
curl http://localhost:8000/ab/results | python -m json.tool
# Shows both variants with exposure counts, rates, p_value field present

# 6. Grafana dashboard shows live data at http://localhost:3000
# PSI score panel visible. Request rate panel visible.

# 7. Test suite passes
pytest tests/ -v
# 0 failed

# 8. Load test meets latency target
python scripts/load_test.py
# p99 latency < 100ms at 50 concurrent users
```

---

## 7. Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Computing PSI on every request | Kills latency | PSI is a background task every 60s. `record()` just pushes to Redis |
| Non-sticky A/B assignment | User sees different variants on refresh — experiment invalid | Always check Redis first; only assign if not found |
| Storing PSI bin edges from current data | PSI becomes meaningless — you're comparing to a moving reference | Bin edges are computed ONCE from reference (training) data and never change |
| Using `p_value < 0.05` as the only criterion | False positives at small sample sizes | Also require minimum sample size (e.g. n > 100 per variant) before declaring significance |
| Blocking FastAPI event loop during model inference | Kills throughput | Use `asyncio.get_event_loop().run_in_executor(None, model.predict, features)` if predict is CPU-bound |
| Not mounting Prometheus at `/metrics` | Prometheus can't scrape | One line: `app.mount("/metrics", make_asgi_app())` |
| Running Grafana without anonymous auth | Can't access dashboard in CI/demo | Set `GF_AUTH_ANONYMOUS_ENABLED=true` in compose |

---

## 8. How This Connects to the Rest of the Plan

| Future Project | How P1b Feeds It |
|----------------|-----------------|
| **P5** (ML network IDS) | Same FastAPI serving pattern — swap the model |
| **P6** (RAG pipeline) | `/predict` pattern generalizes to `/query` endpoint |
| **P39** (LLM Inference Server) | This is the toy version of production LLM serving |
| **P42** (ML Feature Store Production) | Drift detection + feature freshness monitoring are the same concept |
| **P46** (Real-Time ML Inference Pipeline) | Full production version of exactly this architecture |
| **P35** (Quantitative Backtest Engine) | A/B testing statistical framework reused for strategy comparison |

---

## 9. Time Breakdown

| Session | What You Do | Time |
|---------|-------------|------|
| 1 | Setup + Prometheus metrics + Redis store | 60 min |
| 2 | DriftDetector (PSI math + implementation + tests) | 90 min |
| 3 | ABTestFramework (z-test + Redis counts + tests) | 90 min |
| 4 | ModelLoader + FastAPI endpoints | 90 min |
| 5 | Docker Compose + Prometheus config | 60 min |
| 6 | Grafana dashboard + integration test | 60 min |
| 7 | Load test + drift simulation + polish | 60 min |
| **Total** | | **~8.5 hrs across Months 1–4** |

At 1 hr/day this is a background project. Run 1 session per week while other projects are the primary focus.

---

*Previous: [P1a — MiniFlow MLOps v0.1](../README.md) · Next project: [P2 — LLM from Scratch](../../P2_LLM_from_Scratch/README.md)*
