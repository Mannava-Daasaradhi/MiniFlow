# src/miniflow/cli.py

import click
from miniflow.tracker import ExperimentTracker
from miniflow.registry import ModelRegistry
from miniflow.feature_store import FeatureStore
from miniflow.db import DEFAULT_DB_PATH

def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Helper to print data in a nicely aligned table using only str.ljust()."""
    if not headers and not rows:
        return
        
    # Calculate max widths per column
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            # Safety check in case a row has missing columns
            if i < len(widths):
                widths[i] = max(widths[i], len(str(col)))
                
    # Print headers
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    
    # Print separator
    sep_line = "  ".join("-" * w for w in widths)
    print(sep_line)
    
    # Print rows
    for row in rows:
        row_line = "  ".join(str(c).ljust(w) for c, w in zip(row, widths))
        print(row_line)

def get_read_only_tracker() -> ExperimentTracker:
    """
    Instantiates ExperimentTracker without calling __init__
    to avoid creating a spurious new run in the database every 
    time the CLI is used to fetch data.
    """
    tracker = ExperimentTracker.__new__(ExperimentTracker)
    tracker.db_path = DEFAULT_DB_PATH
    return tracker

@click.group()
def main():
    """MiniFlow CLI - Minimal MLOps primitives."""
    pass

# --- RUNS COMMANDS ---

@main.group()
def runs():
    """Manage and view experiment runs."""
    pass

@runs.command("list")
@click.option("--name", default=None, help="Filter by experiment name")
@click.option("--limit", default=20, type=int, help="Number of runs to return")
def runs_list(name, limit):
    """List recent experiment runs."""
    tracker = get_read_only_tracker()
    runs_data = tracker.get_runs(name=name, limit=limit)
    
    if not runs_data:
        print("No runs found.")
        return
        
    headers = ["RUN_ID", "NAME", "TIMESTAMP", "STATUS", "PARAMS", "BEST_METRIC"]
    rows = []
    
    for r in runs_data:
        # Truncate UUID for clean table view
        run_id = r["run_id"][:6] 
        
        # Flatten params dict
        params_str = " ".join([f"{k}={v}" for k, v in r.get("params", {}).items()])
        
        # Extract the last value of the first available metric as a summary
        metrics = r.get("metrics", {})
        best_metric_str = ""
        if metrics:
            first_m = list(metrics.keys())[0]
            last_val = metrics[first_m][-1]
            if isinstance(last_val, float):
                best_metric_str = f"{first_m}={last_val:.4f}"
            else:
                best_metric_str = f"{first_m}={last_val}"
            
        rows.append([
            run_id,
            r["name"],
            r["timestamp"],
            r["status"],
            params_str,
            best_metric_str
        ])
        
    print_table(headers, rows)

@runs.command("compare")
@click.argument("run_ids", nargs=-1, required=True)
def runs_compare(run_ids):
    """Compare params and metrics across multiple run IDs."""
    tracker = get_read_only_tracker()
    comp = tracker.compare(list(run_ids))
    
    if not comp or not comp.get("run_ids"):
        print("No runs found or specified.")
        return
        
    headers = ["metric/param"] + comp["run_ids"]
    rows = []
    
    # Add params pivot block
    if comp["params"]:
        rows.append(["--- PARAMS ---"] + [""] * len(comp["run_ids"]))
        for p_key, p_dict in comp["params"].items():
            row = [p_key]
            for r_id in comp["run_ids"]:
                row.append(str(p_dict.get(r_id, "N/A")))
            rows.append(row)
            
    # Add metrics pivot block
    if comp["metrics"]:
        rows.append(["--- METRICS (Latest) ---"] + [""] * len(comp["run_ids"]))
        for m_key, m_dict in comp["metrics"].items():
            row = [m_key]
            for r_id in comp["run_ids"]:
                vals = m_dict.get(r_id, [])
                val_str = str(vals[-1]) if vals else "N/A"
                row.append(val_str)
            rows.append(row)
            
    print_table(headers, rows)

@runs.command("best")
@click.option("--metric", required=True, help="Metric name to optimize")
@click.option("--mode", default="min", type=click.Choice(["min", "max"]), help="min or max")
def runs_best(metric, mode):
    """Find the best run based on a specific metric."""
    tracker = get_read_only_tracker()
    try:
        best_run = tracker.get_best_run(metric, mode=mode)
    except ValueError as e:
        print(str(e))
        return
        
    print(f"Best run by {metric} ({mode}): {best_run['run_id']}")
    print(f"Name: {best_run['name']}")
    print(f"Timestamp: {best_run['timestamp']}")
    
    vals = best_run['metrics'].get(metric, [])
    if vals:
        best_val = min(vals) if mode == "min" else max(vals)
        print(f"Optimal {metric}: {best_val}")
        
    print("\nAll Params:")
    for k, v in best_run.get('params', {}).items():
        print(f"  {k}: {v}")

# --- MODELS COMMANDS ---

@main.group()
def models():
    """Manage and view registered models."""
    pass

@models.command("list")
@click.option("--name", default=None, help="Filter by model name")
def models_list(name):
    """List registered models."""
    registry = ModelRegistry()
    models_data = registry.list_models(name=name)
    
    if not models_data:
        print("No models found.")
        return
        
    headers = ["MODEL_ID", "NAME", "VERSION", "CREATED_AT", "METADATA"]
    rows = []
    for m in models_data:
        meta_str = " ".join([f"{k}={v}" for k, v in m.get("metadata", {}).items()])
        rows.append([
            m["model_id"],
            m["name"],
            str(m["version"]),
            m["created_at"],
            meta_str
        ])
        
    print_table(headers, rows)

@models.command("compare")
@click.argument("model_ids", nargs=-1, required=True)
def models_compare(model_ids):
    """Compare metadata across multiple model IDs."""
    registry = ModelRegistry()
    # The registry component already handles printing internally as per the spec
    registry.compare(list(model_ids))

# --- FEATURES COMMANDS ---

@main.group()
def features():
    """Manage and view feature definitions."""
    pass

@features.command("list")
def features_list():
    """List all defined feature schemas."""
    fs = FeatureStore()
    features_data = fs.list_features()
    
    if not features_data:
        print("No features defined.")
        return
        
    headers = ["NAME", "VERSION", "DTYPE", "DESCRIPTION", "CREATED_AT"]
    rows = []
    for f in features_data:
        rows.append([
            f["name"],
            str(f["version"]),
            f["dtype"],
            str(f["description"] or ""),
            f["created_at"]
        ])
        
    print_table(headers, rows)


if __name__ == "__main__":
    main()