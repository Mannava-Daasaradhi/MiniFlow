# demo.py — MiniFlow self-contained demo
# Run with: python demo.py

import uuid
from miniflow import ExperimentTracker, ModelRegistry, FeatureStore

print("=== ExperimentTracker ===")
print("Initializing tracker for 'demo_experiment'...")
tracker = ExperimentTracker("demo_experiment")

print(f"Logging parameters and metrics for run: {tracker.run_id}...")
tracker.log_params({"learning_rate": 0.005, "batch_size": 32, "model_type": "ResNet18"})
tracker.log_metric("loss", 0.85, step=1)
tracker.log_metric("loss", 0.42, step=2)
tracker.log_metric("accuracy", 0.91, step=2)

tracker.finish("finished")
runs = tracker.get_runs(name="demo_experiment", limit=1)
print("\nLatest run retrieved from database:")
for key, val in runs[0].items():
    print(f"  {key}: {val}")


print("\n=== ModelRegistry ===")
print("Initializing Model Registry...")
registry = ModelRegistry()

dummy_model = {"layer_1": [0.1, 0.5, 0.9], "bias": 0.01}
print("Saving dummy model artifact...")
model_id = registry.save(
    "resnet18_classifier", 
    dummy_model, 
    metadata={"val_acc": 0.91, "dataset": "ImageNet"}
)
print(f"Model saved successfully as: {model_id}")

print(f"Loading model '{model_id}' back from disk...")
loaded_model = registry.load("resnet18_classifier")
print(f"Loaded model data: {loaded_model}")


print("\n=== FeatureStore ===")
print("Initializing Feature Store...")
fs = FeatureStore()

# Generate a unique feature name to allow the demo to be run multiple times
# without hitting the unique constraint in the database.
feature_name = f"user_age_{str(uuid.uuid4())[:6]}"

print(f"Defining feature schema: {feature_name} (int)...")
fs.define(feature_name, dtype="int", version=1, description="User age in years")

print(f"Setting feature value for entity 'user_123'...")
fs.set(feature_name, entity_id="user_123", value=27, version=1)

print("Retrieving feature value...")
val = fs.get(feature_name, entity_id="user_123")
print(f"Retrieved value for 'user_123': {val} (Type: {type(val).__name__})")

print("\n=== Demo Complete! ===")
print("You can now use the CLI to view these entries:")
print("  miniflow runs list")
print("  miniflow models list")
print("  miniflow features list")