"""
Model training simulation with dependency on preprocessing.

Usage:
    yanex run train_model.py -D data=<prep-id> -p learning_rate=0.01
    yanex run train_model.py -D "data=<id1>,<id2>" -p "learning_rate=0.001,0.01,0.1"
"""

import time

import yanex

# Assert that we have a preprocessing dependency
# This will fail the experiment if the dependency is missing or not assigned the "data" slot
yanex.assert_dependency("prepare_data.py", slot="data")

# Get specific dependency by slot name (recommended approach)
data_exp = yanex.get_dependency("data")

if data_exp:
    print(f"📦 Loading preprocessed data from dependency {data_exp.id}")
    # Show which dataset we're training on from the dependency's params
    dep_params = data_exp.get_params()
    if dep_params:
        print(f"  Dependency dataset: {dep_params.get('dataset', 'unknown')}")

# Load preprocessed data artifact from dependency
data = yanex.load_artifact("processed_data.pkl")
if data:
    print(f"  Dataset: {data['dataset']}")
    print(f"  Training samples: {data['train_size']}")
    print(f"  Features: {data['features']}")
else:
    print("  (Using simulated data - artifact not found)")
    data = {"dataset": "unknown", "train_size": 800, "features": 784}

# Get training parameters
params = yanex.get_params()
learning_rate = params.get("learning_rate", 0.01)
epochs = params.get("epochs", 10)

print(f"\n🤖 Training model with lr={learning_rate}")

# Simulate training
final_loss = 0.5  # Starting loss
for epoch in range(1, epochs + 1):
    time.sleep(0.1)  # Simulate epoch duration

    # Simulate loss decrease
    final_loss = final_loss * 0.85

    if epoch in [1, epochs // 2, epochs]:
        print(f"  Epoch {epoch}/{epochs}: loss={final_loss:.4f}")

# Simulate final accuracy based on learning rate
# (In reality this would come from actual model performance)
accuracy = min(0.95, 0.7 + (0.01 / learning_rate) * 0.1)

# Log training metrics
metrics = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "final_loss": final_loss,
    "train_accuracy": accuracy,
    "dataset": data["dataset"],
}
if data_exp:
    metrics["preprocessing_id"] = data_exp.id

yanex.log_metrics(metrics)

# Save simulated trained model
model_info = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "final_loss": final_loss,
    "accuracy": accuracy,
}

yanex.save_artifact(model_info, "trained_model.pkl")
print("✓ Trained model saved")
