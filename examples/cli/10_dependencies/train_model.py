"""
Model training simulation with dependency on preprocessing.

Usage:
    yanex run train_model.py -D <prep-id> -p learning_rate=0.01
    yanex run train_model.py -D <id1>,<id2> -p "learning_rate=0.001,0.01,0.1"
"""

import time

import yanex

# Assert that we have a preprocessing dependency
# This will fail the experiment if the dependency is missing
yanex.assert_dependency("prepare_data.py")

# Get dependencies (preprocessing experiment)
deps = yanex.get_dependencies()

if deps:
    prep_exp = deps[0]
    print(f"📦 Loading preprocessed data from dependency {prep_exp.id}")

# Load preprocessed data artifact from dependency
data = yanex.load_artifact("processed_data.pkl")
if data:
    if deps:
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
if deps:
    metrics["preprocessing_id"] = deps[0].id

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
