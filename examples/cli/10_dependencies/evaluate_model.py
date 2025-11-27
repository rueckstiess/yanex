"""
Model evaluation simulation with dependency on training.

Usage:
    yanex run evaluate_model.py -D model=<training-id>
    yanex run evaluate_model.py -D "model=<id1>,<id2>,<id3>" --parallel 3
"""

import time

import yanex

# Assert that we have a training dependency
# This will fail the experiment if the dependency is missing
yanex.assert_dependency("train_model.py", slot="model")

# Get specific dependency by slot name (recommended approach)
model_exp = yanex.get_dependency("model")

if model_exp:
    print(f"📦 Loading trained model from dependency {model_exp.id}")
    # Show training info from the dependency's params
    dep_params = model_exp.get_params()
    if dep_params:
        print(f"  Model trained with lr={dep_params.get('learning_rate', 'unknown')}")

# Load trained model artifact from dependency
model = yanex.load_artifact("trained_model.pkl")
if model:
    print(f"  Model learning rate: {model['learning_rate']}")
    print(f"  Training epochs: {model['epochs']}")
    print(f"  Training loss: {model['final_loss']:.4f}")
else:
    print("  (Using simulated model - artifact not found)")
    model = {"learning_rate": 0.01, "epochs": 10, "final_loss": 0.15}

# Can also access transitive dependencies (preprocessing)
all_deps = yanex.get_dependencies(transitive=True)
print(f"\n📊 Full pipeline: {len(all_deps)} dependencies")
for i, dep in enumerate(all_deps, 1):
    print(f"  {i}. {dep.id} ({dep.name or 'unnamed'})")

print("\n📈 Evaluating model...")

# Simulate evaluation
time.sleep(0.3)  # Simulate evaluation time

# Simulate test metrics (based on training metrics)
test_accuracy = model.get("accuracy", 0.90) * 0.95  # Slightly lower than training
test_loss = model["final_loss"] * 1.1  # Slightly higher than training

# Simulate confusion matrix values
tp = int(test_accuracy * 100)
fp = int((1 - test_accuracy) * 50)
fn = int((1 - test_accuracy) * 50)
tn = 100 - tp

print(f"  Test accuracy: {test_accuracy * 100:.1f}%")
print(f"  Test loss: {test_loss:.4f}")
print(f"  Precision: {tp / (tp + fp):.3f}")
print(f"  Recall: {tp / (tp + fn):.3f}")

# Log evaluation metrics
eval_metrics = {
    "test_accuracy": test_accuracy,
    "test_loss": test_loss,
    "precision": tp / (tp + fp),
    "recall": tp / (tp + fn),
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "true_negatives": tn,
}
if model_exp:
    eval_metrics["training_id"] = model_exp.id

yanex.log_metrics(eval_metrics)

# Save evaluation report
eval_report = {
    "test_accuracy": test_accuracy,
    "test_loss": test_loss,
    "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
}

yanex.save_artifact(eval_report, "evaluation_report.pkl")
print("✓ Evaluation complete")
