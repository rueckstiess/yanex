"""
Ensemble model that merges predictions from multiple trained models.

This creates a "merge" point in the dependency graph — multiple
branches converge back into a single experiment, similar to a
git merge.

Usage:
    yanex run ensemble_model.py -D "cnn=<cnn-id>,rnn=<rnn-id>"
"""

import time

import yanex

# Get dependencies by slot name
cnn = yanex.get_dependency("cnn")
rnn = yanex.get_dependency("rnn")

params = yanex.get_params()
ensemble_method = params.get("method", "weighted_average")
cnn_weight = params.get("cnn_weight", 0.6)
rnn_weight = params.get("rnn_weight", 0.4)

print(f"🔀 Building ensemble ({ensemble_method})")
if cnn:
    cnn_params = cnn.get_params()
    print(f"  CNN model: {cnn.id[:8]} (lr={cnn_params.get('learning_rate', '?')})")
if rnn:
    rnn_params = rnn.get_params()
    print(f"  RNN model: {rnn.id[:8]} (lr={rnn_params.get('learning_rate', '?')})")

print(f"  Weights: CNN={cnn_weight}, RNN={rnn_weight}")

# Simulate ensemble combination
time.sleep(0.3)

# Simulate that ensemble outperforms individual models
ensemble_accuracy = 0.96
ensemble_f1 = 0.95

yanex.log_metrics(
    {
        "accuracy": ensemble_accuracy,
        "f1_score": ensemble_f1,
        "method": ensemble_method,
        "cnn_weight": cnn_weight,
        "rnn_weight": rnn_weight,
    }
)

yanex.save_artifact(
    {
        "method": ensemble_method,
        "weights": {"cnn": cnn_weight, "rnn": rnn_weight},
        "accuracy": ensemble_accuracy,
    },
    "ensemble_model.pkl",
)

print(f"✓ Ensemble accuracy: {ensemble_accuracy:.4f}")
print(f"✓ Ensemble F1: {ensemble_f1:.4f}")
