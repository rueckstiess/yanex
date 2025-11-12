#!/usr/bin/env python3
"""
Simple training script for batch execution example.

This script is executed multiple times by grid_search.py with different parameters.
"""

import random
import time

import yanex

# Get parameters
learning_rate = yanex.get_param("learning_rate", default=0.001)
batch_size = yanex.get_param("batch_size", default=32)
epochs = yanex.get_param("epochs", default=5)

print(f"Training with lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")

# Simulate training
time.sleep(2.0)

# Simulate results (better with lower LR, more epochs)
base_accuracy = 0.5 + (epochs * 0.08)
lr_factor = max(0, 1 - abs(learning_rate - 0.01) * 10)
accuracy = base_accuracy * lr_factor + random.uniform(-0.05, 0.05)
accuracy = min(0.95, max(0.3, accuracy))

loss = 1.0 - accuracy + random.uniform(-0.1, 0.1)
loss = max(0.05, loss)

print(f"Final accuracy: {accuracy:.3f}, loss: {loss:.3f}")

# Log results
yanex.log_metrics(
    {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "accuracy": accuracy,
        "loss": loss,
    }
)
