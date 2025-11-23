"""
Data preprocessing simulation for dependency tracking example.

Usage:
    yanex run prepare_data.py -p dataset=mnist
    yanex run prepare_data.py -p "dataset=mnist,cifar10" --parallel 2
"""

import time

import yanex

# Get parameters
params = yanex.get_params()
dataset = params.get("dataset", "mnist")
samples = params.get("samples", 1000)

print(f"📊 Preprocessing dataset: {dataset} ({samples} samples)")

# Simulate data preprocessing
time.sleep(0.5)  # Simulate processing time

# Simulate feature extraction
features = {"mnist": 784, "cifar10": 3072, "fashion": 784}.get(dataset, 784)
print(f"  Features extracted: {features}")

# Simulate train/test split
train_size = int(samples * 0.8)
test_size = samples - train_size
print(f"  Train/test split: {train_size}/{test_size}")

# Log metrics
yanex.log_metrics(
    {
        "dataset": dataset,
        "total_samples": samples,
        "train_samples": train_size,
        "test_samples": test_size,
        "num_features": features,
    }
)

# Save simulated preprocessed data as artifact
# In real workflow, this would be actual processed data
processed_data = {
    "dataset": dataset,
    "features": features,
    "train_size": train_size,
    "test_size": test_size,
}

yanex.save_artifact(processed_data, "processed_data.pkl")
print("✓ Preprocessed data saved")
