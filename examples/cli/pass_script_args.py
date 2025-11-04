#!/usr/bin/env python3
"""
Example demonstrating script-specific argument passing in yanex.

This script shows how to combine yanex parameters (via -p/--param) with
script-specific arguments (passed directly to the script via sys.argv).

Use case: Training a model using data from a previous experiment

Run examples:
    # Basic usage (uses default demo_data)
    yanex run pass_script_args.py --ignore-dirty

    # With data experiment ID
    yanex run pass_script_args.py \\
        -p learning_rate=0.001 \\
        -p batch_size=32 \\
        --data-exp abc123 \\
        --fold 0 \\
        --ignore-dirty

    # With additional script arguments
    yanex run pass_script_args.py \\
        -p learning_rate=0.01 \\
        --data-exp abc123 \\
        --fold 2 \\
        --verbose \\
        --ignore-dirty

    # Parameter sweep with fixed script arguments
    yanex run pass_script_args.py \\
        -p 'learning_rate=list(0.001, 0.01, 0.1)' \\
        --data-exp abc123 \\
        --fold 0 \\
        --ignore-dirty
"""

import argparse

import yanex

# Parse script-specific arguments (operational parameters)
parser = argparse.ArgumentParser(description="Train model with data from experiment")
parser.add_argument(
    "--data-exp",
    default="demo_data",
    help="Experiment ID containing the training data (default: demo_data)",
)
parser.add_argument(
    "--fold",
    type=int,
    default=0,
    help="K-fold index to use for training (default: 0)",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose output",
)
parser.add_argument(
    "--output-format",
    choices=["json", "csv", "parquet"],
    default="json",
    help="Output format for predictions (default: json)",
)

args = parser.parse_args()

# Get configuration parameters from yanex (hyperparameters)
learning_rate = yanex.get_param("learning_rate", default=0.001)
batch_size = yanex.get_param("batch_size", default=32)
epochs = yanex.get_param("epochs", default=10)

# Display configuration
print("=" * 60)
print("Training Configuration")
print("=" * 60)
print("\nScript Arguments (Operational):")
print(f"  Data Experiment: {args.data_exp}")
print(f"  Fold: {args.fold}")
print(f"  Verbose: {args.verbose}")
print(f"  Output Format: {args.output_format}")

print("\nYanex Parameters (Hyperparameters):")
print(f"  Learning Rate: {learning_rate}")
print(f"  Batch Size: {batch_size}")
print(f"  Epochs: {epochs}")
print("=" * 60)

# Simulate loading data from the specified experiment
print(f"\nLoading data from experiment {args.data_exp}, fold {args.fold}...")
print("Data loaded: 1000 samples")

# Simulate training
print(f"\nTraining model for {epochs} epochs...")
print(f"Using learning rate: {learning_rate}, batch size: {batch_size}")

# Simulate logging metrics
for epoch in range(1, epochs + 1):
    accuracy = 0.5 + (epoch / epochs) * 0.45  # Simulate improving accuracy
    loss = 1.0 - (epoch / epochs) * 0.8  # Simulate decreasing loss

    yanex.log_metrics(
        {
            "epoch": epoch,
            "accuracy": accuracy,
            "loss": loss,
            "learning_rate": learning_rate,
        }
    )

    if args.verbose:
        print(f"  Epoch {epoch}/{epochs}: accuracy={accuracy:.4f}, loss={loss:.4f}")

print(f"\nTraining complete! Results saved in {args.output_format} format.")
print("\nKey Takeaway:")
print("- Use --param/-p for hyperparameters that you want to sweep over")
print(
    "- Use direct arguments for operational parameters (which experiment to use, etc.)"
)
