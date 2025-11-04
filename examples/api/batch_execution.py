#!/usr/bin/env python3
"""
Basic Batch Execution Example with Yanex

This example demonstrates how to use yanex.run_multiple() API to execute
multiple experiments in a batch, either sequentially or in parallel.

Use cases:
    - Grid search or random search over hyperparameters
    - Running multiple models with different configurations
    - Batch processing multiple datasets
    - Ensemble training

Usage:
    # Run the batch execution example
    python examples/api/batch_execution.py

    # Run with more parallelism
    python examples/api/batch_execution.py --parallel 4

Features demonstrated:
    - Creating ExperimentSpec objects
    - Sequential vs parallel execution
    - Error handling (continues on failure)
    - Accessing experiment results
"""

import argparse
import random
from pathlib import Path

import yanex


def example_1_hyperparameter_grid():
    """Example 1: Grid search over learning rates."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Hyperparameter Grid Search")
    print("=" * 60 + "\n")

    # Define hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32]

    # Create training script
    train_script = Path(__file__).parent / "train_simple.py"

    # Build experiment specs for grid search
    experiments = []
    for lr in learning_rates:
        for batch_size in batch_sizes:
            experiments.append(
                yanex.ExperimentSpec(
                    script_path=train_script,
                    config={
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "epochs": 10,
                    },
                    name=f"grid-lr{lr}-bs{batch_size}",
                    tags=["grid-search", "hyperparameter-tuning"],
                    description=f"Grid search: lr={lr}, batch_size={batch_size}",
                )
            )

    print(f"Created {len(experiments)} experiment configurations")
    print(f"Grid: learning_rate={learning_rates}, batch_size={batch_sizes}")
    print("\nNote: This example requires 'train_simple.py' - see below for creation")

    # Uncomment to actually run:
    # results = yanex.run_multiple(experiments, parallel=2, allow_dirty=True)
    # print_results_summary(results)


def example_2_random_search():
    """Example 2: Random search over hyperparameter space."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Random Hyperparameter Search")
    print("=" * 60 + "\n")

    # Sample random configurations
    n_samples = 10
    random.seed(42)

    train_script = Path(__file__).parent / "train_simple.py"

    experiments = []
    for i in range(n_samples):
        # Sample from hyperparameter distributions
        lr = 10 ** random.uniform(-4, -1)  # Log scale: 0.0001 to 0.1
        dropout = random.uniform(0.1, 0.5)
        hidden_size = random.choice([64, 128, 256, 512])

        experiments.append(
            yanex.ExperimentSpec(
                script_path=train_script,
                config={
                    "learning_rate": lr,
                    "dropout": dropout,
                    "hidden_size": hidden_size,
                    "epochs": 20,
                },
                name=f"random-search-{i}",
                tags=["random-search", "hyperparameter-tuning"],
                description=f"Random search iteration {i}",
            )
        )

    print(f"Created {n_samples} random configurations")
    print("Hyperparameter distributions:")
    print("  - learning_rate: log uniform [0.0001, 0.1]")
    print("  - dropout: uniform [0.1, 0.5]")
    print("  - hidden_size: choice [64, 128, 256, 512]")

    # Uncomment to actually run:
    # results = yanex.run_multiple(experiments, parallel=4, allow_dirty=True)
    # print_results_summary(results)


def example_3_multiple_datasets():
    """Example 3: Train same model on multiple datasets."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Training on Multiple Datasets")
    print("=" * 60 + "\n")

    datasets = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]
    train_script = Path(__file__).parent / "train_simple.py"

    experiments = []
    for dataset_name in datasets:
        experiments.append(
            yanex.ExperimentSpec(
                script_path=train_script,
                config={
                    "learning_rate": 0.001,
                    "epochs": 50,
                },
                # Pass dataset path as script argument
                script_args=["--dataset", dataset_name],
                name=f"train-{dataset_name}",
                tags=["multi-dataset", "batch-training"],
                description=f"Training on {dataset_name}",
            )
        )

    print(f"Created {len(experiments)} experiments for {len(datasets)} datasets")
    print(f"Datasets: {datasets}")
    print("\nThis demonstrates passing dataset paths as script arguments")

    # Uncomment to actually run:
    # results = yanex.run_multiple(experiments, parallel=2, allow_dirty=True)
    # print_results_summary(results)


def example_4_ensemble_training():
    """Example 4: Train ensemble of models with different random seeds."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Ensemble Training (Multiple Random Seeds)")
    print("=" * 60 + "\n")

    n_models = 5
    base_config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 30,
    }

    train_script = Path(__file__).parent / "train_simple.py"

    experiments = []
    for seed in range(n_models):
        config = base_config.copy()
        config["random_seed"] = seed

        experiments.append(
            yanex.ExperimentSpec(
                script_path=train_script,
                config=config,
                name=f"ensemble-model-{seed}",
                tags=["ensemble", "multi-seed"],
                description=f"Ensemble member with seed={seed}",
            )
        )

    print(f"Created {n_models} ensemble members with different random seeds")
    print(f"Base config: {base_config}")
    print("\nModels can be combined later for ensemble predictions")

    # Uncomment to actually run:
    # results = yanex.run_multiple(experiments, parallel=n_models, allow_dirty=True)
    # print_results_summary(results)


def print_results_summary(results: list):
    """Print summary of batch execution results."""
    print("\n" + "=" * 60)
    print("BATCH EXECUTION RESULTS")
    print("=" * 60)

    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    print(f"\nTotal: {len(results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    if completed:
        print("\nCompleted experiments:")
        for result in completed:
            duration_str = f"{result.duration:.2f}s" if result.duration else "N/A"
            print(f"  ✓ {result.name}: {result.experiment_id} ({duration_str})")

    if failed:
        print("\nFailed experiments:")
        for result in failed:
            print(f"  ✗ {result.name}: {result.error_message}")

    print("\nTo view experiment details:")
    if completed:
        print(f"  yanex show {completed[0].experiment_id}")
    print("  yanex list --tag <your-tag>")


def create_simple_training_script():
    """Create a simple training script for the examples.

    Users should run this first to create train_simple.py.
    """
    script_path = Path(__file__).parent / "train_simple.py"

    script_content = '''#!/usr/bin/env python3
"""Simple training script for batch execution examples."""

import argparse
import random
import time

import yanex


def train_model(learning_rate: float, epochs: int, **kwargs):
    """Simple mock training function."""
    print(f"Training with lr={learning_rate}, epochs={epochs}")

    # Simulate training
    for epoch in range(epochs):
        time.sleep(0.1)  # Simulate computation

        # Mock loss calculation
        loss = 1.0 / (epoch + 1) + random.random() * 0.1
        accuracy = min(0.95, 0.5 + epoch * 0.08 + random.random() * 0.05)

        print(f"  Epoch {epoch+1}/{epochs} - loss: {loss:.4f}, accuracy: {accuracy:.4f}")

    # Log final metrics
    final_metrics = {
        "final_loss": loss,
        "final_accuracy": accuracy,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
    final_metrics.update(kwargs)

    yanex.log_metrics(final_metrics)
    print(f"\\nTraining complete! Logged metrics to yanex.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset name")
    args = parser.parse_args()

    # Get parameters from yanex
    lr = yanex.get_param("learning_rate", default=0.001)
    epochs = yanex.get_param("epochs", default=10)
    batch_size = yanex.get_param("batch_size", default=32)
    dropout = yanex.get_param("dropout", default=0.0)
    hidden_size = yanex.get_param("hidden_size", default=128)
    random_seed = yanex.get_param("random_seed", default=42)

    print(f"Configuration:")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dropout: {dropout}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Random seed: {random_seed}")
    if args.dataset:
        print(f"  Dataset: {args.dataset}")
    print()

    # Set random seed
    random.seed(random_seed)

    # Train model
    train_model(
        learning_rate=lr,
        epochs=epochs,
        batch_size=batch_size,
        dropout=dropout,
        hidden_size=hidden_size,
        random_seed=random_seed,
        dataset=args.dataset if args.dataset else "default",
    )


if __name__ == "__main__":
    main()
'''

    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable

    print(f"\n✓ Created training script: {script_path}")
    print("  You can now run the examples above")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch execution examples with yanex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--create-script",
        action="store_true",
        help="Create train_simple.py script for the examples",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )

    args = parser.parse_args()

    if args.create_script:
        create_simple_training_script()
        return

    print("\n" + "=" * 60)
    print("YANEX BATCH EXECUTION EXAMPLES")
    print("=" * 60)

    print("\nThis script demonstrates various batch execution patterns.")
    print("The examples show the code but don't actually run experiments.")
    print("\nTo create the training script needed for examples:")
    print(f"  python {__file__} --create-script")
    print("\nThen uncomment the yanex.run_multiple() calls in each example.")

    # Run all examples (demonstrations only)
    example_1_hyperparameter_grid()
    example_2_random_search()
    example_3_multiple_datasets()
    example_4_ensemble_training()

    print("\n" + "=" * 60)
    print("For more information:")
    print("  - API documentation: yanex.run_multiple()")
    print("  - K-fold example: examples/api/kfold_training.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
