#!/usr/bin/env python3
"""
K-Fold Cross-Validation with Orchestrator/Executor Pattern

This script acts as BOTH orchestrator (spawns folds) and executor (runs single fold).
Mode detection uses a sentinel parameter (_fold_idx).

Run with: python train_kfold.py
Or with: yanex run train_kfold.py --parallel 3
"""

import random
import time
from pathlib import Path

import yanex


def generate_mock_data(n_samples, n_folds, seed=42):
    """Generate mock dataset split into k folds."""
    random.seed(seed)
    data = [(random.random(), random.random()) for _ in range(n_samples)]

    # Split into k folds
    fold_size = n_samples // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else n_samples
        folds.append(data[start:end])

    return folds


def train_fold(train_data, val_data, learning_rate):
    """Train model on one fold."""
    print(f"Training with {len(train_data)} samples, validating with {len(val_data)}")

    # Simulate training
    time.sleep(1.0)

    # Mock metrics
    train_loss = 0.1 + random.random() * 0.05
    val_loss = 0.15 + random.random() * 0.05
    val_accuracy = 0.85 + random.random() * 0.1

    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val loss: {val_loss:.4f}")
    print(f"  Val accuracy: {val_accuracy:.4f}")

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }


def orchestrate_kfold(n_folds, learning_rate, parallel_workers=None):
    """ORCHESTRATOR MODE: Spawn experiments for each fold."""

    # If parallel_workers not specified, try to get from CLI args
    if parallel_workers is None:
        cli_args = yanex.get_cli_args()
        parallel_workers = cli_args.get("parallel")
        if parallel_workers is not None:
            print(f"Using --parallel={parallel_workers} from CLI args")

    print()
    print("=" * 60)
    print(f"ORCHESTRATOR MODE: Spawning {n_folds}-fold cross-validation")
    print(f"Learning rate: {learning_rate}")
    print(
        f"Parallel workers: {parallel_workers if parallel_workers is not None else 'sequential'}"
    )
    print("=" * 60)
    print()

    # Create experiment specs for each fold
    experiments = [
        yanex.ExperimentSpec(
            script_path=Path(__file__),
            config={
                "_fold_idx": i,  # Sentinel parameter for executor mode
                "learning_rate": learning_rate,
                "n_folds": n_folds,
            },
            name=f"kfold-{i}",
            tags=["kfold", "cross-validation"],
            description=f"K-fold cross-validation fold {i}",
        )
        for i in range(n_folds)
    ]

    # Execute all folds
    print(f"Executing {n_folds} folds...")
    results = yanex.run_multiple(experiments, parallel=parallel_workers, verbose=False)

    # Analyze results
    print()
    print("=" * 60)
    print("K-FOLD RESULTS SUMMARY")
    print("=" * 60)
    print()

    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    print(f"Completed: {len(completed)}/{n_folds}")
    print(f"Failed: {len(failed)}/{n_folds}")
    print()

    if completed:
        print("Experiment IDs:")
        for result in completed:
            duration_str = f"{result.duration:.2f}s" if result.duration else "N/A"
            print(f"  ✓ {result.name}: {result.experiment_id} ({duration_str})")
        print()

        # Aggregate metrics from completed folds using Results API
        print("Aggregating metrics from completed folds...")
        print()

        # Import Results API for reading experiment data
        import yanex.results as yr

        fold_metrics = []

        for result in completed:
            try:
                # Get experiment object
                exp = yr.get_experiment(result.experiment_id)

                # Get metrics for each fold
                train_loss = exp.get_metric("train_loss")
                val_loss = exp.get_metric("val_loss")
                val_accuracy = exp.get_metric("val_accuracy")

                # get_metric returns the value (single value if one step, list if multiple)
                # For single-step metrics like ours, extract the value
                fold_metrics.append(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                    }
                )

            except Exception as e:
                print(f"  Warning: Could not load metrics for {result.name}: {e}")

        if fold_metrics:
            # Calculate aggregated statistics
            avg_train_loss = sum(m.get("train_loss", 0) for m in fold_metrics) / len(
                fold_metrics
            )
            avg_val_loss = sum(m.get("val_loss", 0) for m in fold_metrics) / len(
                fold_metrics
            )
            avg_val_accuracy = sum(
                m.get("val_accuracy", 0) for m in fold_metrics
            ) / len(fold_metrics)

            # Calculate standard deviation for validation accuracy
            val_accuracies = [m.get("val_accuracy", 0) for m in fold_metrics]
            std_val_accuracy = (
                sum((acc - avg_val_accuracy) ** 2 for acc in val_accuracies)
                / len(val_accuracies)
            ) ** 0.5

            aggregated = {
                "n_folds": len(fold_metrics),
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "avg_val_accuracy": avg_val_accuracy,
                "std_val_accuracy": std_val_accuracy,
            }

            print(f"Aggregated Results (n={len(fold_metrics)} folds):")
            print(f"  Average train loss: {avg_train_loss:.4f}")
            print(f"  Average val loss: {avg_val_loss:.4f}")
            print(
                f"  Average val accuracy: {avg_val_accuracy:.4f} ± {std_val_accuracy:.4f}"
            )
            print()

            # If running in experiment context, log aggregated metrics
            if yanex.has_context():
                yanex.log_metrics(aggregated)
                print("✓ Logged aggregated metrics to top-level experiment")
                print(f"  Top-level experiment: {yanex.get_experiment_id()}")
                print()

        print("To view and compare individual fold results:")
        print("  yanex list --tag kfold")
        print("  yanex compare --tag kfold")

    if failed:
        print()
        print("Failed experiments:")
        for result in failed:
            print(f"  ✗ {result.name}: {result.error_message}")


def execute_single_fold(fold_idx, learning_rate, n_folds):
    """EXECUTOR MODE: Train single fold."""

    print()
    print("=" * 60)
    print(f"EXECUTOR MODE: Training fold {fold_idx}/{n_folds - 1}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    print()

    # Generate mock data
    all_folds = generate_mock_data(n_samples=1000, n_folds=n_folds)

    # Use fold_idx as validation, rest as training
    val_data = all_folds[fold_idx]
    train_data = [
        sample for i, fold in enumerate(all_folds) if i != fold_idx for sample in fold
    ]

    # Train model
    results = train_fold(train_data, val_data, learning_rate)

    # Log results
    yanex.log_metrics(
        {
            "fold_idx": fold_idx,
            "train_loss": results["train_loss"],
            "val_loss": results["val_loss"],
            "val_accuracy": results["val_accuracy"],
        }
    )

    print()
    print(f"✓ Fold {fold_idx} complete")


def main():
    # Detect mode using sentinel parameter
    fold_idx = yanex.get_param("_fold_idx", default=None)

    if fold_idx is None:
        # ORCHESTRATOR MODE
        learning_rate = 0.001
        n_folds = 5
        orchestrate_kfold(n_folds, learning_rate)

    else:
        # EXECUTOR MODE
        learning_rate = yanex.get_param("learning_rate")
        n_folds = yanex.get_param("n_folds")
        execute_single_fold(fold_idx, learning_rate, n_folds)


if __name__ == "__main__":
    main()
