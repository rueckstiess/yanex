#!/usr/bin/env python3
"""
K-Fold Cross-Validation Example with Yanex

This example demonstrates the orchestration/execution pattern for k-fold
cross-validation using yanex.run_multiple() API. The same script acts as
both orchestrator (spawns folds) and executor (runs single fold).

Usage:
    # Run directly - orchestration mode (spawns 5 folds sequentially)
    python examples/api/kfold_training.py

    # Run with parallel execution using yanex CLI (NEW!)
    yanex run examples/api/kfold_training.py --parallel 3

    # Or run with specific learning rate
    python examples/api/kfold_training.py --lr 0.01

Mode Detection:
    - ORCHESTRATION MODE: When _fold_idx parameter is None (default)
      The script spawns K experiments, one for each fold
    - EXECUTION MODE: When _fold_idx is set (0, 1, 2, ...)
      The script trains on that specific fold

CLI Arguments via yanex.get_cli_args():
    When run via 'yanex run', the orchestrator script can access CLI flags
    like --parallel using yanex.get_cli_args(). This allows you to control
    parallel execution without modifying the script.

This pattern allows:
    - Single script for both orchestration and execution
    - Easy parameter tuning (just change script args)
    - Automatic experiment tracking for each fold
    - Parallel or sequential execution
    - Access to CLI flags via yanex.get_cli_args()
"""

import argparse
import random
import time
from pathlib import Path

import yanex


def generate_mock_data(n_samples: int, n_folds: int, seed: int = 42):
    """Generate mock dataset for demonstration purposes.

    Returns:
        List of (features, labels) tuples representing folds
    """
    random.seed(seed)

    # Generate random data
    all_data = [(random.random(), random.random()) for _ in range(n_samples)]

    # Split into k folds
    fold_size = n_samples // n_folds
    folds = []

    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else n_samples
        folds.append(all_data[start_idx:end_idx])

    return folds


def train_fold(
    train_data: list, val_data: list, learning_rate: float, epochs: int
) -> dict:
    """Train model on one fold (mock implementation).

    Args:
        train_data: Training data for this fold
        val_data: Validation data for this fold
        learning_rate: Learning rate hyperparameter
        epochs: Number of training epochs

    Returns:
        Dictionary with training metrics
    """
    print(f"Training with {len(train_data)} samples, validating with {len(val_data)}")

    # Simulate training
    for epoch in range(epochs):
        # Mock training step
        time.sleep(0.05)  # Simulate computation

        # Mock metrics
        train_loss = 1.0 / (epoch + 1) + random.random() * 0.1
        val_loss = 1.0 / (epoch + 1) + random.random() * 0.15

        print(
            f"  Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

    # Final results
    final_train_loss = 0.1 + random.random() * 0.05
    final_val_loss = 0.15 + random.random() * 0.05
    val_accuracy = 0.85 + random.random() * 0.1

    return {
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "val_accuracy": val_accuracy,
    }


def orchestrate_kfold(
    n_folds: int, learning_rate: float, epochs: int, parallel_workers: int | None = None
):
    """Orchestrate k-fold cross-validation by spawning fold experiments.

    This function creates ExperimentSpec objects for each fold and uses
    yanex.run_multiple() to execute them in parallel.

    Args:
        n_folds: Number of folds
        learning_rate: Learning rate for all folds
        epochs: Number of epochs for all folds
        parallel_workers: Number of parallel workers (0=auto, None=sequential)
    """
    # If parallel_workers not specified, try to get it from CLI args
    if parallel_workers is None:
        cli_args = yanex.get_cli_args()
        parallel_workers = cli_args.get("parallel")
        if parallel_workers is not None:
            print(f"Using --parallel={parallel_workers} from CLI args")

    print(f"\n{'=' * 60}")
    print(f"ORCHESTRATION MODE: Spawning {n_folds}-fold cross-validation")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(
        f"Parallel workers: {parallel_workers if parallel_workers is not None else 'sequential'}"
    )
    print(f"{'=' * 60}\n")

    # Create experiment specs for each fold
    experiments = [
        yanex.ExperimentSpec(
            script_path=Path(__file__),
            config={
                "_fold_idx": i,  # Mark as execution mode
                "learning_rate": learning_rate,
                "epochs": epochs,
            },
            name=f"kfold-{i}",
            tags=["kfold", "cross-validation"],
            description=f"K-fold cross-validation fold {i}",
        )
        for i in range(n_folds)
    ]

    # Execute all folds
    print(f"Executing {n_folds} folds...")
    results = yanex.run_multiple(
        experiments, parallel=parallel_workers, allow_dirty=True
    )

    # Analyze results
    print(f"\n{'=' * 60}")
    print("K-FOLD RESULTS SUMMARY")
    print(f"{'=' * 60}")

    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    print(f"Completed: {len(completed)}/{n_folds}")
    print(f"Failed: {len(failed)}/{n_folds}")

    if completed:
        print("\nExperiment IDs:")
        for result in completed:
            print(f"  - {result.name}: {result.experiment_id}")

        # Note: In a real scenario, you would aggregate metrics from all folds
        # using yanex.results API or by reading the results directly
        print("\nTo view detailed results, use:")
        print(f"  yanex show {completed[0].experiment_id}")

    if failed:
        print("\nFailed experiments:")
        for result in failed:
            print(f"  - {result.name}: {result.error_message}")


def execute_single_fold(fold_idx: int, learning_rate: float, epochs: int, n_folds: int):
    """Execute training for a single fold.

    This function runs when the script is called with _fold_idx parameter.
    It trains the model on one fold and logs results to yanex.

    Args:
        fold_idx: Index of the fold to train (0 to n_folds-1)
        learning_rate: Learning rate hyperparameter
        epochs: Number of training epochs
        n_folds: Total number of folds
    """
    print(f"\n{'=' * 60}")
    print(f"EXECUTION MODE: Training fold {fold_idx}/{n_folds - 1}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"{'=' * 60}\n")

    # Generate mock data
    all_folds = generate_mock_data(n_samples=1000, n_folds=n_folds)

    # Use fold_idx as validation, rest as training
    val_data = all_folds[fold_idx]
    train_data = [
        sample for i, fold in enumerate(all_folds) if i != fold_idx for sample in fold
    ]

    # Train model
    results = train_fold(train_data, val_data, learning_rate, epochs)

    # Log results to yanex
    yanex.log_metrics(
        {
            "fold_idx": fold_idx,
            "train_loss": results["final_train_loss"],
            "val_loss": results["final_val_loss"],
            "val_accuracy": results["val_accuracy"],
        }
    )

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_idx} COMPLETE")
    print(f"  Train Loss: {results['final_train_loss']:.4f}")
    print(f"  Val Loss: {results['final_val_loss']:.4f}")
    print(f"  Val Accuracy: {results['val_accuracy']:.4f}")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point - detects mode and dispatches accordingly."""
    parser = argparse.ArgumentParser(
        description="K-fold cross-validation with yanex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Parallel workers (0=auto, >0=specific count, -1=sequential, default: 0)",
    )

    args = parser.parse_args()

    # Detect mode using _fold_idx parameter
    fold_idx = yanex.get_param("_fold_idx", default=None)

    if fold_idx is None:
        # ORCHESTRATION MODE: Spawn fold experiments
        parallel_workers = None if args.parallel == -1 else args.parallel
        orchestrate_kfold(args.folds, args.lr, args.epochs, parallel_workers)
    else:
        # EXECUTION MODE: Run single fold
        execute_single_fold(fold_idx, args.lr, args.epochs, args.folds)


if __name__ == "__main__":
    main()
