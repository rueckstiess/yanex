#!/usr/bin/env python3
"""
Batch Execution with Grid Search

Demonstrates using yanex.run_multiple() to execute multiple experiments
in parallel with different parameter configurations.

Run with: python grid_search.py
"""

from pathlib import Path

import yanex


def main():
    # Define parameter grid
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32]
    epochs = 10

    print("Grid Search Configuration")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Epochs: {epochs}")
    print(f"  Total experiments: {len(learning_rates) * len(batch_sizes)}")
    print()

    # Create experiment specifications
    experiments = []
    train_script = Path(__file__).parent / "train.py"

    for lr in learning_rates:
        for bs in batch_sizes:
            # Format learning rate without dots (replace . with _)
            lr_str = str(lr).replace(".", "_")

            experiments.append(
                yanex.ExperimentSpec(
                    script_path=train_script,
                    config={
                        "learning_rate": lr,
                        "batch_size": bs,
                        "epochs": epochs,
                    },
                    name=f"grid-lr{lr_str}-bs{bs}",
                    tags=["grid-search", "batch-execution"],
                    description=f"Grid search: lr={lr}, batch_size={bs}",
                )
            )

    print(f"Created {len(experiments)} experiment specifications")
    print()

    # Execute all experiments in parallel
    print("Running experiments with 3 parallel workers...")
    print()

    results = yanex.run_multiple(experiments, parallel=3, verbose=False)

    # Analyze results
    print()
    print("=" * 60)
    print("GRID SEARCH RESULTS")
    print("=" * 60)
    print()

    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    print(f"Total: {len(results)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print()

    if completed:
        print("Completed experiments:")
        for result in completed:
            duration_str = f"{result.duration:.2f}s" if result.duration else "N/A"
            print(f"  ✓ {result.name}: {result.experiment_id} ({duration_str})")
        print()

    if failed:
        print("Failed experiments:")
        for result in failed:
            print(f"  ✗ {result.name}: {result.error_message}")
        print()

    # Print how to view results
    print("To view experiment details:")
    if completed:
        print(f"  yanex show {completed[0].experiment_id}")
    print("  yanex list --tag grid-search")
    print("  yanex compare --tag grid-search")


if __name__ == "__main__":
    main()
