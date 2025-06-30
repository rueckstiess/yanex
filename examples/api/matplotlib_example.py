#!/usr/bin/env python3
"""
Example demonstrating matplotlib figure logging.

This example shows how to log matplotlib figures as artifacts.
Note: This requires matplotlib to be installed (pip install matplotlib).

Run with Python: python matplotlib_example.py
"""

from pathlib import Path

import numpy as np

import yanex


def create_training_plots(epochs, train_accuracy, val_accuracy, train_loss, val_loss):
    """Create training plots using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Install with: pip install matplotlib")
        return None, None

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot accuracy
    ax1.plot(epochs, train_accuracy, "b-", label="Training Accuracy")
    ax1.plot(epochs, val_accuracy, "r-", label="Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(epochs, train_loss, "b-", label="Training Loss")
    ax2.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Model Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # Create individual figures for separate artifacts
    acc_fig, acc_ax = plt.subplots(figsize=(8, 6))
    acc_ax.plot(epochs, train_accuracy, "b-", label="Training Accuracy", linewidth=2)
    acc_ax.plot(epochs, val_accuracy, "r-", label="Validation Accuracy", linewidth=2)
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_title("Model Accuracy Over Time")
    acc_ax.legend()
    acc_ax.grid(True, alpha=0.3)

    return fig, acc_fig


def simulate_training(epochs, learning_rate):
    """Simulate model training and generate metrics."""
    np.random.seed(42)  # For reproducible results

    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Simulate improving accuracy
        train_acc = 0.3 + (epoch / epochs) * 0.6 + np.random.normal(0, 0.02)
        val_acc = 0.25 + (epoch / epochs) * 0.65 + np.random.normal(0, 0.03)

        # Simulate decreasing loss
        train_l = 2.0 * np.exp(-epoch / 5) + np.random.normal(0, 0.1)
        val_l = 2.2 * np.exp(-epoch / 5) + np.random.normal(0, 0.12)

        train_accuracy.append(max(0, min(1, train_acc)))
        val_accuracy.append(max(0, min(1, val_acc)))
        train_loss.append(max(0, train_l))
        val_loss.append(max(0, val_l))

    return train_accuracy, val_accuracy, train_loss, val_loss


def main():
    with yanex.create_experiment(
        script_path=Path(__file__),
        name="matplotlib-plotting-example",
        config={
            "epochs": 20,
            "learning_rate": 0.001,
            "batch_size": 32,
            "model": "cnn",
            "dataset": "cifar10",
        },
        tags=["example", "plotting", "visualization"],
        description="Example demonstrating matplotlib figure logging",
        allow_dirty=True,  # Allow logging from dirty git state
    ):
        exp_id = yanex.get_experiment_id()
        print(f"Started experiment: {exp_id}")

        epochs = yanex.get_param("epochs")
        learning_rate = yanex.get_param("learning_rate")

        print(f"Simulating training for {epochs} epochs with lr={learning_rate}")

        # Simulate training process
        train_acc, val_acc, train_loss, val_loss = simulate_training(
            epochs, learning_rate
        )
        epoch_list = list(range(1, epochs + 1))

        # Log metrics for each epoch
        for i, epoch in enumerate(epoch_list):
            yanex.log_results(
                {
                    "epoch": epoch,
                    "train_accuracy": round(train_acc[i], 4),
                    "val_accuracy": round(val_acc[i], 4),
                    "train_loss": round(train_loss[i], 4),
                    "val_loss": round(val_loss[i], 4),
                },
                step=epoch,
            )

        # Create and log matplotlib figures
        try:
            combined_fig, accuracy_fig = create_training_plots(
                epoch_list, train_acc, val_acc, train_loss, val_loss
            )

            if combined_fig and accuracy_fig:
                # Log the combined training curves
                yanex.log_matplotlib_figure(
                    combined_fig, "training_curves.png", dpi=150, bbox_inches="tight"
                )
                print("Logged combined training curves plot")

                # Log just the accuracy plot
                yanex.log_matplotlib_figure(
                    accuracy_fig, "accuracy_plot.png", dpi=150, bbox_inches="tight"
                )
                print("Logged accuracy plot")

                # Create a simple scatter plot
                import matplotlib.pyplot as plt

                scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
                scatter_ax.scatter(train_acc, val_acc, alpha=0.6, s=60)
                scatter_ax.set_xlabel("Training Accuracy")
                scatter_ax.set_ylabel("Validation Accuracy")
                scatter_ax.set_title("Training vs Validation Accuracy")
                scatter_ax.plot(
                    [0, 1], [0, 1], "r--", alpha=0.5, label="Perfect Correlation"
                )
                scatter_ax.legend()
                scatter_ax.grid(True, alpha=0.3)

                yanex.log_matplotlib_figure(
                    scatter_fig, "accuracy_correlation.png", dpi=150
                )
                print("Logged accuracy correlation plot")

        except ImportError:
            print("Matplotlib not available - skipping plot generation")
            print("Install matplotlib with: pip install matplotlib")

        # Log final summary
        final_results = {
            "final_train_accuracy": round(train_acc[-1], 4),
            "final_val_accuracy": round(val_acc[-1], 4),
            "final_train_loss": round(train_loss[-1], 4),
            "final_val_loss": round(val_loss[-1], 4),
            "best_val_accuracy": round(max(val_acc), 4),
            "best_val_accuracy_epoch": val_acc.index(max(val_acc)) + 1,
        }

        yanex.log_results(final_results)

        # Create training log
        log_content = f"""Training Completed Successfully
==============================

Configuration:
- Epochs: {epochs}
- Learning Rate: {learning_rate}
- Model: {yanex.get_param("model")}
- Dataset: {yanex.get_param("dataset")}

Final Results:
- Training Accuracy: {final_results["final_train_accuracy"]:.4f}
- Validation Accuracy: {final_results["final_val_accuracy"]:.4f}
- Best Validation Accuracy: {final_results["best_val_accuracy"]:.4f} (epoch {final_results["best_val_accuracy_epoch"]})

Artifacts Generated:
- training_curves.png: Combined accuracy and loss plots
- accuracy_plot.png: Detailed accuracy progression
- accuracy_correlation.png: Training vs validation accuracy scatter plot
"""

        yanex.log_text(log_content, "training_log.txt")

        print("Training simulation completed!")
        print(f"Final validation accuracy: {final_results['final_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
