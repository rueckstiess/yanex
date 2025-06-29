#!/usr/bin/env python3
"""
Basic example of yanex experiment tracking.

This example demonstrates the core functionality of yanex including:
- Creating experiments with configuration
- Logging results and artifacts
- Parameter access
"""

import time
from pathlib import Path

import yanex.experiment as experiment


def train_model(learning_rate, epochs):
    """Simulate model training with some results."""
    print(f"Training with lr={learning_rate}, epochs={epochs}")

    results = []
    for epoch in range(epochs):
        # Simulate training
        time.sleep(0.1)

        # Simulate metrics that improve over time
        accuracy = 0.5 + (epoch + 1) * 0.1 + (learning_rate * 10)
        loss = 1.0 - accuracy + (0.01 * epoch)

        result = {
            "accuracy": round(accuracy, 3),
            "loss": round(loss, 3),
            "epoch": epoch + 1,
        }
        results.append(result)

        print(
            f"Epoch {epoch + 1}: accuracy={result['accuracy']}, loss={result['loss']}"
        )

    return results


def main():
    # Create and run an experiment
    with experiment.create_experiment(
        script_path=Path(__file__),
        name="basic-training-example",
        config={
            "learning_rate": 0.01,
            "epochs": 5,
            "model_type": "linear",
            "dataset": "synthetic",
        },
        tags=["example", "basic", "training"],
        description="Basic training example to demonstrate yanex functionality",
    ):
        print(f"Started experiment: {experiment.get_experiment_id()}")

        # Access experiment parameters
        params = experiment.get_params()
        lr = experiment.get_param("learning_rate")
        epochs = experiment.get_param("epochs")

        print(f"Using parameters: {params}")

        # Run training
        results = train_model(lr, epochs)

        # Log results for each epoch
        for result in results:
            experiment.log_results(
                {"accuracy": result["accuracy"], "loss": result["loss"]},
                step=result["epoch"],
            )

        # Log summary results
        final_accuracy = results[-1]["accuracy"]
        final_loss = results[-1]["loss"]
        experiment.log_results(
            {
                "final_accuracy": final_accuracy,
                "final_loss": final_loss,
                "total_epochs": len(results),
            }
        )

        # Log training summary as text artifact
        summary = f"""Training Summary
================
Model Type: {experiment.get_param("model_type")}
Dataset: {experiment.get_param("dataset")}
Learning Rate: {lr}
Epochs: {epochs}
Final Accuracy: {final_accuracy}
Final Loss: {final_loss}
"""
        experiment.log_text(summary, "training_summary.txt")

        # Log results as CSV artifact
        import io

        csv_content = io.StringIO()
        csv_content.write("epoch,accuracy,loss\n")
        for result in results:
            csv_content.write(
                f"{result['epoch']},{result['accuracy']},{result['loss']}\n"
            )

        experiment.log_text(csv_content.getvalue(), "results.csv")

        print(f"Experiment completed: {experiment.get_experiment_id()}")
        print("Check ~/.yanex/experiments/ for saved results")


if __name__ == "__main__":
    main()
