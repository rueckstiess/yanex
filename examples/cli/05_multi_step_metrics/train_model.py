"""
Multi-Step Metrics Example

Demonstrates logging metrics at each training step/epoch with explicit step tracking.
Shows how to build metrics incrementally using the step parameter.
"""

import random
import time

import yanex

# Get parameters
epochs = yanex.get_param("epochs", default=10)
learning_rate = yanex.get_param("learning_rate", default=0.01)
batch_size = yanex.get_param("batch_size", default=32)
val_frequency = yanex.get_param("val_frequency", default=2)  # Validate every N epochs

print(
    f"Training model for {epochs} epochs (lr={learning_rate}, batch_size={batch_size})..."
)
print(f"Validation every {val_frequency} epoch(s)")
print()

# Simulate training loop
for epoch in range(1, epochs + 1):
    # Simulate training
    time.sleep(0.3)  # Simulate computation time

    # Simulate loss decreasing and accuracy increasing
    base_loss = 2.0 * (1 - epoch / (epochs + 5))
    base_accuracy = 0.5 + 0.45 * (epoch / epochs)

    train_loss = base_loss + random.uniform(-0.1, 0.1)
    train_accuracy = base_accuracy + random.uniform(-0.05, 0.05)

    # Log training metrics for this epoch
    # Using step parameter to track which epoch these metrics belong to
    yanex.log_metrics(
        {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "learning_rate": learning_rate,
        },
        step=epoch,
    )

    # Print training progress
    print(f"Epoch {epoch}/{epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%}")

    # Validation metrics (only every N epochs)
    if epoch % val_frequency == 0:
        val_loss = train_loss + random.uniform(0, 0.2)
        val_accuracy = train_accuracy - random.uniform(0, 0.1)

        # Log validation metrics for the same epoch
        # Metrics are merged with existing metrics for this step
        yanex.log_metrics(
            {
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            },
            step=epoch,
        )

        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_accuracy:.2%}")

    print()

# Final summary metrics (no step parameter - uses auto-increment)
print("Training complete!")
print(f"Final train accuracy: {train_accuracy:.2%}")

yanex.log_metrics(
    {
        "final_train_accuracy": train_accuracy,
        "total_epochs": epochs,
    }
)
