"""
Config Files Example

This example shows how to use YAML configuration files with yanex.
Demonstrates nested parameters and CLI overrides.
"""

import time

import yanex


def train_model(learning_rate, epochs, batch_size):
    """Simulate model training and return final metrics."""
    print(f"Training with lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")

    # Simulate training with improving loss
    loss = 1.0 / (learning_rate * 10)
    for epoch in range(epochs):
        loss *= 0.92  # Simulate improvement each epoch
        time.sleep(0.05)  # Simulate computation
        print(f"  Epoch {epoch + 1}/{epochs}: loss={loss:.4f}")

    accuracy = max(0.5, 1.0 - loss)  # Convert loss to accuracy
    return loss, accuracy


# Access nested parameters using dot notation
learning_rate = yanex.get_param("model.learning_rate", default=0.001)
epochs = yanex.get_param("training.epochs", default=10)
batch_size = yanex.get_param("training.batch_size", default=32)
# ... other params would be accessed similarly

print("Starting training...")
print(f"Dataset: {yanex.get_param('data.dataset', default='mnist')}")

# Train the model
final_loss, final_accuracy = train_model(learning_rate, epochs, batch_size)

print("\nTraining complete!")
print(f"Final loss: {final_loss:.4f}")
print(f"Final accuracy: {final_accuracy:.4f}")

# Log final metrics
yanex.log_metrics(
    {
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
)
