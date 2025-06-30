#!/usr/bin/env python3
"""
Demo script showing yanex dual-mode functionality.

This script works identically in both modes:
1. Standalone: python demo_dual_mode.py
2. Yanex CLI: yanex run demo_dual_mode.py [options]

The script will clearly show which mode it's running in and demonstrate
all the key API features working seamlessly in both contexts.
"""

import time
from pathlib import Path
import yanex


def simulate_training(learning_rate, epochs, model_type):
    """Simulate a machine learning training process."""
    print(f"\n🚀 Starting {model_type} model training...")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Epochs: {epochs}")

    results = []

    for epoch in range(1, epochs + 1):
        # Simulate training with improving metrics
        base_accuracy = 0.6
        base_loss = 1.0

        # Add some realistic progression
        progress = epoch / epochs
        noise = 0.05 * (0.5 - abs(epoch % 2 - 0.5))  # Small variations

        accuracy = base_accuracy + (0.3 * progress) + noise
        loss = base_loss * (1 - 0.7 * progress) + noise

        # Clamp values to realistic ranges
        accuracy = max(0.1, min(0.98, accuracy))
        loss = max(0.05, min(2.0, loss))

        print(f"   Epoch {epoch:2d}: accuracy={accuracy:.3f}, loss={loss:.3f}")

        # Log results - works in both modes!
        yanex.log_results(
            {
                "epoch": epoch,
                "accuracy": accuracy,
                "loss": loss,
                "learning_rate": learning_rate,
                "model_type": model_type,
            },
            step=epoch,
        )

        results.append({"epoch": epoch, "accuracy": accuracy, "loss": loss})

        # Simulate training time
        time.sleep(0.1)

    return results


def main():
    print("=" * 60)
    print("🧪 YANEX DUAL-MODE DEMONSTRATION")
    print("=" * 60)

    # 🔍 MODE DETECTION - Key feature!
    print(f"\n📊 Mode Detection:")
    print(f"   is_standalone(): {yanex.is_standalone()}")
    print(f"   has_context():   {yanex.has_context()}")

    if yanex.is_standalone():
        print("   🔧 Running in STANDALONE mode")
        print("   📝 Parameters will use defaults")
        print("   📋 Logging will be silent (no tracking)")
    else:
        print("   🎯 Running in YANEX EXPERIMENT mode")
        print(f"   📝 Experiment ID: {yanex.get_experiment_id()}")
        print(f"   📋 Status: {yanex.get_status()}")

    # 🔧 PARAMETER ACCESS - Works seamlessly in both modes
    print(f"\n🔧 Parameter Access:")
    learning_rate = yanex.get_param("learning_rate", 0.01)
    epochs = yanex.get_param("epochs", 5)
    model_type = yanex.get_param("model_type", "neural_network")
    batch_size = yanex.get_param("batch_size", 32)

    print(
        f"   learning_rate: {learning_rate} {'(from yanex)' if not yanex.is_standalone() else '(default)'}"
    )
    print(
        f"   epochs:        {epochs} {'(from yanex)' if not yanex.is_standalone() else '(default)'}"
    )
    print(
        f"   model_type:    {model_type} {'(from yanex)' if not yanex.is_standalone() else '(default)'}"
    )
    print(
        f"   batch_size:    {batch_size} {'(from yanex)' if not yanex.is_standalone() else '(default)'}"
    )

    # Show all available params
    all_params = yanex.get_params()
    if all_params:
        print(f"   📋 All params: {all_params}")
    else:
        print(f"   📋 All params: {{}} (standalone mode)")

    # 🏃 SIMULATION
    results = simulate_training(learning_rate, epochs, model_type)

    # 📊 ADDITIONAL LOGGING - Demonstrates various logging functions
    print(f"\n📊 Additional Logging Examples:")

    # Log final summary
    final_accuracy = results[-1]["accuracy"]
    final_loss = results[-1]["loss"]

    yanex.log_results(
        {
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "total_epochs": len(results),
            "convergence_rate": (final_accuracy - results[0]["accuracy"]) / epochs,
        }
    )

    # Log text summary
    summary_text = f"""
Training Summary
================
Model Type: {model_type}
Learning Rate: {learning_rate}
Epochs: {epochs}
Batch Size: {batch_size}

Final Results:
- Accuracy: {final_accuracy:.3f}
- Loss: {final_loss:.3f}
- Improvement: {(final_accuracy - results[0]["accuracy"]):.3f}

Training completed successfully!
"""

    yanex.log_text(summary_text.strip(), "training_summary.txt")

    # Log config as artifact (if in yanex mode)
    if not yanex.is_standalone():
        print("   📄 Logged training summary to training_summary.txt")
        print("   📊 Logged final results and metrics")
    else:
        print("   📄 Would log training summary (standalone mode)")
        print("   📊 Would log results and metrics (standalone mode)")

    # 🎯 EXPERIMENT INFO
    print(f"\n🎯 Experiment Information:")
    if yanex.is_standalone():
        print("   📋 No experiment tracking (standalone mode)")
        print("   💾 Results not saved")
        print("   🔍 No experiment ID")
    else:
        metadata = yanex.get_metadata()
        print(f"   📋 Experiment ID: {yanex.get_experiment_id()}")
        print(f"   📊 Status: {yanex.get_status()}")
        if metadata.get("name"):
            print(f"   🏷️  Name: {metadata.get('name')}")
        if metadata.get("tags"):
            print(f"   🏷️  Tags: {metadata.get('tags')}")
        if metadata.get("description"):
            print(f"   📝 Description: {metadata.get('description')}")
        print(
            f"   💾 Results saved to: ~/.yanex/experiments/{yanex.get_experiment_id()}"
        )

    print(f"\n✅ Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
