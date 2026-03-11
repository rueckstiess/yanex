"""Simulate fine-tuning a language model."""
import time
import yanex

dep = yanex.get_dependency("tokens")
params = yanex.get_params()
model_type = params.get("model_type", "bert-base")
lr = params.get("learning_rate", 0.00003)
epochs = params.get("epochs", 3)

print(f"Fine-tuning {model_type} (lr={lr}, epochs={epochs})")
if dep:
    print(f"  Using tokenizer from {dep.id}")

time.sleep(0.2 * epochs)

accuracy = min(0.96, 0.75 + (1.0 / (lr * 10000)) * 0.05)
yanex.log_metrics({
    "model_type": model_type,
    "accuracy": round(accuracy, 4),
    "f1_score": round(accuracy * 0.98, 4),
    "final_loss": round(0.5 * (1 - accuracy), 4),
})
yanex.save_artifact({"model_type": model_type, "lr": lr}, "finetuned_model.pkl")
print(f"Done. Accuracy: {accuracy:.4f}")
