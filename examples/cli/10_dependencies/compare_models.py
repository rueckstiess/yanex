"""Simulate comparing results from multiple model evaluations."""
import time
import yanex

model_a = yanex.get_dependency("model_a")
model_b = yanex.get_dependency("model_b")

print("Comparing models:")
if model_a:
    print(f"  Model A: {model_a.id} ({model_a.name})")
if model_b:
    print(f"  Model B: {model_b.id} ({model_b.name})")

time.sleep(0.2)

yanex.log_metrics({
    "model_a_id": model_a.id if model_a else "unknown",
    "model_b_id": model_b.id if model_b else "unknown",
    "winner": "model_a",
    "improvement_pct": 3.2,
})
print("Comparison complete.")
