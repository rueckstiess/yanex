"""Simulate running inference / deployment benchmark."""
import time
import yanex

dep = yanex.get_dependency("model")
params = yanex.get_params()
batch_size = params.get("batch_size", 32)
num_samples = params.get("num_samples", 1000)

print(f"Running inference benchmark (batch={batch_size}, samples={num_samples})")
if dep:
    print(f"  Model from {dep.id}")

time.sleep(0.3)

throughput = num_samples / (0.3 * (64 / batch_size))
latency = 1000 / throughput

yanex.log_metrics({
    "throughput_per_sec": round(throughput, 1),
    "avg_latency_ms": round(latency, 2),
    "batch_size": batch_size,
    "num_samples": num_samples,
})
print(f"Done. Throughput: {throughput:.0f}/s, Latency: {latency:.1f}ms")
