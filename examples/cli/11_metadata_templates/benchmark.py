"""Tiny batch-processing benchmark for metadata template examples.

Works standalone or with yanex. With dependencies, reads the prepared data
artifact; otherwise uses a small built-in dataset. No database is required.
"""

import time

import yanex

batch_size = yanex.get_param("batch_size", default=50)
multiplier = yanex.get_param("multiplier", default=1.0)
if batch_size <= 0:
    raise ValueError("batch_size must be positive")

if not yanex.is_standalone() and yanex.get_dependencies():
    data = yanex.load_artifact("data.json")
else:
    data = list(range(1000))

start = time.perf_counter()
checksum = 0.0
batches = 0
for offset in range(0, len(data), batch_size):
    batch = data[offset : offset + batch_size]
    checksum += sum(value * multiplier for value in batch)
    batches += 1
elapsed = time.perf_counter() - start

yanex.log_metrics(
    {
        "items": len(data),
        "batches": batches,
        "checksum": checksum,
        "elapsed_seconds": elapsed,
    }
)
print(f"Processed {len(data)} items in {batches} batches (batch_size={batch_size})")
print(f"Checksum: {checksum:.2f}; elapsed: {elapsed:.6f}s")
