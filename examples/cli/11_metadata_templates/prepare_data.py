"""Prepare a small artifact and record parameters for upstream template lookup."""

import yanex

dataset = yanex.get_param("dataset", default="toy")
num_items = yanex.get_param("num_items", default=1000)

if num_items <= 0:
    raise ValueError("num_items must be positive")

yanex.save_artifact(list(range(num_items)), "data.json")
yanex.log_metrics({"items": num_items})
print(f"Prepared dataset {dataset!r} with {num_items} items")
