import random

import yanex

counter = yanex.get_param("counter", 10)
chance = yanex.get_param("chance", 0.3)

print(
    f"Running experiment with {counter} iterations and a {chance * 100}% success chance.\n"
)

for i in range(counter):
    if random.random() < chance:
        print(f"Iteration {i + 1}: Success!")
        yanex.log_results({"iteration": i + 1, "status": "success"}, step=i + 1)
    else:
        print(f"Iteration {i + 1}: Failure.")
        yanex.log_results({"iteration": i + 1, "status": "failure"}, step=i + 1)

print("\nAll iterations complete.")
