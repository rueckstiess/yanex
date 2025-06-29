import random

import yanex.experiment as experiment

counter = experiment.get_param("counter", 10)
chance = experiment.get_param("chance", 0.3)

print(
    f"Running experiment with {counter} iterations and a {chance * 100}% success chance.\n"
)

for i in range(counter):
    if random.random() < chance:
        print(f"Iteration {i + 1}: Success!")
        experiment.log_results({"iteration": i + 1, "status": "success"}, step=i + 1)
    else:
        print(f"Iteration {i + 1}: Failure.")
        experiment.log_results({"iteration": i + 1, "status": "failure"}, step=i + 1)

print("\nAll iterations complete.")
