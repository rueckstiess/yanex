#!/usr/bin/env python3
"""
Example demonstrating manual experiment control.

This example shows how to manually control experiment lifecycle:
- Manual completion with experiment.completed()
- Manual failure with experiment.fail()
- Manual cancellation with experiment.cancel()
"""

import random
import time
from pathlib import Path

import yanex.experiment as experiment


def risky_computation():
    """Simulate a computation that might fail."""
    time.sleep(0.5)

    # Randomly succeed or fail
    outcome = random.choice(["success", "error", "timeout"])

    if outcome == "success":
        return {"result": "computation_complete", "value": 42}
    elif outcome == "error":
        raise ValueError("Computation failed due to invalid input")
    else:
        return None  # Timeout case


def main():
    print("Running manual control example...")

    with experiment.create_experiment(
        script_path=Path(__file__),
        name="manual-control-example",
        config={"max_retries": 3, "timeout_seconds": 1.0, "algorithm": "monte_carlo"},
        tags=["example", "manual", "control"],
        description="Example showing manual experiment control",
    ):
        exp_id = experiment.get_experiment_id()
        print(f"Started experiment: {exp_id}")

        max_retries = experiment.get_param("max_retries")

        for attempt in range(1, max_retries + 1):
            print(f"Attempt {attempt}/{max_retries}")

            try:
                result = risky_computation()

                if result is None:
                    # Timeout case
                    experiment.log_results(
                        {
                            "attempt": attempt,
                            "outcome": "timeout",
                            "status": "retrying" if attempt < max_retries else "failed",
                        }
                    )

                    if attempt == max_retries:
                        # Out of retries - manually fail the experiment
                        experiment.fail(
                            f"Computation timed out after {max_retries} attempts"
                        )
                        return

                    print("Timeout, retrying...")
                    continue

                else:
                    # Success case
                    experiment.log_results(
                        {
                            "attempt": attempt,
                            "outcome": "success",
                            "result": result["result"],
                            "value": result["value"],
                        }
                    )

                    experiment.log_text(
                        f"Successful computation result: {result}", "success_log.txt"
                    )

                    print(f"Success on attempt {attempt}: {result}")

                    # Manually mark as completed
                    experiment.completed()
                    return

            except ValueError as e:
                # Error case
                experiment.log_results(
                    {"attempt": attempt, "outcome": "error", "error_message": str(e)}
                )

                if attempt == max_retries:
                    # Out of retries - manually fail the experiment
                    experiment.fail(
                        f"Computation failed after {max_retries} attempts: {e}"
                    )
                    return

                print(f"Error on attempt {attempt}: {e}")
                print("Retrying...")
                time.sleep(0.2)

            except KeyboardInterrupt:
                # User interrupted - cancel the experiment
                experiment.cancel("User interrupted the computation")
                return


def cancellation_example():
    """Example showing manual cancellation."""
    print("\nRunning cancellation example...")
    print("Press Ctrl+C during the countdown to see cancellation in action")

    with experiment.create_experiment(
        script_path=Path(__file__),
        name="cancellation-example",
        config={"countdown_seconds": 5},
        tags=["example", "cancellation"],
        description="Example showing experiment cancellation",
    ):
        exp_id = experiment.get_experiment_id()
        print(f"Started experiment: {exp_id}")

        countdown = experiment.get_param("countdown_seconds")

        try:
            for i in range(countdown, 0, -1):
                print(f"Countdown: {i} seconds remaining...")
                experiment.log_results({"countdown": i, "status": "running"})
                time.sleep(1)

            print("Countdown completed!")
            experiment.log_results({"countdown": 0, "status": "completed"})

        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
            experiment.cancel("User requested cancellation during countdown")


if __name__ == "__main__":
    # Run both examples
    main()
    cancellation_example()
