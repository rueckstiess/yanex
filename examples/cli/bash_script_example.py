#!/usr/bin/env python3
"""
Simple bash script integration example.

Run with: yanex run bash_script_example.py --param duration=8 --param threads=4
"""

from pathlib import Path

import yanex


def main():
    # Get experiment info
    exp_dir = yanex.get_experiment_dir()
    print(f"Experiment directory: {exp_dir}")

    # Execute bash script (parameters passed automatically via environment)
    script_path = Path(__file__).parent / "demo_linkbench.sh"
    result = yanex.execute_bash_script(f"{script_path} --workload mixed --verbose")

    # Log results
    yanex.log_results(
        {"exit_code": result["exit_code"], "execution_time": result["execution_time"]}
    )

    print(f"Script completed in {result['execution_time']:.1f}s")


if __name__ == "__main__":
    main()
