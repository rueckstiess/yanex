"""
Bash Script Integration Example

This example shows how to run bash scripts from Python with yanex tracking.
Parameters are automatically passed to the bash script via environment variables.
"""

from pathlib import Path

import yanex


def main():
    # Get experiment directory (where bash script will run)
    exp_dir = yanex.get_experiment_dir()
    if exp_dir:
        print(f"Experiment directory: {exp_dir}")

    # Get parameters (these will be passed to bash automatically)
    duration = yanex.get_param("duration", default=5)
    threads = yanex.get_param("threads", default=2)

    print(f"Running LinkBench with duration={duration}s, threads={threads}")

    # Execute bash script
    # Parameters automatically available as YANEX_PARAM_* environment variables
    # Experiment ID available as YANEX_EXPERIMENT_ID
    script_path = Path(__file__).parent / "linkbench.sh"

    result = yanex.execute_bash_script(
        f"{script_path} --workload mixed --verbose",
        stream_output=True,  # Print output in real-time
    )

    # Log execution metrics
    yanex.log_metrics(
        {
            "exit_code": result["exit_code"],
            "execution_time": result["execution_time"],
            "duration": duration,
            "threads": threads,
        }
    )

    print(f"\n✓ Benchmark completed in {result['execution_time']:.2f}s")
    if result["exit_code"] != 0:
        print(f"  Warning: Script exited with code {result['exit_code']}")


if __name__ == "__main__":
    main()
