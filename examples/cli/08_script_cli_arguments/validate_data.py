"""
Script Arguments Example

Demonstrates using argparse for script-specific flags alongside yanex parameters.
"""

import argparse
import random

import yanex


def main():
    # Parse script-specific arguments
    parser = argparse.ArgumentParser(description="Data validation tool")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed validation output"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "table"],
        default="table",
        help="Output format for results",
    )
    parser.add_argument(
        "--check-type",
        choices=["basic", "thorough"],
        default="basic",
        help="Type of validation to perform",
    )
    args = parser.parse_args()

    # Get yanex parameters (experiment configuration)
    sample_size = yanex.get_param("sample_size", default=100)
    error_threshold = yanex.get_param("error_threshold", default=0.05)

    # Run validation
    print(f"Running {args.check_type} validation on {sample_size} samples...")
    if args.verbose:
        print(f"  Error threshold: {error_threshold}")
        print(f"  Output format: {args.format}")

    # Simulate validation
    errors_found = random.randint(0, int(sample_size * 0.1))
    error_rate = errors_found / sample_size

    # Output results based on format
    if args.format == "json":
        print(
            f'{{"errors": {errors_found}, "total": {sample_size}, "rate": {error_rate:.3f}}}'
        )
    elif args.format == "csv":
        print(f"errors,total,rate\n{errors_found},{sample_size},{error_rate:.3f}")
    else:  # table
        print("\nValidation Results:")
        print(f"  Errors found: {errors_found}/{sample_size}")
        print(f"  Error rate: {error_rate:.1%}")

    # Check threshold
    passed = error_rate <= error_threshold
    status = "PASSED" if passed else "FAILED"
    print(f"\nValidation {status} (threshold: {error_threshold:.1%})")

    # Log metrics
    yanex.log_metrics(
        {
            "sample_size": sample_size,
            "errors_found": errors_found,
            "error_rate": error_rate,
            "threshold": error_threshold,
            "passed": passed,
        }
    )


if __name__ == "__main__":
    main()
