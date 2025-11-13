#!/usr/bin/env python3
"""
Creating Experiments Programmatically

Demonstrates creating experiments using yanex.create_experiment() context manager.
Shows how to log metrics and access the experiment directory.

Run with: python process_data.py
"""

import time
from pathlib import Path

import yanex


def process_dataset(dataset_size, chunk_size):
    """Simulate data processing."""
    num_chunks = dataset_size // chunk_size
    processed_items = 0

    print(f"Processing {dataset_size} items in chunks of {chunk_size}...")

    for i in range(num_chunks):
        # Simulate processing
        time.sleep(0.1)
        processed_items += chunk_size

        # Log progress
        yanex.log_metrics(
            {"chunk": i + 1, "items_processed": processed_items}, step=i + 1
        )

        print(f"  Chunk {i + 1}/{num_chunks}: processed {processed_items} items")

    return processed_items


def main():
    # Create experiment programmatically
    with yanex.create_experiment(
        script_path=Path(__file__),
        name="data-processing-example",
        config={
            "dataset_size": 1000,
            "chunk_size": 200,
            "output_format": "json",
        },
        tags=["data-processing", "example"],
        description="Example demonstrating programmatic experiment creation",
    ):
        exp_id = yanex.get_experiment_id()
        exp_dir = yanex.get_experiment_dir()

        print(f"Started experiment: {exp_id}")
        print(f"Experiment directory: {exp_dir}")
        print()

        # Get parameters
        dataset_size = yanex.get_param("dataset_size")
        chunk_size = yanex.get_param("chunk_size")
        output_format = yanex.get_param("output_format")

        # Process data
        total_processed = process_dataset(dataset_size, chunk_size)

        # Log final results
        yanex.log_metrics(
            {
                "total_processed": total_processed,
                "num_chunks": dataset_size // chunk_size,
                "output_format": output_format,
            }
        )

        print("\nProcessing complete!")
        print(f"Total items processed: {total_processed}")
        print(f"\nExperiment results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
