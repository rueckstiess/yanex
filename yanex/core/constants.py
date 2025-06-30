"""
Core constants used throughout yanex.
"""

# Valid experiment statuses
EXPERIMENT_STATUSES = [
    "created",
    "running",
    "completed",
    "failed",
    "cancelled",
    "staged",
]

# Set version for fast membership testing
EXPERIMENT_STATUSES_SET = set(EXPERIMENT_STATUSES)
