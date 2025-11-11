"""Dependency validation for experiments."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.exceptions import DependencyError, InvalidDependencyError
from .circular_dependency_detector import CircularDependencyDetector


class DependencyValidator:
    """Validates experiment dependencies."""

    def __init__(self, storage, manager=None):
        """Initialize validator.

        Args:
            storage: Storage interface for loading experiment data
            manager: Optional experiment manager (for extended validation)
        """
        self.storage = storage
        self.manager = manager
        self.cycle_detector = CircularDependencyDetector(storage)

    def validate_dependencies(
        self,
        experiment_id: str | None,
        declared_slots: dict[str, Any],
        resolved_deps: dict[str, str],
        check_cycles: bool = True,
    ) -> dict[str, Any]:
        """Validate all dependencies for an experiment.

        Args:
            experiment_id: ID of experiment being created (None if not yet created)
            declared_slots: Declared dependency slots (slot -> config)
            resolved_deps: Resolved dependencies (slot -> experiment_id)
            check_cycles: Whether to check for circular dependencies

        Returns:
            Validation result structure for dependencies.json

        Raises:
            DependencyError: If validation fails
        """
        checks = []
        all_valid = True

        # Normalize declared_slots to full format if needed
        normalized_slots = self._normalize_declared_slots(declared_slots)

        # Check all required slots are provided
        for slot_name, slot_config in normalized_slots.items():
            if slot_config.get("required", True):
                if slot_name not in resolved_deps:
                    raise DependencyError(f"Missing required dependency '{slot_name}'")

        # Validate each provided dependency
        for slot_name, dep_id in resolved_deps.items():
            # Check slot is declared (strict mode)
            if slot_name not in normalized_slots:
                available_slots = list(normalized_slots.keys())
                if available_slots:
                    msg = f"Unknown dependency slot '{slot_name}'\n\n"
                    msg += f"Expected slots: {', '.join(available_slots)}"
                else:
                    msg = f"Unknown dependency slot '{slot_name}'\n\n"
                    msg += "No dependencies declared in config for this script"
                raise InvalidDependencyError(msg)

            slot_config = normalized_slots[slot_name]
            expected_script = slot_config["script"]

            # Validate experiment exists (including archived experiments)
            if not self.storage.experiment_exists(dep_id, include_archived=True):
                raise DependencyError(f"Dependency experiment '{dep_id}' not found")

            # Load dependency metadata (including archived experiments)
            dep_metadata = self.storage.load_metadata(dep_id, include_archived=True)

            # Validate script matches
            actual_script = Path(dep_metadata["script_path"]).name
            script_match = actual_script == expected_script

            if not script_match:
                raise DependencyError(
                    f"Dependency '{slot_name}' requires script '{expected_script}', "
                    f"but experiment {dep_id} ran '{actual_script}'"
                )

            # Validate status - ONLY completed experiments allowed
            # This ensures reproducibility - no failed or running experiments
            dep_status = dep_metadata.get("status")
            if dep_status != "completed":
                raise DependencyError(
                    f"Dependency '{slot_name}' ({dep_id}) has status '{dep_status}'. "
                    f"Only 'completed' experiments can be dependencies. "
                    f"This ensures reproducibility."
                )

            # Record check
            check = {
                "slot": slot_name,
                "experiment_id": dep_id,
                "script_expected": expected_script,
                "script_actual": actual_script,
                "script_match": script_match,
                "experiment_status": dep_status,
                "valid": script_match and dep_status == "completed",
            }
            checks.append(check)

            if not check["valid"]:
                all_valid = False

        # Check for circular dependencies
        if check_cycles and experiment_id and resolved_deps:
            self.cycle_detector.check_for_cycles(experiment_id, resolved_deps)

        return {
            "validated_at": datetime.utcnow().isoformat(),
            "status": "valid" if all_valid else "invalid",
            "checks": checks,
        }

    def _normalize_declared_slots(
        self, declared_slots: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Normalize declared slots to full format.

        Args:
            declared_slots: Either {"slot": "script.py"} or {"slot": {"script": "script.py", "required": true}}

        Returns:
            Normalized format: {"slot": {"script": "script.py", "required": true}}
        """
        normalized = {}
        for slot_name, slot_value in declared_slots.items():
            if isinstance(slot_value, str):
                # Shorthand format: "dataprep": "dataprep.py"
                normalized[slot_name] = {"script": slot_value, "required": True}
            elif isinstance(slot_value, dict):
                # Full format already
                if "script" not in slot_value:
                    raise ValueError(f"Slot '{slot_name}' missing 'script' field")
                normalized[slot_name] = {
                    "script": slot_value["script"],
                    "required": slot_value.get("required", True),
                }
            else:
                raise ValueError(f"Invalid slot configuration for '{slot_name}'")

        return normalized
