"""
Tests for core constants.
"""

from yanex.core.constants import EXPERIMENT_STATUSES, EXPERIMENT_STATUSES_SET


class TestExperimentStatuses:
    """Test experiment status constants."""

    def test_experiment_statuses_list(self):
        """Test EXPERIMENT_STATUSES list contains expected statuses."""
        expected_statuses = [
            "created",
            "running", 
            "completed",
            "failed",
            "cancelled",
            "staged",
        ]
        
        assert EXPERIMENT_STATUSES == expected_statuses

    def test_experiment_statuses_set(self):
        """Test EXPERIMENT_STATUSES_SET matches the list."""
        expected_set = set(EXPERIMENT_STATUSES)
        assert EXPERIMENT_STATUSES_SET == expected_set

    def test_staged_status_included(self):
        """Test staged status is included in both list and set."""
        assert "staged" in EXPERIMENT_STATUSES
        assert "staged" in EXPERIMENT_STATUSES_SET

    def test_all_expected_statuses_present(self):
        """Test all expected statuses are present."""
        required_statuses = {
            "created", "running", "completed", 
            "failed", "cancelled", "staged"
        }
        
        assert required_statuses.issubset(EXPERIMENT_STATUSES_SET)
        
    def test_no_duplicate_statuses(self):
        """Test no duplicate statuses in the list."""
        assert len(EXPERIMENT_STATUSES) == len(set(EXPERIMENT_STATUSES))

    def test_statuses_are_strings(self):
        """Test all statuses are strings."""
        for status in EXPERIMENT_STATUSES:
            assert isinstance(status, str)
            assert len(status) > 0