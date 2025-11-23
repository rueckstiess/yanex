"""
Tests for experiment API standalone mode functionality.
"""

from unittest.mock import patch

import pytest

import yanex
from tests.test_utils import TestDataFactory, TestFileHelpers, create_isolated_manager
from yanex.utils.exceptions import ExperimentContextError


class TestStandaloneMode:
    """Test experiment API in standalone mode (no experiment context)."""

    @pytest.mark.parametrize(
        "api_function,expected_result",
        [
            ("is_standalone", True),
            ("has_context", False),
        ],
    )
    def test_mode_detection(self, api_function, expected_result):
        """Test mode detection functions."""
        result = getattr(yanex, api_function)()
        assert result is expected_result

    def test_get_params_standalone(self):
        """Test get_params returns empty dict in standalone mode."""
        params = yanex.get_params()
        assert params == {}

    @pytest.mark.parametrize(
        "param_name,default_value,expected_result",
        [
            ("learning_rate", 0.01, 0.01),
            ("epochs", 10, 10),
            ("model_type", "linear", "linear"),
            ("batch_size", 32, 32),
            ("nonexistent", None, None),
        ],
    )
    def test_get_param_standalone(self, param_name, default_value, expected_result):
        """Test get_param returns defaults in standalone mode."""
        if default_value is not None:
            result = yanex.get_param(param_name, default_value)
        else:
            result = yanex.get_param(param_name)
        assert result == expected_result

    @pytest.mark.parametrize(
        "dot_notation_param,default_value,expected_result",
        [
            ("model.learning_rate", 0.001, 0.001),
            ("model.architecture", "resnet", "resnet"),
            ("data.batch_size", 32, 32),
            ("model.optimizer.type", "adam", "adam"),
            ("model.optimizer.lr", 1e-3, 1e-3),
            ("model.nonexistent", None, None),
            ("nonexistent.nested.path", None, None),
            ("a.b.c.d.e.f", "deep_default", "deep_default"),
        ],
    )
    def test_get_param_dot_notation_standalone(
        self, dot_notation_param, default_value, expected_result
    ):
        """Test get_param with dot notation returns defaults in standalone mode."""
        if default_value is not None:
            result = yanex.get_param(dot_notation_param, default_value)
        else:
            result = yanex.get_param(dot_notation_param)
        assert result == expected_result

    @pytest.mark.parametrize(
        "api_function,expected_result",
        [
            ("get_experiment_id", None),
            ("get_status", None),
        ],
    )
    def test_context_functions_standalone(self, api_function, expected_result):
        """Test context-dependent functions return None in standalone mode."""
        result = getattr(yanex, api_function)()
        assert result == expected_result

    def test_get_metadata_standalone(self):
        """Test get_metadata returns empty dict in standalone mode."""
        metadata = yanex.get_metadata()
        assert metadata == {}

    @pytest.mark.parametrize(
        "log_function,log_args",
        [
            ("log_metrics", ({"accuracy": 0.95, "loss": 0.05},)),
            ("log_metrics", ({"accuracy": 0.97}, {"step": 1})),
            ("log_results", ({"accuracy": 0.95, "loss": 0.05},)),
            ("log_results", ({"accuracy": 0.97}, {"step": 1})),
        ],
    )
    def test_log_metrics_standalone(self, log_function, log_args):
        """Test log_metrics and log_results are no-op in standalone mode."""
        # Should not raise any exceptions
        if len(log_args) == 2:
            getattr(yanex, log_function)(log_args[0], **log_args[1])
        else:
            getattr(yanex, log_function)(log_args[0])

    @pytest.mark.parametrize(
        "control_function,function_args,expected_error",
        [
            ("completed", (), "No active experiment context"),
            ("fail", ("test error",), "No active experiment context"),
            ("cancel", ("test cancellation",), "No active experiment context"),
        ],
    )
    def test_manual_control_functions_error(
        self, control_function, function_args, expected_error
    ):
        """Test manual control functions raise errors in standalone mode."""
        with pytest.raises(ExperimentContextError, match=expected_error):
            getattr(yanex, control_function)(*function_args)


class TestContextMode:
    """Test experiment API in context mode (with active experiment)."""

    def setup_method(self):
        """Set up test experiment context using utilities."""
        self.manager = create_isolated_manager()

        # Create test experiment using utilities
        self.experiment_id = "test12345"
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=self.experiment_id,
            status="running",
            name="test-experiment",
        )
        config = TestDataFactory.create_experiment_config(
            config_type="ml_training",
            learning_rate=0.01,
            epochs=10,
        )

        # Create experiment files
        exp_dir = self.manager.storage.experiments_dir / self.experiment_id
        TestFileHelpers.create_experiment_files(exp_dir, metadata, config)

        # Create artifacts directory
        (exp_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        # Set current experiment
        yanex._set_current_experiment_id(self.experiment_id)

    def teardown_method(self):
        """Clean up after test."""
        yanex._clear_current_experiment_id()

    @pytest.mark.parametrize(
        "api_function,expected_result",
        [
            ("is_standalone", False),
            ("has_context", True),
        ],
    )
    def test_mode_detection_with_context(self, api_function, expected_result):
        """Test mode detection functions with active context."""
        result = getattr(yanex, api_function)()
        assert result is expected_result

    @patch("yanex.api._get_experiment_manager")
    def test_get_params_with_context(self, mock_get_manager):
        """Test get_params returns actual config in context mode."""
        mock_get_manager.return_value = self.manager

        params = yanex.get_params()
        assert params["learning_rate"] == 0.01
        assert params["epochs"] == 10

    @pytest.mark.parametrize(
        "param_name,expected_value,default_value",
        [
            ("learning_rate", 0.01, None),
            ("epochs", 10, None),
            ("batch_size", 32, 32),  # Non-existent param with default
        ],
    )
    @patch("yanex.api._get_experiment_manager")
    def test_get_param_with_context(
        self, mock_get_manager, param_name, expected_value, default_value
    ):
        """Test get_param returns actual values in context mode."""
        mock_get_manager.return_value = self.manager

        if default_value is not None:
            result = yanex.get_param(param_name, default_value)
        else:
            result = yanex.get_param(param_name)
        assert result == expected_value

    def test_get_experiment_id_with_context(self):
        """Test get_experiment_id returns actual ID in context mode."""
        assert yanex.get_experiment_id() == self.experiment_id

    @patch("yanex.api._get_experiment_manager")
    def test_get_status_with_context(self, mock_get_manager):
        """Test get_status returns actual status in context mode."""
        mock_get_manager.return_value = self.manager
        assert yanex.get_status() == "running"

    @pytest.mark.parametrize(
        "expected_field,expected_value",
        [
            ("id", "test12345"),
            ("status", "running"),
            ("name", "test-experiment"),
        ],
    )
    @patch("yanex.api._get_experiment_manager")
    def test_get_metadata_with_context(
        self, mock_get_manager, expected_field, expected_value
    ):
        """Test get_metadata returns actual metadata in context mode."""
        mock_get_manager.return_value = self.manager

        metadata = yanex.get_metadata()
        assert metadata[expected_field] == expected_value


class TestModeTransition:
    """Test transitioning between standalone and context modes."""

    @patch("yanex.api._get_experiment_manager")
    def test_standalone_to_context_transition(self, mock_get_manager):
        """Test transition from standalone to context mode."""
        # Start in standalone mode
        assert yanex.is_standalone() is True
        assert yanex.get_experiment_id() is None
        assert yanex.get_param("test", "default") == "default"

        # Set up experiment context using utilities
        manager = create_isolated_manager()
        mock_get_manager.return_value = manager

        experiment_id = "transition123"
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=experiment_id,
            status="running",
        )
        config = TestDataFactory.create_experiment_config(
            config_type="simple",
            test="context_value",
        )

        exp_dir = manager.storage.experiments_dir / experiment_id
        TestFileHelpers.create_experiment_files(exp_dir, metadata, config)

        try:
            # Transition to context mode
            yanex._set_current_experiment_id(experiment_id)

            # Should now be in context mode
            assert yanex.is_standalone() is False
            assert yanex.get_experiment_id() == experiment_id
            assert yanex.get_param("test", "default") == "context_value"

            # Clear context - should return to standalone
            yanex._clear_current_experiment_id()

            # Should be back in standalone mode
            assert yanex.is_standalone() is True
            assert yanex.get_experiment_id() is None
            assert yanex.get_param("test", "default") == "default"

        finally:
            # Ensure cleanup
            yanex._clear_current_experiment_id()

    @pytest.mark.parametrize(
        "experiment_config_type,test_param_name,test_param_value",
        [
            ("ml_training", "learning_rate", 0.001),
            ("data_processing", "n_docs", 5000),
            ("simple", "param1", "test_value"),
        ],
    )
    @patch("yanex.api._get_experiment_manager")
    def test_context_param_access_patterns(
        self,
        mock_get_manager,
        experiment_config_type,
        test_param_name,
        test_param_value,
    ):
        """Test various parameter access patterns in context mode."""
        manager = create_isolated_manager()
        mock_get_manager.return_value = manager

        experiment_id = f"param_test_{experiment_config_type}"
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=experiment_id,
            status="running",
        )

        # Create config with specific parameter override
        config_overrides = {test_param_name: test_param_value}
        config = TestDataFactory.create_experiment_config(
            config_type=experiment_config_type, **config_overrides
        )

        exp_dir = manager.storage.experiments_dir / experiment_id
        TestFileHelpers.create_experiment_files(exp_dir, metadata, config)

        try:
            yanex._set_current_experiment_id(experiment_id)

            # Should get the actual parameter value from context
            result = yanex.get_param(test_param_name)
            assert result == test_param_value

            # get_params should include the parameter
            all_params = yanex.get_params()
            assert test_param_name in all_params
            assert all_params[test_param_name] == test_param_value

        finally:
            yanex._clear_current_experiment_id()

    @patch("yanex.api._get_experiment_manager")
    def test_multiple_context_switches(self, mock_get_manager):
        """Test switching between multiple experiment contexts."""
        manager = create_isolated_manager()
        mock_get_manager.return_value = manager

        # Create multiple experiments with different configs
        experiments = [
            ("exp001", {"test_param": "value1"}),
            ("exp002", {"test_param": "value2"}),
            ("exp003", {"test_param": "value3"}),
        ]

        for exp_id, config_overrides in experiments:
            metadata = TestDataFactory.create_experiment_metadata(
                experiment_id=exp_id,
                status="running",
            )
            config = TestDataFactory.create_experiment_config(
                config_type="simple", **config_overrides
            )
            exp_dir = manager.storage.experiments_dir / exp_id
            TestFileHelpers.create_experiment_files(exp_dir, metadata, config)

        try:
            # Test switching between contexts
            for exp_id, config_overrides in experiments:
                yanex._set_current_experiment_id(exp_id)

                assert yanex.get_experiment_id() == exp_id
                assert yanex.get_param("test_param") == config_overrides["test_param"]
                assert not yanex.is_standalone()

                yanex._clear_current_experiment_id()

                # Should be back in standalone after clearing
                assert yanex.is_standalone()
                assert yanex.get_experiment_id() is None

        finally:
            yanex._clear_current_experiment_id()

    def test_error_handling_edge_cases(self):
        """Test error handling in edge cases for mode transitions."""
        # Ensure we start clean
        yanex._clear_current_experiment_id()

        # Multiple clears should not cause issues
        yanex._clear_current_experiment_id()
        yanex._clear_current_experiment_id()

        assert yanex.is_standalone() is True

        # Setting invalid experiment ID should not crash mode detection
        yanex._set_current_experiment_id("nonexistent_experiment")

        # Mode detection should still work
        assert yanex.is_standalone() is False  # Has ID set
        assert yanex.get_experiment_id() == "nonexistent_experiment"

        # Clean up
        yanex._clear_current_experiment_id()
        assert yanex.is_standalone() is True
