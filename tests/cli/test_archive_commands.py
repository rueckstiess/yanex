"""
Tests for yanex CLI archive, delete, and unarchive commands - complete conversion to utilities.

This file replaces test_archive_commands.py with equivalent functionality using the new test utilities.
All test logic and coverage is preserved while reducing CLI test duplication significantly.
"""

import pytest

from yanex.cli.main import cli


class TestArchiveCommandsHelp:
    """Test help functionality for archive commands - improved with utilities."""

    @pytest.mark.parametrize(
        "command,expected_text",
        [
            ("archive", "Archive experiments by moving them to archived directory"),
            ("delete", "Permanently delete experiments"),
            (
                "unarchive",
                "Unarchive experiments by moving them back to experiments directory",
            ),
        ],
    )
    def test_command_help_output(self, cli_runner, command, expected_text):
        """Test help output for all archive-related commands."""
        result = cli_runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert expected_text in result.output

    def test_archive_help_options(self, cli_runner):
        """Test archive command specific help options."""
        result = cli_runner.invoke(cli, ["archive", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output
        assert "--force" in result.output

    def test_delete_help_options(self, cli_runner):
        """Test delete command specific help options."""
        result = cli_runner.invoke(cli, ["delete", "--help"])
        assert result.exit_code == 0
        assert "WARNING: This operation cannot be undone" in result.output
        assert "--archived" in result.output

    def test_unarchive_help_options(self, cli_runner):
        """Test unarchive command specific help options."""
        result = cli_runner.invoke(cli, ["unarchive", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output


class TestMutualExclusivityValidation:
    """Test mutual exclusivity validation - major improvements with utilities."""

    @pytest.mark.parametrize("command", ["archive", "delete", "unarchive"])
    def test_mutual_exclusivity_error(self, cli_runner, command):
        """Test that commands enforce mutual exclusivity between identifiers and filters."""
        # NEW: Single test covers all three commands instead of three separate tests
        result = cli_runner.invoke(cli, [command, "exp1", "--status", "completed"])
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

    @pytest.mark.parametrize("command", ["archive", "delete", "unarchive"])
    def test_no_arguments_error(self, cli_runner, command):
        """Test that commands require either identifiers or filters."""
        # NEW: Single test covers all three commands
        result = cli_runner.invoke(cli, [command])
        assert result.exit_code == 1
        assert (
            "Must specify either experiment identifiers or filter options"
            in result.output
        )


class TestNonexistentExperimentHandling:
    """Test handling of nonexistent experiments - improved with utilities."""

    @pytest.mark.parametrize(
        "command,expected_pattern",
        [
            ("archive", "No regular experiment found with ID or name 'nonexistent123'"),
            ("delete", "No regular experiment found with ID or name 'nonexistent123'"),
            (
                "unarchive",
                "No archived experiment found with ID or name 'nonexistent123'",
            ),
        ],
    )
    def test_nonexistent_experiment_error(self, cli_runner, command, expected_pattern):
        """Test handling of nonexistent experiments across all commands."""
        # NEW: Single parametrized test covers all nonexistent experiment scenarios
        result = cli_runner.invoke(cli, [command, "nonexistent123"])
        assert result.exit_code == 1
        assert expected_pattern in result.output


class TestFilterWithNoMatches:
    """Test filter operations that match no experiments - improved with utilities."""

    @pytest.mark.parametrize(
        "command,operation_name",
        [
            ("archive", "archive"),
            ("delete", "delete"),
            ("unarchive", "unarchive"),
        ],
    )
    def test_filters_no_matches(self, cli_runner, command, operation_name):
        """Test commands with filters that match no experiments."""
        # NEW: Single parametrized test covers all no-match scenarios
        result = cli_runner.invoke(
            cli, [command, "--name", "definitely-nonexistent-*", "--force"]
        )
        assert result.exit_code == 0

        # Check for appropriate "no experiments found" message
        expected_messages = [
            f"No experiments found to {operation_name}",
            f"No regular experiments found to {operation_name}",
            f"No archived experiments found to {operation_name}",
        ]

        assert any(msg in result.output for msg in expected_messages)


class TestValidArgumentPatterns:
    """Test valid argument patterns - improved with utilities."""

    @pytest.mark.parametrize("command", ["archive", "delete", "unarchive"])
    def test_accepts_identifiers_only(self, cli_runner, command):
        """Test commands accept experiment identifiers without filters."""
        # NEW: Single test validates CLI parsing for all commands
        result = cli_runner.invoke(cli, [command, "exp1", "exp2"])
        assert result.exit_code == 1  # Will fail since experiments don't exist

        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    @pytest.mark.parametrize("command", ["archive", "delete", "unarchive"])
    def test_accepts_filters_only(self, cli_runner, command):
        """Test commands accept filters without identifiers."""
        # NEW: Single test validates filter-only usage for all commands
        result = cli_runner.invoke(
            cli, [command, "--name", "definitely-nonexistent-*", "--force"]
        )
        assert result.exit_code == 0  # Should succeed but find no experiments

        # Should not show mutual exclusivity error
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )


class TestErrorMessageConsistency:
    """Test error message consistency across commands - new utility-enabled tests."""

    def test_error_message_format_consistency(self, cli_runner):
        """Test that error messages follow consistent format across commands."""
        # NEW: Systematic testing of error message patterns
        commands_and_errors = [
            # Mutual exclusivity errors
            (["archive", "exp1", "--status", "completed"], "Cannot use both"),
            (["delete", "exp1", "--status", "failed"], "Cannot use both"),
            (["unarchive", "exp1", "--status", "completed"], "Cannot use both"),
            # No arguments errors
            (["archive"], "Must specify either"),
            (["delete"], "Must specify either"),
            (["unarchive"], "Must specify either"),
        ]

        for command_args, expected_error in commands_and_errors:
            result = cli_runner.invoke(cli, command_args)
            assert result.exit_code == 1
            assert expected_error in result.output, (
                f"Command {command_args} didn't have expected error"
            )

    @pytest.mark.parametrize("command", ["archive", "delete", "unarchive"])
    def test_error_exit_codes_consistent(self, cli_runner, command):
        """Test that error exit codes are consistent across commands."""
        # Test various error scenarios
        test_cases = [
            # Mutual exclusivity
            ([command, "exp1", "--status", "completed"], 1),
            # No arguments
            ([command], 1),
            # Nonexistent experiment
            ([command, "nonexistent"], 1),
        ]

        for args, expected_code in test_cases:
            result = cli_runner.invoke(cli, args)
            assert result.exit_code == expected_code, f"Unexpected exit code for {args}"


class TestAdvancedCLIPatterns:
    """Test advanced CLI patterns enabled by utilities."""

    @pytest.mark.parametrize(
        "command,expected_operation",
        [
            ("archive", "Archive experiments"),
            ("delete", "Permanently delete experiments"),
            ("unarchive", "Unarchive experiments"),
        ],
    )
    def test_help_text_consistency(self, cli_runner, command, expected_operation):
        """Test that help text is consistent across commands."""
        result = cli_runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0
        assert expected_operation in result.output
        assert "--help" in result.output

    def test_command_combinations_matrix(self, cli_runner):
        """Test various command argument combinations systematically."""
        # NEW: Easy to test many combinations systematically
        test_combinations = [
            # (command_args, expected_exit_code, expected_text_fragment)
            (["archive"], 1, "Must specify either"),
            (["archive", "--help"], 0, "Archive experiments"),
            (["archive", "exp1", "--status", "completed"], 1, "Cannot use both"),
            (["delete"], 1, "Must specify either"),
            (["delete", "--help"], 0, "Permanently delete"),
            (["unarchive"], 1, "Must specify either"),
            (["unarchive", "--help"], 0, "Unarchive experiments"),
        ]

        for args, expected_code, expected_text in test_combinations:
            result = cli_runner.invoke(cli, args)
            assert result.exit_code == expected_code, f"Failed for args: {args}"
            assert expected_text in result.output, f"Missing text for args: {args}"

    def test_filter_option_patterns(self, cli_runner):
        """Test different filter option patterns."""
        # NEW: Systematic testing of filter combinations
        filter_scenarios = [
            # Single filter options
            (["archive", "--status", "completed", "--force"], 0),
            (["delete", "--name", "test*", "--force"], 0),
            (["unarchive", "--tag", "experimental", "--force"], 0),
            # Multiple filter combinations
            (["archive", "--status", "failed", "--name", "test*", "--force"], 0),
            (["delete", "--archived", "--name", "old*", "--force"], 0),
        ]

        for command_args, expected_code in filter_scenarios:
            result = cli_runner.invoke(cli, command_args)
            assert result.exit_code == expected_code, (
                f"Failed for filters: {command_args}"
            )

    @pytest.mark.parametrize("command", ["archive", "delete", "unarchive"])
    def test_force_flag_behavior(self, cli_runner, command):
        """Test --force flag behavior across commands."""
        # Test that --force flag is accepted and doesn't cause errors
        result = cli_runner.invoke(cli, [command, "--name", "nonexistent*", "--force"])
        assert (
            result.exit_code == 0
        )  # Should succeed even with no matches when using --force


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness - new utility-enabled tests."""

    @pytest.mark.parametrize("command", ["archive", "delete", "unarchive"])
    def test_empty_filter_patterns(self, cli_runner, command):
        """Test commands with empty or minimal filter patterns."""
        # Test edge cases that might occur in practice
        edge_cases = [
            # Very short patterns
            [command, "--name", "a", "--force"],
            [command, "--name", "*", "--force"],
            # Special characters in names
            [command, "--name", "test-*", "--force"],
        ]

        for case_args in edge_cases:
            result = cli_runner.invoke(cli, case_args)
            # Should not crash, either succeed (0) or give reasonable error
            assert result.exit_code in [0, 1], f"Unexpected crash for: {case_args}"

    def test_mixed_valid_invalid_arguments(self, cli_runner):
        """Test combinations of valid and invalid arguments."""
        # Test that invalid combinations are properly caught
        invalid_combinations = [
            # Mixing identifiers with multiple filter types
            (["archive", "exp1", "--status", "completed", "--name", "test*"], 1),
            (["delete", "exp1", "exp2", "--archived"], 1),
            # Using multiple conflicting options
            (["archive", "--name", "test*", "--help"], 0),  # --help takes precedence
        ]

        for args, expected_code in invalid_combinations:
            result = cli_runner.invoke(cli, args)
            assert result.exit_code == expected_code


class TestOriginalFunctionalityComparison:
    """Direct comparison showing improvements over original patterns."""

    def test_original_pattern_separate_methods(self, cli_runner):
        """Example of original pattern: separate test methods for each command."""
        # This represents how the original tests were structured

        # OLD WAY - would have been separate test methods:

        # test_archive_mutual_exclusivity_error
        result = cli_runner.invoke(cli, ["archive", "exp1", "--status", "completed"])
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

        # test_delete_mutual_exclusivity_error
        result = cli_runner.invoke(cli, ["delete", "exp1", "--status", "failed"])
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

        # test_unarchive_mutual_exclusivity_error
        result = cli_runner.invoke(cli, ["unarchive", "exp1", "--status", "completed"])
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

        # This pattern created 3 separate test methods in the original

    @pytest.mark.parametrize("command", ["archive", "delete", "unarchive"])
    def test_utility_pattern_parametrized(self, cli_runner, command):
        """NEW WAY: Single parametrized test covers all commands."""
        # NEW WAY - one test method covers all commands with parametrization
        result = cli_runner.invoke(cli, [command, "exp1", "--status", "completed"])
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

        # This single test method replaces 3 separate methods

    def test_setup_method_eliminated(self, cli_runner):
        """Show how utilities eliminate setup_method boilerplate."""
        # OLD WAY required setup_method in every test class:
        # def setup_method(self):
        #     self.runner = CliRunner()

        # NEW WAY: cli_runner fixture provides runner automatically
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # No setup needed, fixture handles everything


# Summary of improvements in the complete conversion:
#
# 1. **Parametrized Tests**: 67% reduction in duplicate test methods
#    - test_mutual_exclusivity_error: 1 test replaces 3 methods
#    - test_no_arguments_error: 1 test replaces 3 methods
#    - test_nonexistent_experiment_error: 1 test replaces 3 methods
#    - test_filters_no_matches: 1 test replaces 3 methods
#
# 2. **Fixture Usage**: cli_runner fixture eliminates setup_method boilerplate
#
# 3. **Systematic Testing**: Easy to test command combinations and patterns
#    - test_command_combinations_matrix: Tests 7+ scenarios in one method
#    - test_filter_option_patterns: Covers multiple filter combinations
#
# 4. **Consistency Validation**: Easy to verify consistent behavior across commands
#    - test_help_text_consistency: Validates all commands follow same patterns
#    - test_error_message_format_consistency: Ensures uniform error formatting
#
# 5. **Enhanced Coverage**: New edge case and robustness tests
#    - test_empty_filter_patterns: Tests edge cases not covered originally
#    - test_mixed_valid_invalid_arguments: Comprehensive argument validation
#
# 6. **Maintenance**: Changes to error messages only need updates in one place
#
# Overall: ~60-70% reduction in test code duplication for CLI command testing
# Additional: Enhanced test coverage with new edge cases and systematic patterns
