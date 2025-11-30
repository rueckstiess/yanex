"""Tests for custom artifact format handler registration."""

from pathlib import Path

import pytest

import yanex
from yanex.core.artifact_formats import FORMAT_HANDLERS, register_format


@pytest.fixture(autouse=True)
def cleanup_format_handlers():
    """Restore FORMAT_HANDLERS to original state after each test.

    This prevents custom handlers registered in one test from affecting other tests,
    which is especially important when tests run in parallel.
    """
    original_handlers = FORMAT_HANDLERS.copy()
    yield
    FORMAT_HANDLERS.clear()
    FORMAT_HANDLERS.extend(original_handlers)


class CustomObject:
    """Test custom object with save/load methods."""

    def __init__(self, data: str):
        self.data = data

    def save(self, path: str | Path) -> None:
        """Save to file."""
        Path(path).write_text(self.data, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CustomObject":
        """Load from file."""
        data = Path(path).read_text(encoding="utf-8")
        return cls(data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CustomObject):
            return False
        return self.data == other.data


class TestFormatRegistration:
    """Test format registration functionality."""

    def test_register_custom_format(self, temp_dir):
        """Test registering a custom format handler."""
        initial_count = len(FORMAT_HANDLERS)

        # Register custom format
        register_format(
            name="custom",
            extensions=[".custom"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: obj.save(path),
            loader=lambda path: CustomObject.load(path),
        )

        # Verify handler was added
        assert len(FORMAT_HANDLERS) == initial_count + 1
        assert FORMAT_HANDLERS[0].name == "custom"  # Prepended to list

    def test_register_format_duplicate_name_raises_error(self, temp_dir):
        """Test that registering duplicate format name raises ValueError."""
        # Register first format
        register_format(
            name="unique_format",
            extensions=[".uf"],
            type_check=lambda obj: isinstance(obj, str),
            saver=lambda obj, path: Path(path).write_text(obj),
            loader=lambda path: Path(path).read_text(),
        )

        # Try to register same name again
        with pytest.raises(ValueError, match="already registered"):
            register_format(
                name="unique_format",
                extensions=[".uf2"],
                type_check=lambda obj: isinstance(obj, str),
                saver=lambda obj, path: Path(path).write_text(obj),
                loader=lambda path: Path(path).read_text(),
            )

    def test_register_format_no_extensions_raises_error(self, temp_dir):
        """Test that registering with no extensions raises ValueError."""
        with pytest.raises(ValueError, match="At least one extension"):
            register_format(
                name="no_ext",
                extensions=[],
                type_check=lambda obj: isinstance(obj, str),
                saver=lambda obj, path: Path(path).write_text(obj),
                loader=lambda path: Path(path).read_text(),
            )


class TestCustomHandlerSaveLoad:
    """Test save/load with custom handlers."""

    def test_save_and_load_custom_object(self, temp_dir):
        """Test save/load roundtrip with custom object."""
        # Register custom format
        register_format(
            name="customobj",
            extensions=[".customobj"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: obj.save(path),
            loader=lambda path: CustomObject.load(path),
        )

        # Create custom object
        obj = CustomObject("test data 123")

        # Save using yanex API
        yanex.save_artifact(obj, "data.customobj")

        # Load using yanex API with explicit format
        loaded = yanex.load_artifact("data.customobj", format="customobj")

        # Verify
        assert loaded == obj
        assert loaded.data == "test data 123"

    def test_save_custom_load_auto_detect(self, temp_dir):
        """Test that auto-detection works after registration."""
        # Register custom format
        register_format(
            name="autodetect",
            extensions=[".auto"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: obj.save(path),
            loader=lambda path: CustomObject.load(path),
        )

        # Save
        obj = CustomObject("auto detect test")
        yanex.save_artifact(obj, "data.auto")

        # Load without format parameter (auto-detect)
        loaded = yanex.load_artifact("data.auto")

        # Verify
        assert loaded == obj

    def test_explicit_format_overrides_extension(self, temp_dir):
        """Test that explicit format parameter overrides extension-based detection."""
        # Register two formats for same extension
        # Both can parse the same file format but return different types
        register_format(
            name="format1",
            extensions=[".shared"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: Path(path).write_text(obj.data),
            loader=lambda path: CustomObject(Path(path).read_text()),
        )

        register_format(
            name="format2",
            extensions=[".shared"],
            type_check=lambda obj: isinstance(obj, dict) and "format2" in obj,
            saver=lambda obj, path: Path(path).write_text(obj["data"]),
            loader=lambda path: {"data": Path(path).read_text(), "format2": True},
        )

        # Save with format1
        obj = CustomObject("test")
        yanex.save_artifact(obj, "data.shared")

        # Load with explicit format1 - returns CustomObject
        loaded1 = yanex.load_artifact("data.shared", format="format1")
        assert isinstance(loaded1, CustomObject)
        assert loaded1.data == "test"

        # Load with explicit format2 - returns dict (different type from same file)
        loaded2 = yanex.load_artifact("data.shared", format="format2")
        assert isinstance(loaded2, dict)
        assert loaded2["data"] == "test"
        assert loaded2["format2"] is True

    def test_custom_handler_priority_over_builtin(self, temp_dir):
        """Test that custom handlers have priority over built-in handlers."""
        # Register custom handler for .jsonl
        register_format(
            name="custom_jsonl",
            extensions=[".jsonl"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: obj.save(path),
            loader=lambda path: CustomObject.load(path),
        )

        # Save CustomObject as .jsonl (should use custom handler, not built-in)
        obj = CustomObject("custom jsonl data")
        yanex.save_artifact(obj, "data.jsonl")

        # Load with explicit format
        loaded = yanex.load_artifact("data.jsonl", format="custom_jsonl")
        assert loaded == obj

        # Built-in jsonl handler should still work for lists
        list_data = [{"a": 1}, {"b": 2}]
        yanex.save_artifact(list_data, "list.jsonl")
        loaded_list = yanex.load_artifact("list.jsonl", format="jsonl")
        assert loaded_list == list_data


class TestFormatParameter:
    """Test the format parameter in load_artifact."""

    def test_unknown_format_raises_error(self, temp_dir):
        """Test that unknown format raises ValueError."""
        # Create a file
        test_file = temp_dir / "artifacts" / "test.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("test")

        # Change to temp_dir so file is found
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Try to load with unknown format
            with pytest.raises(ValueError, match="Unknown format"):
                yanex.load_artifact("test.txt", format="nonexistent")
        finally:
            os.chdir(original_dir)

    def test_format_parameter_with_builtin_formats(self, temp_dir):
        """Test using format parameter with built-in formats."""
        # Save JSON data
        data = {"key": "value"}
        yanex.save_artifact(data, "data.json")

        # Load with explicit format
        loaded = yanex.load_artifact("data.json", format="json")
        assert loaded == data

    def test_ambiguous_extension_requires_format(self, temp_dir):
        """Test handling of ambiguous extensions (multiple handlers)."""
        # Register custom handler for .txt
        register_format(
            name="custom_text",
            extensions=[".txt"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: obj.save(path),
            loader=lambda path: CustomObject.load(path),
        )

        # Save with custom handler
        obj = CustomObject("custom text")
        yanex.save_artifact(obj, "custom.txt")

        # Load with explicit custom format
        loaded_custom = yanex.load_artifact("custom.txt", format="custom_text")
        assert isinstance(loaded_custom, CustomObject)
        assert loaded_custom.data == "custom text"

        # Load with explicit built-in text format
        loaded_builtin = yanex.load_artifact("custom.txt", format="text")
        assert isinstance(loaded_builtin, str)
        assert loaded_builtin == "custom text"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_custom_loader_takes_precedence_over_format(self, temp_dir):
        """Test that custom loader parameter takes precedence over format parameter."""
        # Register a format
        register_format(
            name="test_format",
            extensions=[".test"],
            type_check=lambda obj: isinstance(obj, str),
            saver=lambda obj, path: Path(path).write_text(obj),
            loader=lambda path: Path(path).read_text(),
        )

        # Save data
        yanex.save_artifact("test data", "file.test")

        # Custom loader that returns something different
        def custom_loader(path):
            return "custom loader result"

        # Custom loader should take precedence
        result = yanex.load_artifact(
            "file.test", loader=custom_loader, format="test_format"
        )
        assert result == "custom loader result"

    def test_nonexistent_file_returns_none(self, temp_dir):
        """Test that loading nonexistent file returns None."""
        result = yanex.load_artifact("nonexistent.txt")
        assert result is None

        # Even with explicit format
        result = yanex.load_artifact("nonexistent.txt", format="text")
        assert result is None

    def test_multiple_extensions_for_one_format(self, temp_dir):
        """Test registering multiple extensions for one format."""
        register_format(
            name="multi_ext",
            extensions=[".ext1", ".ext2", ".ext3"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: obj.save(path),
            loader=lambda path: CustomObject.load(path),
        )

        # Save and load with different extensions
        for ext in [".ext1", ".ext2", ".ext3"]:
            obj = CustomObject(f"data for {ext}")
            yanex.save_artifact(obj, f"file{ext}")
            loaded = yanex.load_artifact(f"file{ext}", format="multi_ext")
            assert loaded.data == f"data for {ext}"


class TestExperimentMode:
    """Test custom handlers in experiment mode."""

    def setup_method(self):
        """Clear CLI context to allow create_experiment() to work."""
        import os

        # Save and clear YANEX_CLI_ACTIVE to ensure non-CLI context
        self._saved_cli_active = os.environ.pop("YANEX_CLI_ACTIVE", None)

    def teardown_method(self):
        """Restore CLI context after test."""
        import os

        # Restore YANEX_CLI_ACTIVE if it was set
        if self._saved_cli_active is not None:
            os.environ["YANEX_CLI_ACTIVE"] = self._saved_cli_active

    def test_custom_handler_in_experiment_context(self, temp_dir, git_repo):
        """Test custom handlers work within experiment context."""
        # Register custom format
        register_format(
            name="exp_custom",
            extensions=[".expcustom"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: obj.save(path),
            loader=lambda path: CustomObject.load(path),
        )

        # Create experiment
        with yanex.create_experiment(script_path=Path(__file__)):
            # Save custom object
            obj = CustomObject("experiment data")
            yanex.save_artifact(obj, "data.expcustom")

            # Load with format
            loaded = yanex.load_artifact("data.expcustom", format="exp_custom")
            assert loaded == obj

    def test_format_parameter_propagates_through_storage_layer(
        self, temp_dir, git_repo
    ):
        """Test that format parameter is correctly passed through all layers."""
        # Register format
        register_format(
            name="propagate_test",
            extensions=[".prop"],
            type_check=lambda obj: isinstance(obj, CustomObject),
            saver=lambda obj, path: obj.save(path),
            loader=lambda path: CustomObject.load(path),
        )

        # Use in experiment
        with yanex.create_experiment(script_path=Path(__file__)):
            obj = CustomObject("propagation test")
            yanex.save_artifact(obj, "data.prop")

            # Load with explicit format
            loaded = yanex.load_artifact("data.prop", format="propagate_test")
            assert loaded.data == "propagation test"
