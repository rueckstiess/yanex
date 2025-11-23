"""Tests for artifact format detection and handlers."""

from pathlib import Path

import pytest

from yanex.core.artifact_formats import (
    get_handler_for_load,
    get_handler_for_save,
    get_supported_extensions,
)


class TestFormatDetection:
    """Test format detection for saving and loading."""

    def test_text_format_detection(self, tmp_path):
        """Test text format detection."""
        obj = "Hello, world!"
        handler = get_handler_for_save(obj, "test.txt")
        assert ".txt" in handler.extensions

        # Save and load
        path = tmp_path / "test.txt"
        handler.saver(obj, path)
        assert path.read_text() == obj

        # Load
        load_handler = get_handler_for_load("test.txt")
        loaded = load_handler.loader(path)
        assert loaded == obj

    def test_json_format_detection(self, tmp_path):
        """Test JSON format detection."""
        obj = {"key": "value", "number": 42}
        handler = get_handler_for_save(obj, "test.json")
        assert ".json" in handler.extensions

        # Save and load
        path = tmp_path / "test.json"
        handler.saver(obj, path)

        load_handler = get_handler_for_load("test.json")
        loaded = load_handler.loader(path)
        assert loaded == obj

    def test_jsonl_format_detection(self, tmp_path):
        """Test JSON Lines format detection."""
        obj = [{"id": 1}, {"id": 2}, {"id": 3}]
        handler = get_handler_for_save(obj, "test.jsonl")
        assert ".jsonl" in handler.extensions

        # Save and load
        path = tmp_path / "test.jsonl"
        handler.saver(obj, path)

        load_handler = get_handler_for_load("test.jsonl")
        loaded = load_handler.loader(path)
        assert loaded == obj

    def test_pickle_format_detection(self, tmp_path):
        """Test pickle format detection."""
        obj = {"key": "value", "data": [1, 2, 3]}
        handler = get_handler_for_save(obj, "test.pkl")
        assert ".pkl" in handler.extensions

        # Save and load
        path = tmp_path / "test.pkl"
        handler.saver(obj, path)

        load_handler = get_handler_for_load("test.pkl")
        loaded = load_handler.loader(path)
        assert loaded == obj

    def test_unsupported_extension(self):
        """Test error for unsupported extension."""
        with pytest.raises(ValueError, match="Cannot auto-detect format"):
            get_handler_for_save("data", "test.unknown")

        with pytest.raises(ValueError, match="Cannot auto-detect format"):
            get_handler_for_load("test.unknown")

    def test_wrong_type_for_extension(self):
        """Test error when object type doesn't match extension."""
        # Try to save a dict as .txt (expects string)
        with pytest.raises(TypeError):
            handler = get_handler_for_save({"key": "value"}, "test.txt")
            handler.saver({"key": "value"}, Path("test.txt"))

    def test_supported_extensions(self):
        """Test getting list of supported extensions."""
        extensions = get_supported_extensions()
        assert ".txt" in extensions
        assert ".json" in extensions
        assert ".pkl" in extensions
        assert ".csv" in extensions
        assert ".pt" in extensions
        assert ".pth" in extensions


class TestTextHandlers:
    """Test text format handlers."""

    def test_save_and_load_text(self, tmp_path):
        """Test saving and loading text files."""
        content = "Line 1\nLine 2\nLine 3"
        path = tmp_path / "test.txt"

        handler = get_handler_for_save(content, "test.txt")
        handler.saver(content, path)

        load_handler = get_handler_for_load("test.txt")
        loaded = load_handler.loader(path)

        assert loaded == content

    def test_text_unicode(self, tmp_path):
        """Test text with unicode characters."""
        content = "Hello 世界 🌍"
        path = tmp_path / "unicode.txt"

        handler = get_handler_for_save(content, "unicode.txt")
        handler.saver(content, path)

        load_handler = get_handler_for_load("unicode.txt")
        loaded = load_handler.loader(path)

        assert loaded == content


class TestJSONHandlers:
    """Test JSON format handlers."""

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading JSON files."""
        data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        path = tmp_path / "test.json"

        handler = get_handler_for_save(data, "test.json")
        handler.saver(data, path)

        load_handler = get_handler_for_load("test.json")
        loaded = load_handler.loader(path)

        assert loaded == data

    def test_json_list(self, tmp_path):
        """Test JSON with list as root."""
        data = [1, 2, 3, 4, 5]
        path = tmp_path / "list.json"

        handler = get_handler_for_save(data, "list.json")
        handler.saver(data, path)

        load_handler = get_handler_for_load("list.json")
        loaded = load_handler.loader(path)

        assert loaded == data


class TestJSONLHandlers:
    """Test JSON Lines format handlers."""

    def test_save_and_load_jsonl(self, tmp_path):
        """Test saving and loading JSON Lines files."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]
        path = tmp_path / "test.jsonl"

        handler = get_handler_for_save(data, "test.jsonl")
        handler.saver(data, path)

        load_handler = get_handler_for_load("test.jsonl")
        loaded = load_handler.loader(path)

        assert loaded == data

    def test_empty_jsonl(self, tmp_path):
        """Test empty JSON Lines file."""
        data = []
        path = tmp_path / "empty.jsonl"

        handler = get_handler_for_save(data, "empty.jsonl")
        handler.saver(data, path)

        load_handler = get_handler_for_load("empty.jsonl")
        loaded = load_handler.loader(path)

        assert loaded == []


class TestPickleHandlers:
    """Test pickle format handlers."""

    def test_save_and_load_pickle(self, tmp_path):
        """Test saving and loading pickle files."""
        data = {
            "string": "value",
            "number": 42,
            "list": [1, 2, 3],
            "set": {1, 2, 3},
            "tuple": (1, 2, 3),
        }
        path = tmp_path / "test.pkl"

        handler = get_handler_for_save(data, "test.pkl")
        handler.saver(data, path)

        load_handler = get_handler_for_load("test.pkl")
        loaded = load_handler.loader(path)

        assert loaded["string"] == data["string"]
        assert loaded["number"] == data["number"]
        assert loaded["list"] == data["list"]
        assert loaded["set"] == data["set"]
        assert loaded["tuple"] == data["tuple"]

    def test_pickle_dict_with_complex_types(self, tmp_path):
        """Test pickling dict with complex types."""
        obj = {
            "set": {1, 2, 3},
            "tuple": (1, 2, 3),
            "nested": {"key": "value"},
        }
        path = tmp_path / "complex.pkl"

        handler = get_handler_for_save(obj, "complex.pkl")
        handler.saver(obj, path)

        load_handler = get_handler_for_load("complex.pkl")
        loaded = load_handler.loader(path)

        assert loaded["set"] == obj["set"]
        assert loaded["tuple"] == obj["tuple"]
        assert loaded["nested"] == obj["nested"]


class TestCSVHandlers:
    """Test CSV format handlers."""

    def test_csv_list_of_dicts(self, tmp_path):
        """Test CSV with list of dicts."""
        pytest.importorskip("pandas")

        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        path = tmp_path / "test.csv"

        handler = get_handler_for_save(data, "test.csv")
        handler.saver(data, path)

        # Load returns pandas DataFrame
        load_handler = get_handler_for_load("test.csv")
        loaded_df = load_handler.loader(path)

        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["name", "age"]

    def test_csv_pandas_dataframe(self, tmp_path):
        """Test CSV with pandas DataFrame."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [30, 25, 35]})
        path = tmp_path / "test.csv"

        handler = get_handler_for_save(df, "test.csv")
        handler.saver(df, path)

        load_handler = get_handler_for_load("test.csv")
        loaded_df = load_handler.loader(path)

        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["name", "age"]

    def test_csv_empty_list(self, tmp_path):
        """Test CSV with empty list."""
        pytest.importorskip("pandas")

        data = []
        path = tmp_path / "empty.csv"

        handler = get_handler_for_save(data, "empty.csv")
        handler.saver(data, path)

        # File should exist but be empty
        assert path.exists()


class TestNumpyHandlers:
    """Test NumPy format handlers."""

    def test_npy_array(self, tmp_path):
        """Test saving and loading .npy files."""
        np = pytest.importorskip("numpy")

        arr = np.array([[1, 2, 3], [4, 5, 6]])
        path = tmp_path / "test.npy"

        handler = get_handler_for_save(arr, "test.npy")
        handler.saver(arr, path)

        load_handler = get_handler_for_load("test.npy")
        loaded = load_handler.loader(path)

        assert np.array_equal(loaded, arr)

    def test_npz_dict(self, tmp_path):
        """Test saving and loading .npz files."""
        np = pytest.importorskip("numpy")

        data = {"arr1": np.array([1, 2, 3]), "arr2": np.array([4, 5, 6])}
        path = tmp_path / "test.npz"

        handler = get_handler_for_save(data, "test.npz")
        handler.saver(data, path)

        load_handler = get_handler_for_load("test.npz")
        loaded = load_handler.loader(path)

        assert np.array_equal(loaded["arr1"], data["arr1"])
        assert np.array_equal(loaded["arr2"], data["arr2"])


class TestPyTorchHandlers:
    """Test PyTorch format handlers."""

    def test_torch_tensor(self, tmp_path):
        """Test saving and loading PyTorch tensors."""
        torch = pytest.importorskip("torch")

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        path = tmp_path / "tensor.pt"

        handler = get_handler_for_save(tensor, "tensor.pt")
        handler.saver(tensor, path)

        load_handler = get_handler_for_load("tensor.pt")
        loaded = load_handler.loader(path)

        assert torch.equal(loaded, tensor)

    def test_torch_state_dict(self, tmp_path):
        """Test saving and loading PyTorch state dicts."""
        torch = pytest.importorskip("torch")

        state_dict = {
            "weight": torch.randn(10, 10),
            "bias": torch.randn(10),
        }
        path = tmp_path / "model.pt"

        handler = get_handler_for_save(state_dict, "model.pt")
        handler.saver(state_dict, path)

        load_handler = get_handler_for_load("model.pt")
        loaded = load_handler.loader(path)

        assert torch.equal(loaded["weight"], state_dict["weight"])
        assert torch.equal(loaded["bias"], state_dict["bias"])


class TestMatplotlibHandlers:
    """Test Matplotlib format handlers."""

    def test_matplotlib_figure(self, tmp_path):
        """Test saving matplotlib figures."""
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        path = tmp_path / "plot.png"

        handler = get_handler_for_save(fig, "plot.png")
        handler.saver(fig, path)

        # Verify file was created
        assert path.exists()
        assert path.stat().st_size > 0

        # Load as PIL Image
        pytest.importorskip("PIL")
        load_handler = get_handler_for_load("plot.png")
        img = load_handler.loader(path)

        assert img is not None
        plt.close(fig)
